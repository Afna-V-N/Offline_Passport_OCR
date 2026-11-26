# src/mrz.py

"""
Offline Passport MRZ OCR
Optimized for CPU performance while preserving
MRZ validation and fallback accuracy.

Pipeline:

    Passport image
        ↓
    Fast OpenCV MRZ detection
        ↓
    MRZ crop
        ↓
    PaddleOCR
        ↓
    ICAO 9303 validation
        ↓
    Conservative OCR correction
        ↓
    Success / retake_required

No internet or external API is used.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# Maximum processing budget after OCR initialization (seconds).
TIME_BUDGET = 1.8

# Maximum image width used by OpenCV detection.
# The whole image is shrunk to this ONCE at the start of
# extract_mrz(), so every later stage (bottom regions,
# glare removal, OCR) works on a small image.
MAX_DETECTION_WIDTH = 1400

# OCR preprocessing width (fallback variants only).
MRZ_TARGET_WIDTH = 1200

# --- PaddleOCR speed settings ---------------------------------
# Mobile models are several times faster than the default
# server models on CPU and are accurate enough for MRZ text.
DET_MODEL_NAME = "PP-OCRv5_mobile_det"
REC_MODEL_NAME = "PP-OCRv5_mobile_rec"

# Cap the longest side the detector sees.
DET_LIMIT_SIDE_LEN = 1280

# oneDNN speeds up CPU inference. If it crashes on your machine,
# the code switches it off automatically and retries.
ENABLE_MKLDNN = True

CPU_THREADS = min(8, os.cpu_count() or 4)

# Set environment variable MRZ_DEBUG=1 to print what each stage
# sees and save the detected crop to debug_roi.jpg
DEBUG = os.environ.get("MRZ_DEBUG") == "1"


def _dbg(*args):
    if DEBUG:
        print("[MRZ]", *args)


# PaddleOCR object is created only once.
_PADDLE_OCR = None

# Duration of the most recent FULL OCR call (reset for every image).
# Used to decide whether another attempt fits inside TIME_BUDGET.
_LAST_FULL_OCR_SECONDS = 0.0

# Assumed cost of one full OCR call before one has been timed.
_FULL_OCR_ESTIMATE = 1.0

# Images narrower than this are upscaled before detection.
MIN_IMAGE_WIDTH = 1000


# ============================================================
# PADDLEOCR
# ============================================================

def _get_paddle_ocr():
    """
    Create PaddleOCR only once and reuse it.

    Tries the fastest configuration first and falls back to
    simpler ones if the installed PaddleOCR version does not
    support a parameter or model name.
    """

    global _PADDLE_OCR

    if _PADDLE_OCR is not None:
        return _PADDLE_OCR

    from paddleocr import PaddleOCR

    base = dict(
        lang="en",
        device="cpu",
        enable_mkldnn=ENABLE_MKLDNN,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    attempts = [
        # 1. Fast: mobile models, capped detector size, threads
        dict(
            base,
            text_detection_model_name=DET_MODEL_NAME,
            text_recognition_model_name=REC_MODEL_NAME,
            text_det_limit_side_len=DET_LIMIT_SIDE_LEN,
            text_det_limit_type="max",
            cpu_threads=CPU_THREADS,
        ),
        # 2. Same without the thread option
        dict(
            base,
            text_detection_model_name=DET_MODEL_NAME,
            text_recognition_model_name=REC_MODEL_NAME,
            text_det_limit_side_len=DET_LIMIT_SIDE_LEN,
            text_det_limit_type="max",
        ),
        # 3. Original configuration
        base,
        # 4. Minimal
        dict(lang="en", device="cpu", enable_mkldnn=False),
    ]

    last_error = None

    for kwargs in attempts:
        try:
            _PADDLE_OCR = PaddleOCR(**kwargs)
            return _PADDLE_OCR
        except Exception as error:
            last_error = error

    raise last_error


def initialize_ocr():
    """
    Load PaddleOCR and run one dummy prediction.

    Call this ONCE when the application starts. The first
    prediction after loading is much slower (graph setup),
    so warming up here keeps the first real image fast.
    """

    ocr = _get_paddle_ocr()

    try:
        rec = _get_rec_model()

        if rec is not None:
            rec.predict(
                [
                    np.full(
                        (REC_LINE_HEIGHT, REC_LINE_WIDTH, 3),
                        255,
                        dtype=np.uint8,
                    )
                ] * 2,
                batch_size=2,
            )
    except Exception as error:
        print("[MRZ] Fast recognition warm-up failed:", repr(error))

    try:
        dummy = np.full((96, 800, 3), 255, dtype=np.uint8)
        cv2.putText(
            dummy,
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
            (5, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        ocr.predict(dummy)
    except Exception:
        pass

    return ocr


# ============================================================
# ICAO 9303 CHECK DIGIT
# ============================================================

def icao_check_digit(value: str) -> str:
    """
    ICAO 9303 check digit calculation.

    Character values:
        0-9 = 0-9
        A-Z = 10-35
        <   = 0

    Repeating weights:
        7, 3, 1
    """

    weights = (7, 3, 1)
    total = 0

    for i, char in enumerate(value):

        if char == "<":
            number = 0

        elif "0" <= char <= "9":
            number = ord(char) - ord("0")

        elif "A" <= char <= "Z":
            number = ord(char) - ord("A") + 10

        else:
            number = 0

        total += number * weights[i % 3]

    return str(total % 10)


def check_digit_valid(
    value: str,
    digit: str,
) -> bool:

    if not value:
        return False

    if len(digit) != 1 or not digit.isdigit():
        return False

    return icao_check_digit(value) == digit


# ============================================================
# OCR TEXT CLEANING
# ============================================================

def clean_ocr_text(text: str) -> str:

    if not text:
        return ""

    text = text.upper()

    # Faster than repeatedly calling replace()
    # for the common whitespace characters.
    text = text.replace(" ", "")
    text = text.replace("\t", "")
    text = text.replace("\r", "")
    text = text.replace("\n", "")

    text = text.replace("|", "<")
    text = text.replace("¦", "<")
    text = text.replace("«", "<")
    text = text.replace("‹", "<")

    text = re.sub(
        r"[^A-Z0-9<]",
        "",
        text,
    )

    return text


def repair_numeric(value: str) -> str:
    """
    Repair OCR errors in fields that should contain digits.
    """

    mapping = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    }

    return "".join(
        mapping.get(c, c)
        for c in value
    )


# ============================================================
# PADDLEOCR RESULT EXTRACTION
# ============================================================

def _collect_texts(obj: Any) -> List[str]:
    """
    Extract recognized strings from different PaddleOCR
    result formats.
    """

    output = []

    if obj is None:
        return output

    if isinstance(obj, str):

        value = obj.strip()

        if value:
            output.append(value)

        return output

    if isinstance(obj, dict):

        preferred = (
            "rec_texts",
            "text",
            "texts",
            "rec_text",
            "transcription",
            "transcriptions",
        )

        for key in preferred:

            if key not in obj:
                continue

            value = obj[key]

            if isinstance(value, str):

                value = value.strip()

                if value:
                    output.append(value)

            elif isinstance(value, (list, tuple)):

                for item in value:

                    if isinstance(item, str):

                        item = item.strip()

                        if item:
                            output.append(item)

                    else:
                        output.extend(
                            _collect_texts(item)
                        )

        return output

    if isinstance(obj, (list, tuple)):

        for item in obj:
            output.extend(
                _collect_texts(item)
            )

        return output

    # PaddleOCR result object.
    try:

        json_data = obj.json

        if callable(json_data):
            json_data = json_data()

        output.extend(
            _collect_texts(json_data)
        )

    except Exception:
        pass

    return output


def _ocr(image: np.ndarray) -> List[str]:
    """
    Run PaddleOCR on the supplied MRZ crop.
    """

    global _LAST_FULL_OCR_SECONDS, ENABLE_MKLDNN, _PADDLE_OCR

    if image is None or image.size == 0:
        return []

    began = time.perf_counter()
    result = None

    for attempt in (1, 2):

        try:
            result = _get_paddle_ocr().predict(image)
            break

        except Exception as error:

            _dbg("ocr error:", repr(error))

            # oneDNN problem? Rebuild without it and retry once.
            if attempt == 1 and ENABLE_MKLDNN:
                ENABLE_MKLDNN = False
                _PADDLE_OCR = None
                continue

            return []

    _LAST_FULL_OCR_SECONDS = time.perf_counter() - began

    texts = _collect_texts(result)

    # Remove duplicates while preserving order.
    unique = []
    seen = set()

    for text in texts:

        text = text.strip()

        if not text:
            continue

        if text in seen:
            continue

        seen.add(text)
        unique.append(text)

    return unique


# ============================================================
# FAST IMAGE PREPARATION
# ============================================================

def _resize_for_detection(
    image: np.ndarray,
) -> np.ndarray:
    """
    Resize large images to make OpenCV processing fast.
    """

    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]

    if w <= MAX_DETECTION_WIDTH:
        return image

    scale = MAX_DETECTION_WIDTH / float(w)

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# FAST MRZ DETECTION
# ============================================================

def locate_mrz_roi(
    image: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Fast automatic MRZ region detection.

    Accepts either:
        - BGR image
        - grayscale image
    """

    if image is None or image.size == 0:
        return None

    # Resize only once.
    image = _resize_for_detection(image)

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    if len(image.shape) == 2:

        gray = image

        color_image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )

    else:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        color_image = image

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(gray)

    # --------------------------------------------------------
    # Blackhat
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (31, 7),
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel,
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    _, binary = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # --------------------------------------------------------
    # Connect MRZ characters
    # --------------------------------------------------------

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 5),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2,
    )

    # --------------------------------------------------------
    # Contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    h, w = gray.shape

    candidates = []

    for contour in contours:

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        if cw < w * 0.30:
            continue

        if ch < 15:
            continue

        aspect = cw / float(max(ch, 1))

        if aspect < 3.0:
            continue

        vertical_position = y / float(
            max(h, 1)
        )

        width_score = cw / float(
            max(w, 1)
        )

        lower_score = vertical_position

        score = (
            width_score * 5.0
            + lower_score * 3.0
            + min(aspect / 10.0, 1.0)
        )

        candidates.append(
            (
                score,
                x,
                y,
                cw,
                ch,
            )
        )

    if not candidates:
        return None

    # Only sorting a small candidate list.
    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    _, x, y, cw, ch = candidates[0]

    # --------------------------------------------------------
    # Expand detected text region
    # --------------------------------------------------------

    pad_x = int(cw * 0.08)
    pad_y = int(ch * 2.0)

    x1 = max(
        0,
        x - pad_x,
    )

    y1 = max(
        0,
        y - pad_y,
    )

    x2 = min(
        w,
        x + cw + pad_x,
    )

    y2 = min(
        h,
        y + ch + pad_y,
    )

    roi = color_image[
        y1:y2,
        x1:x2,
    ]

    if roi.size == 0:
        return None

    return roi


# ============================================================
# BOTTOM REGION FALLBACK
# ============================================================

def _bottom_regions(
    image: np.ndarray,
) -> List[np.ndarray]:
    """
    Generate bottom-region candidates.

    Uses relative dimensions rather than
    passport-specific coordinates.
    """

    if image is None or image.size == 0:
        return []

    h, w = image.shape[:2]

    regions = []

    for fraction in (0.30, 0.40):

        y = int(
            h * (1.0 - fraction)
        )

        region = image[
            y:h,
            0:w,
        ]

        if region.size:
            regions.append(region)

    return regions


# ============================================================
# PREPROCESS MRZ
# ============================================================

def _prepare_mrz(
    roi: np.ndarray,
) -> List[np.ndarray]:
    """
    Produce OCR-friendly variants.

    The same variants as the current implementation
    are retained to avoid reducing recognition coverage.
    """

    if roi is None or roi.size == 0:
        return []

    if len(roi.shape) == 2:

        gray = roi

    else:

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY,
        )

    h, w = gray.shape

    # Resize only when necessary.
    if w < MRZ_TARGET_WIDTH:

        scale = (
            MRZ_TARGET_WIDTH /
            float(max(w, 1))
        )

        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    # --------------------------------------------------------
    # Enhanced grayscale
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    enhanced_bgr = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2BGR,
    )

    variants = [
        enhanced_bgr
    ]

    # --------------------------------------------------------
    # Otsu
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU,
    )

    variants.append(
        cv2.cvtColor(
            otsu,
            cv2.COLOR_GRAY2BGR,
        )
    )

    return variants


# ============================================================
# GLARE
# ============================================================

def remove_glare(
    image: np.ndarray,
) -> np.ndarray:
    """
    Lightweight glare suppression.
    """

    if image is None or image.size == 0:
        return image

    if len(image.shape) == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    mask = cv2.inRange(
        hsv,
        np.array(
            [0, 0, 235],
            dtype=np.uint8,
        ),
        np.array(
            [180, 70, 255],
            dtype=np.uint8,
        ),
    )

    if cv2.countNonZero(mask) == 0:
        return image

    blurred = cv2.GaussianBlur(
        image,
        (7, 7),
        0,
    )

    result = image.copy()

    result[mask > 0] = blurred[
        mask > 0
    ]

    return result


# ============================================================
# MRZ NORMALIZATION
# ============================================================

def normalize_line1(
    text: str,
) -> str:

    text = clean_ocr_text(text)

    # Recover common OCR errors at passport prefix.
    if len(text) >= 2:

        if text[0] != "P":

            if text[0] in (
                "F",
                "R",
                "B",
                "8",
            ):

                text = (
                    "P" +
                    text[1:]
                )

        if text[1] != "<":

            if text[1] in (
                "O",
                "0",
                "Q",
                "C",
                "1",
            ):

                text = (
                    "P<" +
                    text[2:]
                )

    if len(text) > 44:
        text = text[:44]

    if len(text) < 44:
        text = text.ljust(
            44,
            "<",
        )

    return text


def normalize_line2(
    text: str,
) -> str:

    text = clean_ocr_text(text)

    if len(text) > 44:
        text = text[:44]

    if len(text) < 44:
        text = text.ljust(
            44,
            "<",
        )

    return text


# ============================================================
# MRZ VALIDATION
# ============================================================

def validate_line1(
    line1: str,
) -> bool:

    if len(line1) != 44:
        return False

    if not line1.startswith("P<"):
        return False

    if not re.fullmatch(
        r"[A-Z0-9<]{44}",
        line1,
    ):
        return False

    if not re.fullmatch(
        r"[A-Z]{3}",
        line1[2:5],
    ):
        return False

    return True


def validate_line2(
    line2: str,
) -> bool:

    if len(line2) != 44:
        return False

    if not re.fullmatch(
        r"[A-Z0-9<]{44}",
        line2,
    ):
        return False

    passport_number = line2[0:9]
    passport_cd = line2[9]

    dob = line2[13:19]
    dob_cd = line2[19]

    expiry = line2[21:27]
    expiry_cd = line2[27]

    personal_number = line2[28:42]
    personal_cd = line2[42]

    composite = (
        line2[0:10]
        + line2[13:20]
        + line2[21:43]
    )

    composite_cd = line2[43]

    return all(
        (
            check_digit_valid(
                passport_number,
                passport_cd,
            ),
            check_digit_valid(
                dob,
                dob_cd,
            ),
            check_digit_valid(
                expiry,
                expiry_cd,
            ),
            check_digit_valid(
                personal_number,
                personal_cd,
            ),
            check_digit_valid(
                composite,
                composite_cd,
            ),
        )
    )


def validate_mrz_pair(
    line1: str,
    line2: str,
) -> bool:

    return (
        validate_line1(line1)
        and
        validate_line2(line2)
    )


# ============================================================
# OCR → MRZ
# ============================================================

def _looks_like_line1(
    text: str,
) -> bool:

    text = clean_ocr_text(text)

    return (
        len(text) >= 30
        and
        (
            text.startswith("P<")
            or text.startswith("PO")
            or text.startswith("P0")
            or text.startswith("PI")
        )
    )


def _looks_like_line2(
    text: str,
) -> bool:

    text = clean_ocr_text(text)

    if len(text) < 30:
        return False

    digit_count = sum(
        c.isdigit()
        for c in text
    )

    return digit_count >= 6


def _candidate_pairs(
    texts: List[str],
) -> List[Tuple[str, str]]:

    cleaned = []

    for text in texts:

        value = clean_ocr_text(text)

        if len(value) >= 20:
            cleaned.append(value)

    candidates = []

    # --------------------------------------------------------
    # Separate OCR lines
    # --------------------------------------------------------

    for i, first in enumerate(cleaned):

        if not _looks_like_line1(first):
            continue

        for j, second in enumerate(cleaned):

            if i == j:
                continue

            if not _looks_like_line2(second):
                continue

            candidates.append(
                (
                    normalize_line1(first),
                    normalize_line2(second),
                )
            )

    # --------------------------------------------------------
    # OCR may combine both lines
    # --------------------------------------------------------

    for value in cleaned:

        if len(value) < 80:
            continue

        start = value.find("P<")

        if start < 0:
            continue

        block = value[
            start:start + 88
        ]

        if len(block) >= 88:

            candidates.append(
                (
                    normalize_line1(
                        block[:44]
                    ),
                    normalize_line2(
                        block[44:88]
                    ),
                )
            )

    return candidates


def _repair_and_validate(
    line1: str,
    line2: str,
) -> Optional[
    Tuple[str, str]
]:

    line1 = normalize_line1(
        line1
    )

    line2 = list(
        normalize_line2(
            line2
        )
    )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    for start, end in (
        (13, 20),
        (21, 28),
    ):

        value = "".join(
            line2[start:end]
        )

        value = repair_numeric(
            value
        )

        line2[start:end] = list(value)

    # --------------------------------------------------------
    # Check digits
    # --------------------------------------------------------

    for index in (
        9,
        19,
        27,
        42,
        43,
    ):

        char = line2[index]

        if char in (
            "O",
            "Q",
            "D",
        ):

            line2[index] = "0"

        elif char in (
            "I",
            "L",
        ):

            line2[index] = "1"

        elif char == "B":

            line2[index] = "8"

    line2 = "".join(line2)

    # --------------------------------------------------------
    # Final authoritative validation
    # --------------------------------------------------------

    if validate_mrz_pair(
        line1,
        line2,
    ):

        return line1, line2

    return None


def _find_valid_pair(
    texts: List[str],
) -> Optional[
    Tuple[str, str]
]:

    candidates = _candidate_pairs(
        texts
    )

    # --------------------------------------------------------
    # First: use OCR exactly as received.
    # --------------------------------------------------------

    for line1, line2 in candidates:

        if validate_mrz_pair(
            line1,
            line2,
        ):

            return line1, line2

    # --------------------------------------------------------
    # Second: conservative correction.
    # --------------------------------------------------------

    for line1, line2 in candidates:

        result = _repair_and_validate(
            line1,
            line2,
        )

        if result is not None:
            return result

    return None


# ============================================================
# RECOGNITION-ONLY FAST PATH (skips text detection)
# ============================================================

_REC_MODEL = None
_REC_DISABLED = False
_REC_FAILS = 0

# Fixed input size for the recognition model (constant shapes
# avoid re-building inference graphs for every new image).
REC_LINE_HEIGHT = 48
REC_LINE_WIDTH = 1920


def _get_rec_model():
    """
    Recognition-only model (no text detection). Returns None
    if it cannot be created, in which case the full OCR
    pipeline is used instead.
    """

    global _REC_MODEL, _REC_DISABLED

    if _REC_DISABLED:
        return None

    if _REC_MODEL is not None:
        return _REC_MODEL

    from paddleocr import TextRecognition

    attempts = [
        dict(
            model_name=REC_MODEL_NAME,
            device="cpu",
            enable_mkldnn=ENABLE_MKLDNN,
        ),
        dict(model_name=REC_MODEL_NAME, device="cpu"),
        dict(model_name=REC_MODEL_NAME),
    ]

    last_error = None

    for kwargs in attempts:
        try:
            _REC_MODEL = TextRecognition(**kwargs)
            return _REC_MODEL
        except Exception as error:
            last_error = error

    _REC_DISABLED = True
    print("[MRZ] Fast recognition unavailable, using full OCR:", repr(last_error))
    return None


def _prep_rec_line(crop: np.ndarray) -> np.ndarray:
    """
    Resize one MRZ line to a constant 48 x 1920 input,
    keeping the aspect ratio and padding on the right.
    """

    h, w = crop.shape[:2]

    new_w = max(
        1,
        int(round(w * REC_LINE_HEIGHT / float(max(h, 1)))),
    )

    if new_w >= REC_LINE_WIDTH:
        return cv2.resize(
            crop,
            (REC_LINE_WIDTH, REC_LINE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    resized = cv2.resize(
        crop,
        (new_w, REC_LINE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    return cv2.copyMakeBorder(
        resized,
        0,
        0,
        0,
        REC_LINE_WIDTH - new_w,
        cv2.BORDER_REPLICATE,
    )


def _split_mrz_lines(roi: np.ndarray) -> Optional[List[np.ndarray]]:
    """
    Cut the MRZ crop into its two text lines using a
    horizontal projection of dark pixels.

    Returns [line1_crop, line2_crop] or None.
    """

    if roi is None or roi.size == 0:
        return None

    if len(roi.shape) == 2:
        color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        gray = roi
    else:
        color = roi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    if h < 20 or w < 100:
        return None

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    rows = binary.sum(axis=1) / 255.0

    peak = float(rows.max())

    if peak <= 0:
        return None

    mask = rows > peak * 0.15

    # Find runs of text rows.
    runs = []
    start = None

    for y, on in enumerate(mask):
        if on and start is None:
            start = y
        elif not on and start is not None:
            runs.append([start, y])
            start = None

    if start is not None:
        runs.append([start, h])

    # Drop thin noise.
    runs = [r for r in runs if (r[1] - r[0]) >= 8]

    if len(runs) < 2:
        return None

    # Merge runs split by small gaps (e.g. the '<' glyphs).
    median_h = float(np.median([r[1] - r[0] for r in runs]))

    merged = [runs[0]]

    for r in runs[1:]:
        if r[0] - merged[-1][1] < median_h * 0.35:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    if len(merged) < 2:
        return None

    # MRZ lines = the two ADJACENT bands with the most ink.
    best = None
    best_ink = -1.0

    for i in range(len(merged) - 1):
        ink = float(
            rows[merged[i][0]:merged[i][1]].sum()
            + rows[merged[i + 1][0]:merged[i + 1][1]].sum()
        )

        if ink > best_ink:
            best_ink = ink
            best = (merged[i], merged[i + 1])

    top_band, bottom_band = best

    crops = []

    for y1, y2 in (top_band, bottom_band):
        pad = max(3, int((y2 - y1) * 0.25))
        y1 = max(0, y1 - pad)
        y2 = min(h, y2 + pad)
        crops.append(color[y1:y2, 0:w])

    return crops


def _fast_recognize(
    roi: np.ndarray,
) -> Optional[Tuple[str, str]]:
    """
    Try to read the MRZ with recognition only.
    Returns a validated (line1, line2) pair or None.
    """

    global _REC_FAILS, _REC_DISABLED

    model = _get_rec_model()

    if model is None:
        return None

    try:
        lines = _split_mrz_lines(roi)

        if lines is None:
            _dbg("fast path: could not split MRZ lines")
            return None

        prepared = [_prep_rec_line(line) for line in lines]

        began = time.perf_counter()

        result = model.predict(prepared, batch_size=2)

        elapsed = time.perf_counter() - began

        texts = []

        for item in result:
            texts.extend(_collect_texts(item))

        _REC_FAILS = 0

        _dbg("fast rec texts:", texts, "%.0f ms" % (elapsed * 1000))

        return _find_valid_pair(texts)

    except Exception as error:
        _REC_FAILS += 1
        _dbg("fast path error:", repr(error))

        if _REC_FAILS >= 2:
            _REC_DISABLED = True
            print("[MRZ] Fast recognition disabled after errors:", repr(error))

        return None


# ============================================================
# TIME BUDGET
# ============================================================

def _can_afford_ocr(start_time: float) -> bool:
    """
    True if another OCR call is expected to finish inside
    TIME_BUDGET (based on how long the last call took).
    """

    elapsed = time.perf_counter() - start_time

    cost = _LAST_FULL_OCR_SECONDS or _FULL_OCR_ESTIMATE

    return elapsed + cost * 1.1 < TIME_BUDGET


# ============================================================
# FAST OCR OF MRZ CROP
# ============================================================

def _ocr_mrz_crop(
    roi: np.ndarray,
    start_time: float,
) -> Optional[
    Tuple[str, str]
]:

    if roi is None or roi.size == 0:
        return None

    # ========================================================
    # FAST PATH
    # ========================================================

    # Recognition-only attempt (no text detection).
    pair = _fast_recognize(roi)

    if pair is not None:
        return pair

    # Full OCR attempt.
    texts = _ocr(roi)

    _dbg("crop", roi.shape, "ocr texts:", texts)

    if DEBUG:
        cv2.imwrite("debug_roi.jpg", roi)

    pair = _find_valid_pair(
        texts
    )

    if pair is not None:
        return pair

    # ========================================================
    # FALLBACK PATH
    # ========================================================

    if not _can_afford_ocr(start_time):
        return None

    variants = _prepare_mrz(
        roi
    )

    # Same two fallback variants, but each one only runs
    # if it can still finish inside the time budget.
    for variant in variants:

        if not _can_afford_ocr(start_time):
            break

        texts = _ocr(
            variant
        )

        _dbg("variant ocr texts:", texts)

        pair = _find_valid_pair(
            texts
        )

        if pair is not None:
            return pair

    return None


# ============================================================
# QUALITY
# ============================================================

def _blur_score(
    image: np.ndarray,
) -> float:

    if image is None or image.size == 0:
        return 0.0

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    else:

        gray = image

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )


# ============================================================
# COUNTRY / NAME CLEANUP
# ============================================================

_DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}


def _fix_country_codes(
    line1: str,
    line2: str,
) -> Tuple[str, str]:
    """
    Country codes and names are NOT covered by check digits, so
    OCR slips like 'IN0' for 'IND' pass validation unnoticed.
    Fix letters-only fields that contain digits.
    """

    issuing = line1[2:5]
    nationality = line2[10:13]

    def has_digit(value):
        return any(c.isdigit() for c in value)

    if has_digit(nationality) and issuing.isalpha():
        nationality = issuing

    elif has_digit(issuing) and nationality.isalpha():
        issuing = nationality

    else:
        issuing = "".join(
            _DIGIT_TO_LETTER.get(c, c) for c in issuing
        )
        nationality = "".join(
            _DIGIT_TO_LETTER.get(c, c) for c in nationality
        )

    names = "".join(
        _DIGIT_TO_LETTER.get(c, c) for c in line1[5:]
    )

    line1 = line1[:2] + issuing + names
    line2 = line2[:10] + nationality + line2[13:]

    return line1, line2


# ============================================================
# RESULT BUILDER
# ============================================================

def _success_result(
    pair: Tuple[str, str],
    start_time: float,
) -> Dict[str, Any]:

    pair = _fix_country_codes(pair[0], pair[1])

    elapsed = int(
        (
            time.perf_counter()
            - start_time
        ) * 1000
    )

    return {
        "status": "success",
        "line1": pair[0],
        "line2": pair[1],
        "processing_ms": elapsed,
    }


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_mrz(
    image: np.ndarray,
) -> Dict[str, Any]:
    """
    Main MRZ extraction function.

    Fast path:
        MRZ detection
        ↓
        OCR
        ↓
        validation

    Fallback:
        bottom regions
        ↓
        OCR

    Final fallback:
        glare correction
        ↓
        MRZ detection
        ↓
        OCR

    No result is accepted without ICAO validation.
    """

    global _LAST_FULL_OCR_SECONDS

    # Forget the previous image's OCR timing (stale values used to
    # make later images skip their fallbacks).
    _LAST_FULL_OCR_SECONDS = 0.0

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if image is None or image.size == 0:

        return {
            "status": "retake_required",
            "reason": "invalid_image",
            "processing_ms": 0,
        }

    h, w = image.shape[:2]

    if h < 200 or w < 200:

        return {
            "status": "retake_required",
            "reason": "image_too_small",
            "processing_ms": 0,
        }

    # Shrink once. Bottom regions, glare removal and OCR all
    # reuse this smaller image (big speed-up on phone photos).
    image = _resize_for_detection(image)

    # Small images: upscale so MRZ characters are large enough
    # for the detection kernels and OCR.
    if image.shape[1] < MIN_IMAGE_WIDTH:
        up = MIN_IMAGE_WIDTH / float(image.shape[1])
        image = cv2.resize(
            image,
            None,
            fx=up,
            fy=up,
            interpolation=cv2.INTER_CUBIC,
        )

    # ========================================================
    # STEP 1
    # FAST MRZ DETECTION
    # ========================================================

    roi = locate_mrz_roi(
        image
    )

    _dbg(
        "image", image.shape, "roi found:", roi is not None,
        "%.0f ms so far" % ((time.perf_counter() - start_time) * 1000),
    )

    if roi is not None:

        pair = _ocr_mrz_crop(
            roi,
            start_time,
        )

        if pair is not None:

            return _success_result(
                pair,
                start_time,
            )

    # ========================================================
    # STEP 2
    # BOTTOM REGION FALLBACK
    # ========================================================

    if _can_afford_ocr(start_time):

        regions = _bottom_regions(
            image
        )

        for region in regions:

            if not _can_afford_ocr(start_time):
                break

            pair = _ocr_mrz_crop(
                region,
                start_time,
            )

            if pair is not None:

                return _success_result(
                    pair,
                    start_time,
                )

    # ========================================================
    # STEP 3
    # GLARE FALLBACK
    # ========================================================

    if _can_afford_ocr(start_time):

        glare = remove_glare(
            image
        )

        roi = locate_mrz_roi(
            glare
        )

        if roi is not None:

            pair = _ocr_mrz_crop(
                roi,
                start_time,
            )

            if pair is not None:

                return _success_result(
                    pair,
                    start_time,
                )

    # ========================================================
    # FAILURE
    # ========================================================

    elapsed = int(
        (
            time.perf_counter()
            - start_time
        ) * 1000
    )

    blur = _blur_score(
        image
    )

    if blur < 25:

        reason = "image_too_blurry"

    else:

        reason = "mrz_not_detected"

    return {
        "status": "retake_required",
        "reason": reason,
        "processing_ms": elapsed,
    }


# ============================================================
# COMPATIBILITY FUNCTIONS
# ============================================================

def find_mrz_and_validate(
    image: np.ndarray,
) -> Dict[str, Any]:

    return extract_mrz(
        image
    )


def validate_mrz(
    line1: str,
    line2: str,
) -> bool:

    line1 = normalize_line1(
        line1
    )

    line2 = normalize_line2(
        line2
    )

    return validate_mrz_pair(
        line1,
        line2
    )


# ============================================================
# DIRECT INITIALIZATION
# ============================================================

if __name__ == "__main__":

    print(
        "Offline Passport MRZ module"
    )

    print(
        "Initializing PaddleOCR..."
    )

    initialize_ocr()

    print(
        "PaddleOCR initialized successfully."
    )