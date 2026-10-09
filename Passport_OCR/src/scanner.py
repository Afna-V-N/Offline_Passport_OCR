import cv2
import numpy as np

from src.mrz import locate_mrz_roi, remove_glare


# =========================================================
# MRZ detection
# =========================================================

def detect_live_mrz(frame):
    """
    Detect the MRZ region from a live camera frame.

    First tries the normal image, then a glare-corrected
    version.
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Normal MRZ detection
    roi = locate_mrz_roi(gray)

    if roi is not None:
        return roi

    # Glare fallback
    try:
        glare_gray = remove_glare(gray)

        roi = locate_mrz_roi(
            glare_gray
        )

        if roi is not None:
            return roi

    except Exception:
        pass

    return None


# =========================================================
# Live scanner display
# =========================================================

def draw_scanner(frame, detected=False):
    """
    Draw the MRZ scanning guide.
    """

    display = frame.copy()

    h, w = display.shape[:2]

    # ---------------------------------------------------------
    # MRZ guide area
    # ---------------------------------------------------------

    left = int(w * 0.06)
    right = int(w * 0.94)

    top = int(h * 0.52)
    bottom = int(h * 0.88)

    if detected:
        color = (0, 255, 0)

        message = "MRZ DETECTED - PRESS SPACE"

    else:
        color = (255, 255, 255)

        message = "PLACE MRZ INSIDE THE BOX"

    cv2.rectangle(
        display,
        (left, top),
        (right, bottom),
        color,
        3
    )

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------

    cv2.putText(
        display,
        "OFFLINE PASSPORT SCANNER",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------

    cv2.putText(
        display,
        message,
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )

    # ---------------------------------------------------------
    # Instructions
    # ---------------------------------------------------------

    cv2.putText(
        display,
        "SPACE = Capture",
        (25, h - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        "ESC = Cancel",
        (25, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return display


# =========================================================
# Captured-image confirmation
# =========================================================

def confirm_capture(image):
    """
    Display the captured image and ask the user whether
    to proceed or retake.

    Keys:
        P = Proceed
        R = Retake
        ESC = Cancel

    Returns:
        True  -> proceed
        False -> retake
        None  -> cancel
    """

    if image is None:
        return None

    display = image.copy()

    # ---------------------------------------------------------
    # Resize captured image for display
    # ---------------------------------------------------------

    max_width = 1100
    max_height = 700

    h, w = display.shape[:2]

    scale = min(
        max_width / float(w),
        max_height / float(h),
        1.0
    )

    if scale < 1.0:

        display = cv2.resize(
            display,
            (
                int(w * scale),
                int(h * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    # ---------------------------------------------------------
    # Add border / instructions
    # ---------------------------------------------------------

    display = cv2.copyMakeBorder(
        display,
        80,
        80,
        20,
        20,
        cv2.BORDER_CONSTANT,
        value=(30, 30, 30)
    )

    cv2.putText(
        display,
        "CAPTURED IMAGE - CONFIRM",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        "P = Proceed    R = Retake    ESC = Cancel",
        (30, display.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    while True:

        cv2.imshow(
            "Confirm Captured Passport",
            display
        )

        key = cv2.waitKey(30) & 0xFF

        # -----------------------------------------------------
        # Proceed
        # -----------------------------------------------------

        if key in (
            ord("p"),
            ord("P")
        ):

            cv2.destroyWindow(
                "Confirm Captured Passport"
            )

            return True

        # -----------------------------------------------------
        # Retake
        # -----------------------------------------------------

        if key in (
            ord("r"),
            ord("R")
        ):

            cv2.destroyWindow(
                "Confirm Captured Passport"
            )

            return False

        # -----------------------------------------------------
        # Cancel
        # -----------------------------------------------------

        if key == 27:

            cv2.destroyWindow(
                "Confirm Captured Passport"
            )

            return None


# =========================================================
# Real-time passport scanner
# =========================================================

def scan_passport_camera(
    camera_index=0
):
    """
    Open the camera and allow the user to capture the MRZ.

    Workflow:

        Camera
          ↓
        Detect MRZ
          ↓
        User presses SPACE
          ↓
        Captured image shown
          ↓
        P = proceed
        R = retake
        ESC = cancel

    Returns:
        Captured MRZ image if user chooses Proceed.
        None if cancelled.
    """

    camera = cv2.VideoCapture(
        camera_index
    )

    if not camera.isOpened():

        print(
            "Unable to open camera."
        )

        return None

    # ---------------------------------------------------------
    # Camera resolution
    # ---------------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    print()
    print("=" * 60)
    print("OFFLINE PASSPORT SCANNER")
    print("=" * 60)
    print()
    print("Place the MRZ inside the box.")
    print("Press SPACE to capture.")
    print("You can then confirm the image.")
    print("ESC = cancel")
    print("=" * 60)

    detected = False

    while True:

        ok, frame = camera.read()

        if not ok:

            print(
                "Unable to read camera frame."
            )

            break

        # -----------------------------------------------------
        # Detect MRZ
        # -----------------------------------------------------

        roi = detect_live_mrz(
            frame
        )

        detected = roi is not None

        # -----------------------------------------------------
        # Draw live scanner
        # -----------------------------------------------------

        display = draw_scanner(
            frame,
            detected=detected
        )

        cv2.imshow(
            "Offline Passport Scanner",
            display
        )

        # -----------------------------------------------------
        # Keyboard
        # -----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # -----------------------------------------------------
        # SPACE = capture
        # -----------------------------------------------------

        if key == 32:

            if not detected:

                print(
                    "MRZ has not been detected. "
                    "Place it inside the box."
                )

                continue

            # -------------------------------------------------
            # Capture the CURRENT FULL CAMERA FRAME.
            #
            # This is important:
            # We keep the complete passport image instead of
            # returning only the MRZ crop.
            # -------------------------------------------------

            captured = frame.copy()

            camera.release()

            cv2.destroyAllWindows()

            # -------------------------------------------------
            # Show captured image for confirmation
            # -------------------------------------------------

            decision = confirm_capture(
                captured
            )

            # -------------------------------------------------
            # Proceed
            # -------------------------------------------------

            if decision is True:

                return captured

            # -------------------------------------------------
            # Retake
            # -------------------------------------------------

            if decision is False:

                print(
                    "Retaking..."
                )

                # Reopen camera
                camera = cv2.VideoCapture(
                    camera_index
                )

                camera.set(
                    cv2.CAP_PROP_FRAME_WIDTH,
                    1280
                )

                camera.set(
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    720
                )

                continue

            # -------------------------------------------------
            # Cancel
            # -------------------------------------------------

            return None

        # -----------------------------------------------------
        # ESC = cancel
        # -----------------------------------------------------

        if key == 27:

            camera.release()

            cv2.destroyAllWindows()

            return None

    camera.release()

    cv2.destroyAllWindows()

    return None