

import time
from threading import Lock
from contextlib import asynccontextmanager

import cv2
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool


from scan_passport import process_image
from src.mrz import initialize_ocr
from src.scanner import scan_passport_camera

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
OCR_LOCK = Lock()


# --------------------------------------------------
# Initialize existing OCR
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await run_in_threadpool(initialize_ocr)
        app.state.ocr_ready = True
        print("OCR engine initialized successfully.")

    except Exception as exc:
        app.state.ocr_ready = False
        print("OCR initialization failed:", repr(exc))
        raise

    yield


app = FastAPI(
    title="Offline Passport OCR API",
    description=(
        "Offline passport OCR with camera capture, image upload, "
        "and JSON responses."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health", tags=["Status"])
def health():
    ready = getattr(app.state, "ocr_ready", False)

    return {
        "status": "ok" if ready else "not_ready",
        "service": "Offline Passport OCR",
        "processing": "offline",
        "ocr_initialized": ready
    }



# --------------------------------------------------
# Common image-processing function
# --------------------------------------------------

async def process_uploaded_image(file: UploadFile):
    started = time.perf_counter()

    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Maximum image size is 10 MB.",
            )

        image = cv2.imdecode(
            np.frombuffer(content, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image is None or image.size == 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid or unreadable image.",
            )

        # Keep the existing OCR pipeline unchanged.
        def run_ocr():
            with OCR_LOCK:
                return process_image(image)

        result = await run_in_threadpool(run_ocr)

        if not isinstance(result, dict):
            raise RuntimeError(
                "process_image() did not return a dictionary."
            )

        result["api_total_ms"] = int(
            (time.perf_counter() - started) * 1000
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        print("OCR error:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail="OCR processing failed. Check the server terminal.",
        ) from exc

    finally:
        await file.close()




# --------------------------------------------------
# API 1: Scan passport using the OpenCV camera
# --------------------------------------------------

@app.post("/api/passport/scan", tags=["Passport OCR"])
async def scan_passport():
    """
    Opens the existing OpenCV camera scanner.
    Captures a passport and returns the OCR result as JSON.
    No file upload is required.
    """

    started = time.perf_counter()

    try:
        def capture_and_process():
            # Open the existing OpenCV scanner.
            with OCR_LOCK:
                image = scan_passport_camera()

                # User cancelled or scanner could not capture an image.
                if image is None:
                    return {
                        "status": "cancelled",
                        "reason": "Camera scan cancelled or no image captured."
                    }

                # Reuse the existing OCR pipeline.
                return process_image(image)

        result = await run_in_threadpool(capture_and_process)

        result["api_total_ms"] = int(
            (time.perf_counter() - started) * 1000
        )

        return result

    except Exception as exc:
        print("Camera OCR error:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail="Camera scanning failed. Check the server terminal."
        ) from exc


# --------------------------------------------------
# API 2: Upload an existing passport image
# --------------------------------------------------

@app.post("/api/passport/upload", tags=["Passport OCR"])
async def upload_passport(file: UploadFile = File(...)):
    """Receives an image selected from the client's device."""
    return await process_uploaded_image(file)
