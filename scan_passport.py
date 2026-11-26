import argparse
import json
import os
import sys

import cv2

from src.mrz import extract_mrz
from src.parser import parse_mrz
from src.scanner import scan_passport_camera


# =========================================================
# Process one image
# =========================================================

def process_image(image, output_path=None):
    """
    Process one passport image using the offline PaddleOCR MRZ pipeline.

    The OCR/validation work is handled by src.mrz.extract_mrz().
    No internet or external API is used.
    """

    result = extract_mrz(image)

    # -----------------------------------------------------
    # OCR failed
    # -----------------------------------------------------

    if result["status"] != "success":

        output = {
            "status": "retake_required",
            "reason": result.get(
                "reason",
                "mrz_not_detected"
            ),
            "processing_ms": result.get(
                "processing_ms",
                0
            )
        }

    # -----------------------------------------------------
    # OCR succeeded
    # -----------------------------------------------------

    else:

        line1 = result["line1"]
        line2 = result["line2"]

        # Use the parser already present in the project.
        fields = parse_mrz(
            line1,
            line2
        )

        output = {
            "status": "success",

            **fields,

            "mrz_raw": [
                line1,
                line2
            ],

            # The MRZ extractor has already performed
            # ICAO check-digit validation before returning
            # success.
            "check_digits_valid": True,

            "processing_ms": result.get(
                "processing_ms",
                0
            )
        }

    # -----------------------------------------------------
    # Save JSON
    # -----------------------------------------------------

    if output_path:

        output_directory = os.path.dirname(
            output_path
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True
            )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                output,
                f,
                indent=2,
                ensure_ascii=False
            )

    return output


# =========================================================
# Camera mode
# =========================================================

def camera_mode():

    print()
    print("=" * 60)
    print("                    CAMERA SCAN")
    print("=" * 60)
    print()
    print("Place the passport in front of the camera.")
    print("Align the MRZ inside the scanning guide.")
    print()
    print("The camera will detect the MRZ.")
    print("Press SPACE to capture.")
    print("After capture:")
    print("  P = Proceed")
    print("  R = Retake")
    print("  ESC = Cancel")
    print()

    # -----------------------------------------------------
    # Camera scanner
    # -----------------------------------------------------

    passport_image = scan_passport_camera()

    if passport_image is None:

        print()
        print("Scan cancelled.")

        return

    print()
    print("Passport captured.")
    print("Running offline PaddleOCR...")
    print("Please wait...")

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    output_path = os.path.join(
        "results",
        "camera_passport.json"
    )

    result = process_image(
        passport_image,
        output_path
    )

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("                         RESULT")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

    print()
    print(
        "Result saved to: "
        f"{output_path}"
    )


# =========================================================
# Upload mode
# =========================================================

def upload_mode():

    try:

        import tkinter as tk
        from tkinter import filedialog

    except ImportError:

        print(
            "Tkinter is not available."
        )

        return

    # -----------------------------------------------------
    # Open file picker
    # -----------------------------------------------------

    root = tk.Tk()

    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Passport Image",
        filetypes=[
            (
                "Image files",
                "*.jpg *.jpeg *.png *.bmp *.webp"
            ),
            (
                "All files",
                "*.*"
            )
        ]
    )

    root.destroy()

    if not file_path:

        print(
            "No image selected."
        )

        return

    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------

    image = cv2.imread(
        file_path
    )

    if image is None:

        print(
            "Could not read selected image."
        )

        return

    print()
    print(
        f"Selected: "
        f"{os.path.basename(file_path)}"
    )

    print()
    print(
        "Running offline PaddleOCR..."
    )

    # -----------------------------------------------------
    # Output filename
    # -----------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    base_name = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    output_path = os.path.join(
        "results",
        f"{base_name}.json"
    )

    # -----------------------------------------------------
    # Process
    # -----------------------------------------------------

    result = process_image(
        image,
        output_path
    )

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("                         RESULT")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

    print()
    print(
        f"Result saved to: "
        f"{output_path}"
    )


# =========================================================
# Batch mode
# =========================================================

def batch_mode(
    input_dir,
    output_dir
):

    if not os.path.isdir(
        input_dir
    ):

        print(
            f"Input directory does not exist: "
            f"{input_dir}"
        )

        return

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    files = [
        filename
        for filename in os.listdir(
            input_dir
        )
        if filename.lower().endswith(
            extensions
        )
    ]

    if not files:

        print()
        print(
            "No supported image files found."
        )

        return

    print()
    print("=" * 60)
    print("                    BATCH PROCESSING")
    print("=" * 60)
    print(
        f"Input : {input_dir}"
    )
    print(
        f"Output: {output_dir}"
    )
    print(
        f"Images: {len(files)}"
    )

    # -----------------------------------------------------
    # Process every image
    # -----------------------------------------------------

    for filename in files:

        print()
        print("=" * 60)
        print(
            f"Processing: {filename}"
        )
        print("=" * 60)

        path = os.path.join(
            input_dir,
            filename
        )

        image = cv2.imread(
            path
        )

        if image is None:

            print(
                "Could not read image."
            )

            continue

        # -------------------------------------------------
        # Output JSON path
        # -------------------------------------------------

        output_name = (
            os.path.splitext(
                filename
            )[0]
            + ".json"
        )

        output_path = os.path.join(
            output_dir,
            output_name
        )

        # -------------------------------------------------
        # Process
        # -------------------------------------------------

        result = process_image(
            image,
            output_path
        )

        # -------------------------------------------------
        # Display status
        # -------------------------------------------------

        print(
            f"Status: "
            f"{result['status']}"
        )

        if (
            result["status"]
            == "success"
        ):

            print(
                "Passport number: "
                f"{result.get('document_number', '')}"
            )

            print(
                "Name: "
                f"{result.get('surname', '')} "
                f"{result.get('given_names', '')}"
            )

            print(
                "Nationality: "
                f"{result.get('nationality', '')}"
            )

            print(
                "Date of birth: "
                f"{result.get('date_of_birth', '')}"
            )

            print(
                "Expiry date: "
                f"{result.get('date_of_expiry', '')}"
            )

            print(
                "Check digits valid: "
                f"{result.get('check_digits_valid', False)}"
            )

        else:

            print(
                "Reason: "
                f"{result.get('reason', '')}"
            )

        print(
            "Processing time: "
            f"{result['processing_ms']} ms"
        )

        print(
            f"Result saved: "
            f"{output_path}"
        )


# =========================================================
# Interactive menu
# =========================================================

def interactive_mode():

    while True:

        print()
        print("=" * 60)
        print("              OFFLINE PASSPORT OCR")
        print("=" * 60)
        print()
        print("OCR Engine: PaddleOCR")
        print("Processing: Offline")
        print()
        print("1. Scan passport using camera")
        print("2. Upload passport image")
        print("3. Exit")
        print()

        choice = input(
            "Choose an option: "
        ).strip()

        # -------------------------------------------------
        # Camera
        # -------------------------------------------------

        if choice == "1":

            camera_mode()

        # -------------------------------------------------
        # Upload
        # -------------------------------------------------

        elif choice == "2":

            upload_mode()

        # -------------------------------------------------
        # Exit
        # -------------------------------------------------

        elif choice == "3":

            print()
            print(
                "Exiting."
            )

            break

        # -------------------------------------------------
        # Invalid choice
        # -------------------------------------------------

        else:

            print()
            print(
                "Invalid option. "
                "Please choose 1, 2 or 3."
            )


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Offline Passport OCR "
            "using PaddleOCR and ICAO 9303 validation"
        )
    )

    parser.add_argument(
        "--input",
        help=(
            "Input image directory "
            "for batch processing"
        )
    )

    parser.add_argument(
        "--output",
        help=(
            "Output directory "
            "for JSON results"
        )
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Batch mode
    #
    # Example:
    # python scan_passport.py \
    #     --input ./images \
    #     --output ./results
    # -----------------------------------------------------

    if args.input:

        output_dir = (
            args.output
            if args.output
            else "./results"
        )

        batch_mode(
            args.input,
            output_dir
        )

        return

    # -----------------------------------------------------
    # Interactive mode
    # -----------------------------------------------------

    interactive_mode()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()