# Offline Passport OCR

An offline Python-based Passport OCR tool that automatically detects and extracts Machine Readable Zone (MRZ) information from passport images using OpenCV and PaddleOCR.

The system is designed to work completely offline without cloud services or external APIs.

## Features

- Offline passport MRZ extraction
- Automatic MRZ region detection
- Support for passport images captured using a camera or uploaded from a computer
- PaddleOCR-based text recognition
- ICAO 9303 check-digit validation
- Context-aware correction of common OCR errors
- Glare handling and image preprocessing
- Blur/quality checking
- Fallback MRZ detection for difficult images
- Rejects unreliable images with `retake_required`
- CPU-optimized processing
- Batch processing of multiple passport images
- JSON output for each processed passport

## Processing Pipeline

```text
Passport Image
      ↓
Image Resize
      ↓
MRZ Detection using OpenCV
      ↓
MRZ Region Cropping
      ↓
Image Preprocessing
      ↓
PaddleOCR Recognition
      ↓
MRZ Line Extraction
      ↓
OCR Error Correction
      ↓
ICAO 9303 Check-Digit Validation
      ↓
Passport Field Parsing
      ↓
JSON Result
```

## Project Structure

```text
Passport_OCR/
│
├── scan_passport.py
├── requirements.txt
│
├── src/
│   ├── mrz.py
│   ├── parser.py
│   └── scanner.py
│
├── images/
│   └── passport images
│
└── results/
    └── JSON results
```

### `scan_passport.py`

Main application controller. It provides:

- Camera scanning
- Image upload
- Batch processing
- Interactive menu
- JSON result generation

### `src/mrz.py`

Core MRZ processing module responsible for:

- MRZ detection
- Image preprocessing
- PaddleOCR recognition
- OCR correction
- ICAO 9303 validation
- Quality checking
- Fallback processing
- Processing-time measurement

### `src/parser.py`

Parses validated MRZ lines and extracts:

- Passport number
- Surname
- Given names
- Nationality
- Date of birth
- Sex
- Expiry date
- Issuing country

### `src/scanner.py`

Handles the live camera scanner, including:

- Camera initialization
- Live MRZ detection
- Scanning guide
- Image capture
- Capture confirmation
- Retake functionality

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode

Run:

```bash
python scan_passport.py
```

The application provides:

```text
1. Scan passport using camera
2. Upload passport image
3. Exit
```

### Batch Processing

To process multiple passport images:

```bash
python scan_passport.py --input ./images --output ./results
```

Example:

```text
images/
├── passport1.jpg
├── passport2.jpg
├── blurry.jpg
└── passport3.png
```

Results are saved as:

```text
results/
├── passport1.json
├── passport2.json
├── blurry.json
└── passport3.json
```

## Output

### Successful Result

```json
{
  "status": "success",
  "surname": "EXAMPLE",
  "given_names": "JOHN",
  "passport_number": "A1234567",
  "nationality": "IND",
  "date_of_birth": "1999-01-01",
  "sex": "M",
  "expiry_date": "2030-01-01",
  "issuing_country": "IND",
  "mrz_raw": [
    "P<INDEXAMPLE<<JOHN<<<<<<<<<<<<<<<<<<<<",
    "A1234567<8IND9901011M3001012<<<<<<<<"
  ],
  "check_digits_valid": true,
  "processing_ms": 500
}
```

### Failed / Poor Quality Image

If the system cannot confidently extract and validate the MRZ:

```json
{
  "status": "retake_required",
  "reason": "mrz_not_detected",
  "processing_ms": 400
}
```

The system does not return unvalidated passport information when the MRZ cannot be reliably processed.

## ICAO 9303 Validation

The system implements the ICAO 9303 MRZ check-digit algorithm directly.

The algorithm uses:

```text
Character values:
0-9 → 0-9
A-Z → 10-35
<   → 0

Repeating weights:
7, 3, 1
```

The check digits are used to validate important MRZ fields such as the passport number, date of birth, expiry date, personal number, and composite MRZ data.

## OCR Error Correction

The system handles common OCR character confusions in numeric fields, including:

```text
O → 0
Q → 0
D → 0
I → 1
L → 1
Z → 2
S → 5
B → 8
G → 6
```

Corrections are applied conservatively and the resulting MRZ must still pass ICAO validation before it is accepted.

## Fallback Processing

If the primary MRZ detection or OCR attempt fails, the system uses fallback strategies such as:

1. Bottom-region MRZ search
2. Alternative OCR preprocessing
3. Glare correction
4. Recognition-only OCR
5. Final ICAO validation

A result is accepted only after successful MRZ validation.

## Performance

The system is optimized for CPU processing using:

- PaddleOCR mobile models
- Image resizing
- Limited OCR input dimensions
- OCR model reuse
- OCR warm-up
- Recognition-only fast path
- Time-budget-aware fallback processing

The project uses a target processing budget of approximately **1.8 seconds after OCR initialization**.

## Limitations

Although the system is designed to handle real-world passport images, the following limitations apply:

- **Severely blurred images:** Extremely blurred images may not contain enough character information for reliable OCR. Such images may be rejected with `retake_required`.

- **Heavy glare or reflections:** The system includes glare correction, but strong reflections covering MRZ characters may still prevent accurate recognition.

- **Extreme perspective or rotation:** The system can handle moderate variations in passport orientation, but extremely tilted, rotated, or distorted images may fail MRZ detection.

- **Damaged or partially visible MRZ:** If MRZ characters are missing, covered, cropped, or severely damaged, the system may not be able to reconstruct a valid MRZ.

- **OCR ambiguity:** Some characters, such as `O/0`, `I/1`, and `B/8`, can look similar. The system applies context-aware corrections, but corrections are accepted only when the resulting MRZ passes ICAO 9303 validation.

- **Passport format:** The current implementation is primarily designed for standard machine-readable passports using the two-line TD3 MRZ format. Other travel-document formats may require additional handling.

- **CPU performance:** The target is approximately **1.8 seconds after OCR model initialization** on the tested CPU environment. Actual processing time can vary depending on CPU performance, image resolution, and system load.

- **OCR model initialization:** The first OCR operation may take longer because the PaddleOCR models need to be loaded and initialized. Subsequent processing benefits from model reuse and warm-up.

- **No complete passport-page OCR:** The system focuses on extracting and validating MRZ information. It does not attempt to extract every piece of visual information printed elsewhere on the passport page.

- **Validation does not guarantee visual correctness:** ICAO 9303 check-digit validation significantly improves reliability, but a technically valid MRZ should still be treated as machine-extracted data and verified when used in a real-world identity-verification workflow.
## Privacy

This project is designed for offline processing.

- No cloud OCR service
- No external API
- No internet connection required during processing
- Passport images are processed locally
- Results are stored locally as JSON files

## Technologies

- Python
- OpenCV
- NumPy
- PaddleOCR
- PaddlePaddle
- ICAO 9303 MRZ validation

## License

This project is intended for development, testing, and demonstration purposes.