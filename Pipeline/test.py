import pytesseract
import numpy as np
import cv2
import re
from pathlib import Path
from shutil import which

PROJECT_ROOT = Path(__file__).resolve().parents[1]
filename = PROJECT_ROOT / 'Images' / 'con_carga.jpeg'


def configure_tesseract_cmd():
    detected = which("tesseract")
    candidates = []
    if detected:
        candidates.append(Path(detected))

    candidates.extend(
        [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]
    )

    for exe in candidates:
        if exe.exists():
            pytesseract.pytesseract.tesseract_cmd = str(exe)
            return exe

    raise FileNotFoundError(
        "Could not find tesseract.exe. Add it to PATH or set "
        "pytesseract.pytesseract.tesseract_cmd explicitly."
    )


TESSERACT_EXE = configure_tesseract_cmd()
print(f"Using Tesseract executable: {TESSERACT_EXE}")

def is_valid_line(text):
    # Must have at least 2 consecutive letters
    return bool(re.search(r'[a-zA-Z]{2,}', text))

def extract_text_generic(image_path):
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}. Current working directory: {Path.cwd()}"
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = set()

    # --- Pass 1: Standard ---
    print( "Running standard OCR pass..." )
    standard_text = pytesseract.image_to_string(gray).strip().split('\n')
    print(standard_text)
    results.update(standard_text)

    # --- Pass 2: Inverted ---
    inverted_text = pytesseract.image_to_string(cv2.bitwise_not(gray)).strip().split('\n')
    print( "Running inverted OCR pass..." )
    print(inverted_text)
    results.update(inverted_text)

    # --- Pass 3: Detect horizontal bands by saturation ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    row_sat = hsv[:, :, 1].mean(axis=1)

    colored_rows = np.where(row_sat > 30)[0]

    if len(colored_rows) > 0:
        # Group consecutive colored rows into bands
        bands = []
        start = colored_rows[0]
        prev = colored_rows[0]

        for r in colored_rows[1:]:
            if r - prev > 10:
                bands.append((start, prev))
                start = r
            prev = r
        bands.append((start, prev))

        for (y1, y2) in bands:
            # Skip bands too thin to contain text
            if y2 - y1 < 15:
                continue

            # Add padding so letters aren't cut off at edges
            padding = 10
            y1_pad = max(0, y1 - padding)
            y2_pad = min(img.shape[0], y2 + padding)
            crop = img[y1_pad:y2_pad, :]
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # Run both normal and inverted on each band
            for version in [crop_gray, cv2.bitwise_not(crop_gray)]:
                upscaled = cv2.resize(version, None, fx=3, fy=3,
                                      interpolation=cv2.INTER_CUBIC)
                text = pytesseract.image_to_string(upscaled, config='--psm 7')
                print( "Running color band OCR pass..." )
                print(text.strip().split('\n'))
                results.update(text.strip().split('\n'))

    # Filter out garbage lines
    return [line for line in results if line.strip() and is_valid_line(line)]

all_text = extract_text_generic(filename)
print('\n'.join(all_text))

