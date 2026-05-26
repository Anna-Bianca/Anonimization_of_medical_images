import pytesseract
import numpy as np
import cv2
import re

filename = 'Images/frase.jpg'

def is_valid_line(text):
    # Must have at least 2 consecutive letters
    return bool(re.search(r'[a-zA-Z]{2,}', text))

def extract_text_generic(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = set()

    # --- Pass 1: Standard ---
    results.update(pytesseract.image_to_string(gray).strip().split('\n'))

    # --- Pass 2: Inverted ---
    results.update(pytesseract.image_to_string(cv2.bitwise_not(gray)).strip().split('\n'))

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
                results.update(text.strip().split('\n'))

    # Filter out garbage lines
    return [line for line in results if line.strip() and is_valid_line(line)]

all_text = extract_text_generic(filename)
print('\n'.join(all_text))

