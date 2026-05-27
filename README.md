# Anonimization_of_medical_images

OCR pipeline for medical-image text detection and traceable validation.

## Project Structure

```text
Anonimization_of_medical_images/
|-- requirements.txt
|-- README.md
|-- Images/
`-- Pipeline/
    `-- OCR.py
```

## What `Pipeline/OCR.py` Produces

For each input image, the script writes:

1. `*.txt` with final words and their pixel coordinates.
2. `*_boxed.png` with only final validated bounding boxes.
3. `*_gray.png` grayscale reference.
4. `*_debug_report.txt` with full debug trace (all detections + filters + final result).
5. `debug/*.png` (optional) with per-pass intermediate images when `--save-debug-images` is enabled.

## Final Validation Logic

The final output is built in this exact order:

1. Detect everything with all enabled OCR passes.
2. Apply confidence gate: keep detections with `conf >= min_conf_primary`.
3. Apply overlap/repetition rule: keep tokens repeated in at least `min_repetitions` passes with `IoU >= overlap_threshold`.
4. Deduplicate remaining boxes (same token/location).
5. Save final boxes and final text.

## Enabled OCR Passes

1. `standard_gray`
2. `inverted_gray`
3. `unsharp`
4. `unsharp_inverted`
5. `gray_upscale_x2`
6. `gray_upscale_x3`
7. `unsharp_upscale_x3`
8. Band passes:
   `band_*_gray_x3`, `band_*_gray_inv_x3`, `band_*_unsharp_x3`

These were kept after debug analysis because they recovered target text better than the disabled variants.

## Setup (PowerShell)

1. Create and activate virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Install Tesseract OCR (system binary):

```powershell
winget install UB-Mannheim.TesseractOCR
```

If needed, pass explicit path:

```powershell
python .\Pipeline\OCR.py --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Run

Recommended command (your latest rule set):

```powershell
python .\Pipeline\OCR.py --input-dir .\Images --output-dir .\Pipeline\ocr_output --min-conf-primary 60 --overlap-threshold 0.6 --min-repetitions 2
```

With debug images:

```powershell
python .\Pipeline\OCR.py --input-dir .\Images --output-dir .\Pipeline\ocr_output --min-conf-primary 60 --overlap-threshold 0.6 --min-repetitions 2 --save-debug-images
```

## CLI Notes

1. `--min-conf-primary`: active confidence threshold for final validation.
2. `--overlap-threshold`: IoU threshold for overlap rule.
3. `--min-repetitions`: minimum number of distinct passes that must detect the same token.
4. `--min-conf-secondary` and `--min-conf` are kept for backward compatibility with older runs/reports.

## Final TXT Format

Each output text file has this structure:

```text
# final_word    x   y   w   h   conf    source_pass
WORD1   ...
WORD2   ...

# final_text
WORD1
WORD2
```
