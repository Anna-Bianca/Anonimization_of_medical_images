from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pytesseract

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecta y extrae texto de imagenes usando OpenCV + Tesseract OCR."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Images"),
        help="Carpeta con imagenes de entrada (default: Images).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Pipeline") / "ocr_output",
        help="Carpeta donde se guardan textos y cajas detectadas.",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=50.0,
        help="Confianza minima (0-100) para dibujar cajas por palabra.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default="",
        help="Ruta al ejecutable tesseract si no esta en PATH (Windows).",
    )
    return parser.parse_args()


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def preprocess_for_ocr(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, thresh


def ocr_image(image_path: Path, output_dir: Path, min_conf: float) -> None:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"[WARN] No se pudo leer {image_path}")
        return

    gray, thresh = preprocess_for_ocr(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    extracted_text = pytesseract.image_to_string(thresh)
    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

    n_boxes = len(data.get("level", []))
    boxes_drawn = 0
    for i in range(n_boxes):
        raw_text = data["text"][i].strip()
        if not raw_text:
            continue

        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0

        if conf < min_conf:
            continue

        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )

        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
        boxes_drawn += 1

    stem = image_path.stem
    text_file = output_dir / f"{stem}.txt"
    boxed_image_file = output_dir / f"{stem}_boxed.png"
    gray_file = output_dir / f"{stem}_gray.png"

    text_file.write_text(extracted_text.strip() + "\n", encoding="utf-8")
    cv2.imwrite(str(boxed_image_file), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(gray_file), gray)

    print(f"[OK] {image_path.name}")
    print(f"     -> Texto: {text_file}")
    print(f"     -> Cajas: {boxed_image_file} (palabras detectadas: {boxes_drawn})")


def main() -> None:
    args = parse_args()

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de entrada: {args.input_dir}. "
            "Crea la carpeta o pasa --input-dir con una ruta valida."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = iter_images(args.input_dir)
    if not images:
        print(f"[INFO] No hay imagenes en {args.input_dir}")
        return

    print(f"Procesando {len(images)} imagen(es) de {args.input_dir} ...")
    for image_path in images:
        ocr_image(image_path=image_path, output_dir=args.output_dir, min_conf=args.min_conf)

    print(f"\nListo. Resultados en: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
