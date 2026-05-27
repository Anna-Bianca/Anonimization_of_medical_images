from __future__ import annotations
"""
OCR pipeline for medical-image text detection with per-pass debugging.

Design goals:
1. Use one unified OCR flow for both extracted text and bounding boxes.
2. Keep debug visibility for each OCR strategy/pipeline pass.
3. Produce a final conservative result with explicit validation rules.

Current final-validation workflow:
1. Detect everything with enabled OCR passes (full trace in debug report).
2. Apply confidence filter (`conf >= min_conf_primary`).
3. Keep only words repeated across passes with IoU overlap >= threshold.
4. Deduplicate repeated boxes for the same token/region.
5. Write final boxes + final text with pixel coordinates.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from shutil import which

import cv2
import numpy as np
import pytesseract

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SATURATION_THRESHOLD = 30
BAND_GAP_THRESHOLD = 10
MIN_BAND_HEIGHT = 15
BAND_PADDING = 10
OVERLAP_THRESHOLD_DEFAULT = 0.6
MIN_REPETITIONS_DEFAULT = 2


@dataclass(frozen=True)
class OCRPass:
    """One OCR execution pass with its image, geometry mapping and Tesseract config."""
    name: str
    image: np.ndarray
    x_offset: int = 0
    y_offset: int = 0
    scale: float = 1.0
    tesseract_config: str = ""


@dataclass(frozen=True)
class WordDetection:
    """A normalized OCR token detection mapped back to original-image coordinates."""
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int
    source_pass: str


@dataclass(frozen=True)
class PassDebug:
    """Debug summary for one pass (text preview + detection stats)."""
    name: str
    raw_lines: list[str]
    total_data_rows: int
    accepted_detections: int
    mean_conf: float


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Note:
    - `--min-conf-primary` is the active confidence gate in the final pipeline.
    - `--min-conf-secondary` is currently kept for compatibility with older runs/reports.
    """
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
        "--min-conf-primary",
        type=float,
        default=50.0,
        help="Confianza primaria (0-100) para cajas 'seguras'.",
    )
    parser.add_argument(
        "--min-conf-secondary",
        type=float,
        default=30.0,
        help="Confianza secundaria (0-100) para cajas 'candidatas'.",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=None,
        help="Alias de --min-conf-primary por compatibilidad.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default="",
        help="Ruta al ejecutable tesseract si no esta en PATH (Windows).",
    )
    parser.add_argument(
        "--save-debug-images",
        action="store_true",
        help="Guarda imagenes intermedias de cada paso en output/debug.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=OVERLAP_THRESHOLD_DEFAULT,
        help="IoU minimo para considerar que dos cajas pertenecen a la misma palabra.",
    )
    parser.add_argument(
        "--min-repetitions",
        type=int,
        default=MIN_REPETITIONS_DEFAULT,
        help="Repeticiones minimas de la misma palabra (en pases distintos) para validarla.",
    )
    return parser.parse_args()


def iter_images(input_dir: Path) -> list[Path]:
    """Return supported image files in stable order."""
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def preprocess_for_ocr(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Basic grayscale + Otsu threshold fallback preprocessing."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, thresh


def configure_tesseract_cmd(tesseract_cmd: str) -> Path:
    """Resolve and configure the Tesseract executable used by pytesseract."""
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return Path(tesseract_cmd)

    detected = which("tesseract")
    candidates: list[Path] = []
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
        "No se encontro tesseract.exe. Agregalo al PATH o pasalo con --tesseract-cmd."
    )


def is_valid_token(text: str) -> bool:
    # Accept letters, numbers, and mixed tokens (IDs included).
    return bool(re.search(r"[a-zA-Z0-9]", text))


def _dedupe_preserve_order(lines: list[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            ordered.append(line)
    return ordered


def _dedupe_word_detections(detections: list[WordDetection]) -> list[WordDetection]:
    """
    Merge near-duplicate detections from multiple passes.

    We normalize token text and quantize geometry to merge jittered boxes,
    keeping the highest-confidence detection per bucket.
    """
    by_key: dict[tuple[str, int, int, int, int], WordDetection] = {}
    for det in detections:
        token = _normalize_token_for_match(det.text)
        key = (
            token,
            det.x // 8,
            det.y // 8,
            det.w // 8,
            det.h // 8,
        )
        current = by_key.get(key)
        if current is None or det.conf > current.conf:
            by_key[key] = det
    return list(by_key.values())


def _normalize_token_for_match(text: str) -> str:
    """Normalize token so punctuation variants match (e.g. 'AP,' == 'AP')."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "", text).lower()
    return normalized if normalized else text.strip().lower()


def _bbox_iou(a: WordDetection, b: WordDetection) -> float:
    """Intersection over Union between two detections."""
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _apply_overlap_repetition_filter(
    detections: list[WordDetection],
    min_repetitions: int,
    overlap_threshold: float,
) -> tuple[list[dict], list[WordDetection]]:
    """
    Rule 2:
    Keep only detections that belong to token clusters repeated in at least
    `min_repetitions` passes with IoU >= `overlap_threshold`.
    Returns:
    - valid_clusters: clusters that satisfy the overlap/repetition rule
    - candidate_detections_after_rule_2: all detections that belong to valid clusters
    """
    clusters: list[dict] = []
    for det in detections:
        token = _normalize_token_for_match(det.text)
        assigned = False
        for cluster in clusters:
            if token != cluster["token"]:
                continue
            if any(_bbox_iou(det, member) >= overlap_threshold for member in cluster["members"]):
                cluster["members"].append(det)
                assigned = True
                break
        if not assigned:
            clusters.append({"token": token, "members": [det]})

    valid_clusters: list[dict] = []
    candidates: list[WordDetection] = []
    for cluster in clusters:
        members: list[WordDetection] = cluster["members"]
        supporting_passes = {m.source_pass for m in members}
        if len(supporting_passes) < min_repetitions:
            continue
        valid_clusters.append(cluster)
        candidates.extend(members)

    return valid_clusters, candidates


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Local-contrast enhancement helper (kept for controlled experiments)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _apply_unsharp(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.7, blurred, -0.7, 0)


def _adaptive_binary(gray: np.ndarray, invert: bool) -> np.ndarray:
    """Adaptive threshold helper (currently not enabled in production passes)."""
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        mode,
        31,
        5,
    )


def _apply_morphology(binary_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Morphology helper (currently not enabled in production passes)."""
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return closed, opened


def _build_global_passes(image_bgr: np.ndarray) -> list[OCRPass]:
    """Build full-image OCR passes currently enabled for production."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = _apply_clahe(gray)
    unsharp = _apply_unsharp(clahe)

    # Enabled passes are precision-first based on current debug reports:
    # - con_carga_debug_report.txt
    # - normal-102_debug_report.txt
    # We keep only methods that detected target words in at least one report.
    #
    # Disabled (only hallucinations in current evaluations):
    # - clahe, clahe_inverted
    # - adaptive_binary, adaptive_binary_inv
    # - morph_close, morph_open
    # - adaptive_upscale_x3
    # Reason: these strategies increased false positives without reliably
    # recovering target tokens in the two debug samples analyzed.
    return [
        OCRPass(name="standard_gray", image=gray, tesseract_config="--psm 6"),
        OCRPass(name="inverted_gray", image=cv2.bitwise_not(gray), tesseract_config="--psm 6"),
        OCRPass(name="unsharp", image=unsharp, tesseract_config="--psm 6"),
        OCRPass(name="unsharp_inverted", image=cv2.bitwise_not(unsharp), tesseract_config="--psm 6"),
        OCRPass(
            name="gray_upscale_x2",
            image=cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
            scale=2.0,
            tesseract_config="--psm 6",
        ),
        OCRPass(
            name="gray_upscale_x3",
            image=cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
            scale=3.0,
            tesseract_config="--psm 6",
        ),
        OCRPass(
            name="unsharp_upscale_x3",
            image=cv2.resize(unsharp, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
            scale=3.0,
            tesseract_config="--psm 6",
        ),
    ]


def _detect_colored_bands(image_bgr: np.ndarray) -> list[tuple[int, int]]:
    """Detect horizontal high-saturation bands where colored text/background may exist."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    row_sat = hsv[:, :, 1].mean(axis=1)
    colored_rows = np.where(row_sat > SATURATION_THRESHOLD)[0]
    if len(colored_rows) == 0:
        return []

    bands: list[tuple[int, int]] = []
    start = int(colored_rows[0])
    prev = int(colored_rows[0])
    for row in colored_rows[1:]:
        row = int(row)
        if row - prev > BAND_GAP_THRESHOLD:
            bands.append((start, prev))
            start = row
        prev = row
    bands.append((start, prev))
    return bands


def _build_band_passes(image_bgr: np.ndarray) -> list[OCRPass]:
    """Build OCR passes over detected color bands, mapped back to original coordinates."""
    passes: list[OCRPass] = []
    bands = _detect_colored_bands(image_bgr)
    for idx, (y1, y2) in enumerate(bands):
        if y2 - y1 < MIN_BAND_HEIGHT:
            continue

        y1_pad = max(0, y1 - BAND_PADDING)
        y2_pad = min(image_bgr.shape[0], y2 + BAND_PADDING)
        crop = image_bgr[y1_pad:y2_pad, :]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_clahe = _apply_clahe(crop_gray)
        crop_unsharp = _apply_unsharp(crop_clahe)

        passes.extend(
            [
                OCRPass(
                    name=f"band_{idx}_gray_x3",
                    image=cv2.resize(crop_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                    y_offset=y1_pad,
                    scale=3.0,
                    tesseract_config="--psm 7",
                ),
                OCRPass(
                    name=f"band_{idx}_gray_inv_x3",
                    image=cv2.resize(
                        cv2.bitwise_not(crop_gray), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
                    ),
                    y_offset=y1_pad,
                    scale=3.0,
                    tesseract_config="--psm 7",
                ),
                OCRPass(
                    name=f"band_{idx}_unsharp_x3",
                    image=cv2.resize(crop_unsharp, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
                    y_offset=y1_pad,
                    scale=3.0,
                    tesseract_config="--psm 7",
                ),
            ]
        )

        # Disabled in band passes for the same reason as global passes:
        # clahe/adaptive variants produced high hallucination rates during evaluation.
    return passes


def _extract_from_pass(ocr_pass: OCRPass) -> tuple[list[str], list[WordDetection], PassDebug]:
    """Run OCR for one pass and return raw lines, word detections and debug metrics."""
    raw_lines = pytesseract.image_to_string(ocr_pass.image, config=ocr_pass.tesseract_config).strip().splitlines()
    data = pytesseract.image_to_data(
        ocr_pass.image,
        config=ocr_pass.tesseract_config,
        output_type=pytesseract.Output.DICT,
    )
    total_rows = len(data.get("level", []))
    detections: list[WordDetection] = []
    conf_values: list[float] = []

    for i in range(total_rows):
        raw_text = data["text"][i].strip()
        if not raw_text or not is_valid_token(raw_text):
            continue

        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0

        x = int(round(data["left"][i] / ocr_pass.scale)) + ocr_pass.x_offset
        y = int(round(data["top"][i] / ocr_pass.scale)) + ocr_pass.y_offset
        w = int(round(data["width"][i] / ocr_pass.scale))
        h = int(round(data["height"][i] / ocr_pass.scale))
        detections.append(
            WordDetection(
                text=raw_text,
                conf=conf,
                x=x,
                y=y,
                w=w,
                h=h,
                source_pass=ocr_pass.name,
            )
        )
        conf_values.append(conf)

    mean_conf = float(np.mean(conf_values)) if conf_values else -1.0
    debug_info = PassDebug(
        name=ocr_pass.name,
        raw_lines=raw_lines,
        total_data_rows=total_rows,
        accepted_detections=len(detections),
        mean_conf=mean_conf,
    )
    return raw_lines, detections, debug_info


def _print_pass_debug(pass_name: str, raw_lines: list[str]) -> None:
    if pass_name == "standard_gray":
        print("Running standard OCR pass...")
    elif pass_name == "inverted_gray":
        print("Running inverted OCR pass...")
    elif pass_name.startswith("band_"):
        print("Running color band OCR pass...")
    else:
        print(f"Running {pass_name} OCR pass...")
    print(raw_lines)


def _save_pass_image(debug_dir: Path, stem: str, ocr_pass: OCRPass) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{stem}__{ocr_pass.name}.png"
    cv2.imwrite(str(debug_path), ocr_pass.image)


def _collect_detections(
    image_bgr: np.ndarray,
    stem: str,
    debug_dir: Path,
    save_debug_images: bool,
) -> tuple[list[WordDetection], list[PassDebug]]:
    """
    Run all enabled passes and aggregate detections.

    This is the raw stage: no confidence, overlap, or dedup filters are applied here.
    """
    passes = _build_global_passes(image_bgr) + _build_band_passes(image_bgr)
    all_detections: list[WordDetection] = []
    debug_rows: list[PassDebug] = []

    for ocr_pass in passes:
        raw_lines, detections, debug_info = _extract_from_pass(ocr_pass)
        _print_pass_debug(ocr_pass.name, raw_lines)
        if save_debug_images:
            _save_pass_image(debug_dir, stem, ocr_pass)
        all_detections.extend(detections)
        debug_rows.append(debug_info)

    return all_detections, debug_rows


def _make_text_from_detections(detections: list[WordDetection]) -> str:
    """Create final text output from filtered detections preserving first-seen order."""
    lines = [d.text.strip() for d in detections if d.text.strip() and is_valid_token(d.text.strip())]
    lines = _dedupe_preserve_order(lines)
    return "\n".join(lines).strip()


def _write_debug_report(
    debug_report_path: Path,
    image_name: str,
    min_conf_primary: float,
    min_conf_secondary: float,
    overlap_threshold: float,
    min_repetitions: int,
    total_detections: int,
    detections_after_conf_filter: int,
    candidates_after_overlap: int,
    final_count: int,
    valid_clusters: list[dict],
    all_detections: list[WordDetection],
    final_detections: list[WordDetection],
    pass_debug_rows: list[PassDebug],
) -> None:
    """Persist a full trace report: raw detections, filters, clusters and final detections."""
    lines = [
        f"Image: {image_name}",
        f"Confidence step 1 (primary): {min_conf_primary}",
        f"Confidence step 2 (secondary): {min_conf_secondary}",
        f"All detections (raw): {total_detections}",
        f"After confidence filter (conf >= {min_conf_primary}): {detections_after_conf_filter}",
        (
            "Rule 2 (overlap): token repeated in >= "
            f"{min_repetitions} passes with IoU >= {overlap_threshold}"
        ),
        f"Candidates after overlap rule: {candidates_after_overlap}",
        "Rule 3 (dedup): merge duplicate boxes for same token/location",
        f"Final detections: {final_count}",
        "",
        "Per-pass debug:",
    ]

    for row in pass_debug_rows:
        lines.append(
            f"- {row.name}: raw_lines={len(row.raw_lines)}, "
            f"accepted_detections={row.accepted_detections}, "
            f"total_data_rows={row.total_data_rows}, mean_conf={row.mean_conf:.2f}"
        )
        preview = row.raw_lines[:5]
        lines.append(f"  raw_preview={preview}")

    lines.append("")
    lines.append("Clusters that passed overlap rule:")
    for idx, cluster in enumerate(valid_clusters, start=1):
        members: list[WordDetection] = cluster["members"]
        passes = sorted({m.source_pass for m in members})
        lines.append(
            f"- cluster_{idx}: token={cluster['token']!r}, members={len(members)}, passes={passes}"
        )

    lines.append("")
    lines.append("All detections (before filters):")
    for det in all_detections:
        lines.append(
            f"- pass={det.source_pass}, text={det.text!r}, conf={det.conf:.2f}, "
            f"bbox=({det.x},{det.y},{det.w},{det.h})"
        )

    lines.append("")
    lines.append("Final detections (after overlap + dedup):")
    for det in final_detections:
        lines.append(
            f"- text={det.text!r}, conf={det.conf:.2f}, "
            f"bbox=({det.x},{det.y},{det.w},{det.h}), pass={det.source_pass}"
        )

    debug_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ocr_image(
    image_path: Path,
    output_dir: Path,
    min_conf_primary: float,
    min_conf_secondary: float,
    save_debug_images: bool,
    overlap_threshold: float,
    min_repetitions: int,
) -> None:
    """
    Process one image end-to-end.

    Final-result rules:
    1. Keep detections with conf >= min_conf_primary.
    2. Keep only repeated+overlapping tokens (IoU rule).
    3. Deduplicate remaining boxes.
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"[WARN] No se pudo leer {image_path}")
        return

    gray, _ = preprocess_for_ocr(image_bgr)
    stem = image_path.stem
    debug_dir = output_dir / "debug"

    raw_detections, pass_debug_rows = _collect_detections(
        image_bgr=image_bgr,
        stem=stem,
        debug_dir=debug_dir,
        save_debug_images=save_debug_images,
    )
    # Confidence gate: only keep tokens with conf >= min_conf_primary.
    detections_after_conf = [d for d in raw_detections if d.conf >= min_conf_primary]
    valid_clusters, overlap_candidates = _apply_overlap_repetition_filter(
        detections=detections_after_conf,
        min_repetitions=min_repetitions,
        overlap_threshold=overlap_threshold,
    )
    final_detections = _dedupe_word_detections(overlap_candidates)
    extracted_text = _make_text_from_detections(final_detections)

    boxed_image_bgr = image_bgr.copy()
    final_boxes = 0
    for det in final_detections:
        x1, y1 = max(det.x, 0), max(det.y, 0)
        x2 = min(det.x + det.w, image_bgr.shape[1] - 1)
        y2 = min(det.y + det.h, image_bgr.shape[0] - 1)
        cv2.rectangle(boxed_image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        final_boxes += 1

    text_file = output_dir / f"{stem}.txt"
    boxed_image_file = output_dir / f"{stem}_boxed.png"
    gray_file = output_dir / f"{stem}_gray.png"
    debug_report_file = output_dir / f"{stem}_debug_report.txt"

    text_lines = ["# final_word\tx\ty\tw\th\tconf\tsource_pass"]
    for det in final_detections:
        text_lines.append(
            f"{det.text}\t{det.x}\t{det.y}\t{det.w}\t{det.h}\t{det.conf:.2f}\t{det.source_pass}"
        )
    text_lines.append("")
    text_lines.append("# final_text")
    text_lines.append(extracted_text)
    text_file.write_text("\n".join(text_lines).rstrip() + "\n", encoding="utf-8")
    cv2.imwrite(str(boxed_image_file), boxed_image_bgr)
    cv2.imwrite(str(gray_file), gray)
    _write_debug_report(
        debug_report_path=debug_report_file,
        image_name=image_path.name,
        min_conf_primary=min_conf_primary,
        min_conf_secondary=min_conf_secondary,
        overlap_threshold=overlap_threshold,
        min_repetitions=min_repetitions,
        total_detections=len(raw_detections),
        detections_after_conf_filter=len(detections_after_conf),
        candidates_after_overlap=len(overlap_candidates),
        final_count=len(final_detections),
        valid_clusters=valid_clusters,
        all_detections=raw_detections,
        final_detections=final_detections,
        pass_debug_rows=pass_debug_rows,
    )

    print(f"[OK] {image_path.name}")
    print(f"     -> Texto: {text_file}")
    print(
        "     -> Cajas: "
        f"{boxed_image_file} (finales: {final_boxes})"
    )
    print(f"     -> Debug report: {debug_report_file}")


def main() -> None:
    """Entry point: validate args, run OCR for all input images, print execution summary."""
    args = parse_args()
    if args.min_conf is not None:
        args.min_conf_primary = args.min_conf

    if args.min_conf_secondary > args.min_conf_primary:
        print(
            "[WARN] min-conf-secondary era mayor que min-conf-primary. "
            "Se ajusta automaticamente al valor primario."
        )
        args.min_conf_secondary = args.min_conf_primary

    configured_exe = configure_tesseract_cmd(args.tesseract_cmd)
    print(f"Usando Tesseract: {configured_exe}")

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
    print("Modo OCR: advanced_unified")
    print(
        "Regla final: "
        f"min_repetitions={args.min_repetitions}, overlap_iou>={args.overlap_threshold}"
    )
    print("Debug por paso: activado (incluye lista completa de detecciones)")

    for image_path in images:
        ocr_image(
            image_path=image_path,
            output_dir=args.output_dir,
            min_conf_primary=args.min_conf_primary,
            min_conf_secondary=args.min_conf_secondary,
            save_debug_images=args.save_debug_images,
            overlap_threshold=args.overlap_threshold,
            min_repetitions=args.min_repetitions,
        )

    print(f"\nListo. Resultados en: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
