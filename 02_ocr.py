# -*- coding: utf-8 -*-
"""
AL-BAYAN# | Step 1: OCR — Extract Text from PDF
================================================
Converts scanned Arabic PDF files to clean text using Tesseract.
Includes image preprocessing, rotation correction, and checkpointing.

Usage:
    python 02_ocr.py --input data/pdfs --output data/ocr_results.csv
"""

import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd
import pytesseract
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm

# ── Settings ──────────────────────────────────────────────────
DPI         = 300        # render quality (150=fast, 300=accurate)
PSM_MODE    = 6          # Tesseract page segmentation: uniform text block
LANGUAGES   = "ara+eng"  # Arabic primary + English for mixed content
SAVE_EVERY  = 50         # checkpoint frequency


# ── 1. Fix page rotation ───────────────────────────────────────
def detect_and_fix_rotation(image):
    """Use Tesseract OSD to detect and correct page rotation."""
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        angle = osd.get("rotate", 0)
        if angle != 0:
            image = image.rotate(-angle, expand=True)
    except Exception:
        pass
    return image


# ── 2. Remove shadows from scan ───────────────────────────────
def remove_shadows(gray):
    """Normalize uneven lighting by subtracting estimated background."""
    kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    background = cv2.dilate(gray, kernel, iterations=1)
    diff       = cv2.absdiff(gray, background)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


# ── 3. Enhance text contrast ──────────────────────────────────
def enhance_text(gray):
    """Apply CLAHE for local contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ── 4. Full preprocessing pipeline ───────────────────────────
def preprocess_image(pil_image):
    """
    Pipeline: rotation fix → grayscale → denoise → threshold → erosion.
    Erosion thickens Arabic characters to close gaps before OCR.
    """
    pil_image = detect_and_fix_rotation(pil_image)
    img       = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray      = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # thicken Arabic strokes
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=2)
    return Image.fromarray(binary)


# ── 5. Remove table lines ─────────────────────────────────────
def remove_table_lines(img_array):
    """Erase horizontal and vertical table rules using morphology."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) \
           if len(img_array.shape) == 3 else img_array
    # horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines  = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel)
    # vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines  = cv2.morphologyEx(gray, cv2.MORPH_OPEN, v_kernel)
    # subtract lines from image
    cleaned = cv2.subtract(gray, h_lines)
    cleaned = cv2.subtract(cleaned, v_lines)
    return cleaned


# ── 6. Run Tesseract OCR ──────────────────────────────────────
def extract_text(pil_image):
    """Run Tesseract with PSM 6 + LSTM engine on preprocessed image."""
    config = f"--psm {PSM_MODE} --oem 3"
    text   = pytesseract.image_to_string(pil_image,
                                          lang=LANGUAGES,
                                          config=config)
    # clean empty lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


# ── 7. Checkpoint helpers ─────────────────────────────────────
def save_checkpoint(processed, path="data/ocr_checkpoint.json"):
    """Save list of processed filenames to allow resume."""
    with open(path, "w") as f:
        json.dump(list(processed), f)


def load_checkpoint(path="data/ocr_checkpoint.json"):
    """Load previously processed files. Returns empty set if missing."""
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


# ── 8. Process single PDF ─────────────────────────────────────
def process_pdf(pdf_path):
    """
    Convert PDF pages to images, preprocess, and extract text.
    Returns concatenated text from all pages.
    """
    pages = convert_from_path(str(pdf_path), dpi=DPI)
    all_text = []
    for page in pages:
        processed = preprocess_image(page)
        text      = extract_text(processed)
        if text.strip():
            all_text.append(text)
    return "\n\n".join(all_text)


# ── 9. Process entire folder ──────────────────────────────────
def process_folder(input_dir, output_csv, checkpoint_file):
    """
    Iterate over all PDFs in input_dir, extract text, save to CSV.
    Saves checkpoint every SAVE_EVERY files to support resume.
    """
    pdf_files  = sorted(Path(input_dir).glob("*.pdf"))
    done       = load_checkpoint(checkpoint_file)
    results    = []

    print(f"Found {len(pdf_files)} PDFs | Already processed: {len(done)}")

    for i, pdf_path in enumerate(tqdm(pdf_files, desc="OCR")):
        name = pdf_path.name
        if name in done:
            continue

        text = process_pdf(pdf_path)
        results.append({
            "filename": name,
            "text":     text,
            "chars":    len(text),
        })
        done.add(name)

        # checkpoint + partial save every N files
        if (i + 1) % SAVE_EVERY == 0:
            save_checkpoint(done, checkpoint_file)
            pd.DataFrame(results).to_csv(output_csv, index=False,
                                          encoding="utf-8-sig")

    # final save
    save_checkpoint(done, checkpoint_file)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nDone. {len(df)} files → {output_csv}")
    return df


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="OCR for Arabic PDF documents")
    parser.add_argument("--input",      default="data/pdfs")
    parser.add_argument("--output",     default="data/ocr_results.csv")
    parser.add_argument("--checkpoint", default="data/ocr_checkpoint.json")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    process_folder(args.input, args.output, args.checkpoint)


if __name__ == "__main__":
    main()
