# -*- coding: utf-8 -*-
"""
AL-BAYAN# | Step 7: Inference — Classify New Documents
======================================================
Load trained ensemble models and predict sectors for new documents.

Usage:
    python 08_inference.py --text "نص الوثيقة هنا"
    python 08_inference.py --file document.txt
    python 08_inference.py --folder data/new_pdfs
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# ── Settings ──────────────────────────────────────────────────
MODEL_DIRS = {
    "arabert":   "models/arabert",
    "marbert":   "models/marbert",
    "camelbert": "models/camelbert",
}
SPECIALIST_DIR = "models/specialist"

ALL_SECTORS = [
    "أراضي وتنظيم", "مالية وضرائب", "تشريعات وقرارات عليا",
    "إدارة ووظيفة عامة", "عقوبات وجرائم", "أحوال شخصية",
    "عمل وضمان اجتماعي", "تجارة وشركات", "أشغال وبنية تحتية",
    "تعليم وبحث علمي", "صحة وسلامة عامة", "بيئة وزراعة",
    "أمن ودفاع", "سياحة وآثار", "إعلام ونشر",
    "نقل وسير", "قضاء وتنفيذ",
]
N = len(ALL_SECTORS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model architecture (same as 04_train.py) ─────────────────
class Classifier(nn.Module):
    def __init__(self, model_name, n=N):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(h, 256),
            nn.GELU(), nn.Dropout(0.15), nn.Linear(256, n)
        )

    def forward(self, ids, mask):
        return self.head(self.bert(ids, mask).pooler_output)


# ── Load all trained models ───────────────────────────────────
def load_models():
    """Load ensemble models + thresholds from saved checkpoints."""
    models = {}
    for name, mdir in MODEL_DIRS.items():
        ckpt_path = f"{mdir}/best_model.pt"
        tok_path  = f"{mdir}/tokenizer"
        mj_path   = f"{mdir}/metrics.json"
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {name} (not trained)")
            continue
        metrics    = json.load(open(mj_path))
        model_name = json.load(open(f"{mdir}/metrics.json")).get(
            "model", "aubmindlab/bert-base-arabertv2")
        tok  = AutoTokenizer.from_pretrained(tok_path)
        mdl  = Classifier(model_name).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        mdl.load_state_dict(ckpt["model"])
        mdl.eval()
        models[name] = {
            "model":     mdl,
            "tokenizer": tok,
            "thrs":      np.array(ckpt["thrs"]),
        }
        print(f"  Loaded {name}")
    return models


# ── Text preprocessing ────────────────────────────────────────
def truncate(text, max_words=600):
    """Preserve beginning + middle + end of long documents."""
    words = str(text).split()
    if len(words) <= max_words: return text
    c = max_words // 3; m = len(words) // 2
    return " ".join(words[:c] + ["[...]"] +
                    words[m-c//2:m+c//2] + ["[...]"] + words[-c:])


def clean_text(text):
    """Remove noise from Arabic text."""
    t = str(text)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[\u0610-\u061A\u064B-\u065F]", "", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\u0600-\u06FF\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ── Single model prediction ───────────────────────────────────
def predict_one(text, model_info, max_len=384):
    """Get probability vector from one model."""
    clean  = truncate(clean_text(text))
    tok    = model_info["tokenizer"]
    enc    = tok(clean, max_length=max_len, padding="max_length",
                 truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model_info["model"](enc["input_ids"],
                                      enc["attention_mask"])
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs


# ── Ensemble prediction ───────────────────────────────────────
def predict_ensemble(text, models):
    """
    Average probabilities across all loaded models.
    Apply per-sector thresholds from ensemble results.
    """
    all_probs = [predict_one(text, m) for m in models.values()]
    avg_probs = np.mean(all_probs, axis=0)

    # load ensemble thresholds if available
    ens_path = "models/ensemble/results.json"
    if os.path.exists(ens_path):
        ens = json.load(open(ens_path))
        thrs = np.array(ens.get("average", {}).get(
            "thresholds", [0.5] * N))
    else:
        thrs = np.full(N, 0.5)

    predictions = []
    for i, sec in enumerate(ALL_SECTORS):
        if avg_probs[i] >= thrs[i]:
            predictions.append({
                "sector":      sec,
                "probability": round(float(avg_probs[i]), 4),
            })

    predictions.sort(key=lambda x: x["probability"], reverse=True)
    return predictions, avg_probs


# ── Predict from PDF ──────────────────────────────────────────
def predict_pdf(pdf_path, models):
    """Extract text from PDF then classify."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
        from PIL import Image
        pages = convert_from_path(str(pdf_path), dpi=200)
        text  = " ".join(
            pytesseract.image_to_string(p, lang="ara+eng") for p in pages)
    except Exception as e:
        print(f"  OCR failed for {pdf_path}: {e}")
        text = ""
    return predict_ensemble(text, models) if text.strip() else ([], None)


# ── Format output ─────────────────────────────────────────────
def format_output(filename, predictions, probs):
    """Print clean results table."""
    print(f"\n{'='*55}")
    print(f"Document: {filename}")
    print(f"{'='*55}")
    if not predictions:
        print("  No sector classified (all below threshold)")
        return
    print(f"  {'Sector':35s} {'Probability':>12s}")
    print(f"  {'-'*49}")
    for p in predictions:
        print(f"  {p['sector']:35s} {p['probability']:>12.4f}")


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",   type=str, default="",
                        help="Direct text input")
    parser.add_argument("--file",   type=str, default="",
                        help="Path to .txt or .pdf file")
    parser.add_argument("--folder", type=str, default="",
                        help="Folder with PDF files")
    parser.add_argument("--output", type=str, default="",
                        help="Save results to CSV")
    args = parser.parse_args()

    print("Loading models...")
    models = load_models()
    if not models:
        print("No trained models found. Run 04_train.py first.")
        return

    results = []

    if args.text:
        preds, probs = predict_ensemble(args.text, models)
        format_output("input_text", preds, probs)
        results.append({"doc": "input_text",
                         "sectors": " | ".join(p["sector"] for p in preds)})

    elif args.file:
        path = Path(args.file)
        if path.suffix == ".pdf":
            preds, probs = predict_pdf(path, models)
        else:
            text  = path.read_text(encoding="utf-8")
            preds, probs = predict_ensemble(text, models)
        format_output(path.name, preds, probs)
        results.append({"doc": path.name,
                         "sectors": " | ".join(p["sector"] for p in preds)})

    elif args.folder:
        for pdf in sorted(Path(args.folder).glob("*.pdf")):
            preds, probs = predict_pdf(pdf, models)
            format_output(pdf.name, preds, probs)
            results.append({"doc": pdf.name,
                             "sectors": " | ".join(p["sector"] for p in preds)})

    if args.output and results:
        pd.DataFrame(results).to_csv(args.output, index=False,
                                      encoding="utf-8-sig")
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
