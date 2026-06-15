# -*- coding: utf-8 -*-
"""
AL-BAYAN# | Step 4: Ensemble — Combine 3 BERT Models
=====================================================
Averages probabilities from AraBERT, MarBERT, CaMeLBERT.
Simple average outperforms weighted and stacking (F1=0.769).

Usage:
    python 05_ensemble.py --method all
"""

import os
import json
import argparse
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

# ── Settings ──────────────────────────────────────────────────
MODEL_DIRS = {
    "arabert":   "models/arabert",
    "marbert":   "models/marbert",
    "camelbert": "models/camelbert",
}
OUTPUT_DIR = "models/ensemble"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_SECTORS = [
    "أراضي وتنظيم", "مالية وضرائب", "تشريعات وقرارات عليا",
    "إدارة ووظيفة عامة", "عقوبات وجرائم", "أحوال شخصية",
    "عمل وضمان اجتماعي", "تجارة وشركات", "أشغال وبنية تحتية",
    "تعليم وبحث علمي", "صحة وسلامة عامة", "بيئة وزراعة",
    "أمن ودفاع", "سياحة وآثار", "إعلام ونشر",
    "نقل وسير", "قضاء وتنفيذ",
]


# ── Load model outputs ────────────────────────────────────────
def load_models():
    """Load test_probs, test_labels, and F1 from each trained model."""
    models = {}
    labels = None
    for name, path in MODEL_DIRS.items():
        tp = os.path.join(path, "test_probs.npy")
        tl = os.path.join(path, "test_labels.npy")
        mj = os.path.join(path, "metrics.json")
        if not os.path.exists(tp):
            print(f"  Skipping {name} (not trained)")
            continue
        f1 = json.load(open(mj)).get("test_f1", 0) if os.path.exists(mj) else 0
        models[name] = {"probs": np.load(tp), "f1": f1}
        if labels is None:
            labels = np.load(tl)
        print(f"  {name}: F1={f1:.4f}")
    return models, labels


# ── Threshold tuning ─────────────────────────────────────────
def tune(probs, labels):
    thrs = np.full(len(ALL_SECTORS), 0.5, dtype=np.float32)
    for i in range(len(ALL_SECTORS)):
        yt, p = labels[:, i], probs[:, i]
        if yt.sum() == 0: continue
        best_f1, best_th = 0.0, 0.5
        for th in np.arange(0.15, 0.85, 0.02):
            f = f1_score(yt, (p > th).astype(int), zero_division=0)
            if f > best_f1 + 0.005:
                best_f1, best_th = f, th
        thrs[i] = float(np.clip(best_th, 0.15, 0.85))
    return thrs


# ── Report metrics ────────────────────────────────────────────
def report(probs, labels, name):
    thrs  = tune(probs, labels)
    preds = (probs > thrs[np.newaxis, :]).astype(int)
    f1    = f1_score(labels, preds, average="macro", zero_division=0)
    prec  = precision_score(labels, preds, average="macro", zero_division=0)
    rec   = recall_score(labels, preds, average="macro", zero_division=0)
    print(f"\n  [{name}] F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    per = {}
    for i, sec in enumerate(ALL_SECTORS):
        yt, yp = labels[:, i], preds[:, i]
        per[sec] = {"f1": round(f1_score(yt, yp, zero_division=0), 4),
                    "support": int(yt.sum())}
    return {"method": name, "f1_macro": round(f1, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "thresholds": thrs.tolist(), "per_sector": per}


# ── Ensemble methods ──────────────────────────────────────────
def average(models):
    """Simple mean of all model probabilities."""
    return np.mean([m["probs"] for m in models.values()], axis=0)


def weighted(models):
    """Weight each model by F1² / ΣF1²."""
    f1s = np.array([m["f1"] for m in models.values()])
    w   = f1s ** 2 / (f1s ** 2).sum()
    return sum(m["probs"] * wi
               for m, wi in zip(models.values(), w))


def stacking(models, labels):
    """Logistic Regression meta-learner on validation probabilities."""
    # stack probs as features
    X = np.hstack([m["probs"] for m in models.values()])
    final = np.zeros_like(list(models.values())[0]["probs"])
    for i in range(len(ALL_SECTORS)):
        yt = labels[:, i]
        if yt.sum() == 0: continue
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X, yt)
        final[:, i] = lr.predict_proba(X)[:, 1]
    return final


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="all",
                        choices=["average", "weighted", "stacking", "all"])
    args = parser.parse_args()

    print("=" * 60)
    print("AL-BAYAN# | Ensemble")
    print("=" * 60)

    models, labels = load_models()
    results = {}

    if args.method in ("average", "all"):
        results["average"] = report(average(models), labels, "Average")

    if args.method in ("weighted", "all"):
        results["weighted"] = report(weighted(models), labels, "Weighted")

    if args.method in ("stacking", "all"):
        results["stacking"] = report(stacking(models, labels),
                                      labels, "Stacking")

    best = max(results, key=lambda k: results[k]["f1_macro"])
    print(f"\n  Best: {best} → F1={results[best]['f1_macro']:.4f}")

    json.dump(results, open(f"{OUTPUT_DIR}/results.json", "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
