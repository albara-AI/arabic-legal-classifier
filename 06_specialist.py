# -*- coding: utf-8 -*-
"""
AL-BAYAN# | Step 5: Specialist Model for Weak Sectors
======================================================
CaMeLBERT fine-tuned on 5 weak sectors + 'other' class.
Uses strategic train/test split: legislation 65/35, others 80/20.

Usage:
    python 06_specialist.py
    python 06_specialist.py --resume --extra-epochs 10
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

# ── Settings ──────────────────────────────────────────────────
MODEL_NAME  = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
OUTPUT_DIR  = "models/specialist"
BERT_LR     = 8e-6      # lower LR for specialist fine-tuning
HEAD_LR     = 4e-5
MAX_LEN     = 384
BATCH       = 4
ACCUM       = 8
EPOCHS      = 40
PATIENCE    = 12
SEED        = 42
OTHER       = "أخرى"

# Target weak sectors (lowest F1 in ensemble)
SECTORS = [
    "تشريعات وقرارات عليا",  # F1=0.597
    "إدارة ووظيفة عامة",     # F1=0.773 (included for context)
    "تجارة وشركات",           # F1=0.646
    "مالية وضرائب",           # F1=0.710
    "تعليم وبحث علمي",        # F1=0.906
]

ALL_SECTORS = [
    "أراضي وتنظيم", "مالية وضرائب", "تشريعات وقرارات عليا",
    "إدارة ووظيفة عامة", "عقوبات وجرائم", "أحوال شخصية",
    "عمل وضمان اجتماعي", "تجارة وشركات", "أشغال وبنية تحتية",
    "تعليم وبحث علمي", "صحة وسلامة عامة", "بيئة وزراعة",
    "أمن ودفاع", "سياحة وآثار", "إعلام ونشر",
    "نقل وسير", "قضاء وتنفيذ",
]
SECTOR_ID = {s: i for i, s in enumerate(ALL_SECTORS)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────
def parse_sectors(s):
    if pd.isna(s): return []
    return [p.strip() for p in re.split(r"[|¦,;/]", str(s)) if p.strip()]


def truncate(text, max_words=600):
    words = str(text).split()
    if len(words) <= max_words: return text
    c = max_words // 3; m = len(words) // 2
    return " ".join(words[:c] + ["[...]"] +
                    words[m-c//2:m+c//2] + ["[...]"] + words[-c:])


# ── Dataset & Model (same architecture as 04_train.py) ────────
class SpecialistDataset(Dataset):
    def __init__(self, texts, labels, tok):
        self.texts   = texts
        self.labels  = labels
        self.tok     = tok
        self.weights = np.ones(len(texts), dtype=np.float32)

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(truncate(self.texts[i]), max_length=MAX_LEN,
                       padding="max_length", truncation=True,
                       return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels":  torch.tensor(self.labels[i], dtype=torch.float32),
            "weight":  torch.tensor(self.weights[i], dtype=torch.float32),
            "idx":     torch.tensor(i, dtype=torch.long),
        }


class SpecialistClassifier(nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        h = self.bert.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(h, 256),
            nn.GELU(), nn.Dropout(0.15), nn.Linear(256, n_labels)
        )

    def forward(self, ids, mask):
        return self.head(self.bert(ids, mask).pooler_output)


class FocalLoss(nn.Module):
    def __init__(self, pos_weight, gamma=2.0, smooth=0.05):
        super().__init__()
        self.gamma  = gamma
        self.smooth = smooth
        self.register_buffer("pw", pos_weight)

    def forward(self, logits, targets, sw=None):
        t   = targets * (1 - self.smooth) + 0.5 * self.smooth
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, t, pos_weight=self.pw, reduction="none")
        pt  = torch.sigmoid(logits) * targets + \
              (1 - torch.sigmoid(logits)) * (1 - targets)
        loss = (1 - pt).pow(self.gamma) * bce
        if sw is not None:
            loss = loss * sw.unsqueeze(1)
        return loss.mean()


# ── Prepare specialist labels [n × (k+1)] ─────────────────────
def prepare_data(df):
    """
    Build labels for specialist model.
    Sectors in SECTORS → specific label. All others → OTHER label.
    """
    spec_sectors = SECTORS + [OTHER]
    sec_to_idx   = {s: i for i, s in enumerate(spec_sectors)}
    texts, labels = [], []

    for _, row in df.iterrows():
        doc_secs = parse_sectors(str(row["القطاعات"]))
        label    = np.zeros(len(spec_sectors), dtype=np.float32)
        found    = False
        for sec in doc_secs:
            if sec in sec_to_idx:
                label[sec_to_idx[sec]] = 1.0
                found = True
        if not found:
            label[sec_to_idx[OTHER]] = 1.0
        texts.append(str(row["text"]))
        labels.append(label)

    # balance 'other': cap at 1.5× largest target sector
    labels_arr  = np.array(labels, dtype=np.float32)
    other_i     = sec_to_idx[OTHER]
    n_other     = int((labels_arr[:, other_i] == 1.0).sum())
    target_max  = int(labels_arr[:, :other_i].sum(0).max())
    max_other   = int(target_max * 1.5)
    if n_other > max_other:
        other_idx  = np.where(labels_arr[:, other_i] == 1.0)[0]
        target_idx = np.where(labels_arr[:, other_i] != 1.0)[0]
        keep       = np.random.choice(other_idx, max_other, replace=False)
        keep_all   = np.concatenate([target_idx, keep])
        np.random.shuffle(keep_all)
        texts      = [texts[i] for i in keep_all]
        labels_arr = labels_arr[keep_all]
        print(f"  Balanced 'other': {n_other} → {max_other}")

    return np.array(texts), labels_arr, spec_sectors


# ── Strategic split ───────────────────────────────────────────
def split(texts, labels, spec_sectors):
    """
    Legislation: 65/35 (harder test due to high prevalence).
    Other sectors: 80/20.
    """
    if "تشريعات وقرارات عليا" not in spec_sectors:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            texts, labels, test_size=0.2, random_state=SEED)
        X_v, X_t, y_v, y_t = train_test_split(
            X_tmp, y_tmp, test_size=0.5, random_state=SEED)
        return X_tr, X_v, X_t, y_tr, y_v, y_t

    leg_i  = spec_sectors.index("تشريعات وقرارات عليا")
    is_leg = labels[:, leg_i] == 1.0

    def _split(X, y, ratio):
        try:
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X, y, test_size=1-ratio, random_state=SEED,
                stratify=y.argmax(1))
            X_v, X_t, y_v, y_t = train_test_split(
                X_tmp, y_tmp, test_size=0.5, random_state=SEED)
        except Exception:
            X_tr, X_tmp, y_tr, y_tmp = train_test_split(
                X, y, test_size=1-ratio, random_state=SEED)
            X_v, X_t, y_v, y_t = train_test_split(
                X_tmp, y_tmp, test_size=0.5, random_state=SEED)
        return X_tr, X_v, X_t, y_tr, y_v, y_t

    tr_l, v_l, t_l, ytr_l, yv_l, yt_l = _split(
        texts[is_leg], labels[is_leg], 0.65)
    tr_o, v_o, t_o, ytr_o, yv_o, yt_o = _split(
        texts[~is_leg], labels[~is_leg], 0.80)

    def cat(a, b):
        c = np.concatenate([a, b])
        p = np.random.permutation(len(c))
        return c[p]

    return (cat(tr_l, tr_o), cat(v_l, v_o), cat(t_l, t_o),
            cat(ytr_l, ytr_o), cat(yv_l, yv_o), cat(yt_l, yt_o))


# ── Expand back to 17 sectors ─────────────────────────────────
def expand(probs, spec_sectors):
    """Map specialist [n×k] probabilities back to full [n×17]."""
    full = np.zeros((probs.shape[0], len(ALL_SECTORS)), dtype=np.float32)
    for i, sec in enumerate(spec_sectors):
        if sec == OTHER or sec not in SECTOR_ID: continue
        full[:, SECTOR_ID[sec]] = probs[:, i]
    return full


# ── Main training ─────────────────────────────────────────────
def train(resume=False, extra_epochs=10):
    df    = pd.read_excel("data/augmented.xlsx", engine="openpyxl")
    texts, labels, spec_sectors = prepare_data(df)
    X_tr, X_v, X_t, y_tr, y_v, y_t = split(texts, labels, spec_sectors)

    n_labels = labels.shape[1]
    tok      = AutoTokenizer.from_pretrained(MODEL_NAME)
    tr_ds    = SpecialistDataset(X_tr, y_tr, tok)
    val_ds   = SpecialistDataset(X_v,  y_v,  tok)
    tst_ds   = SpecialistDataset(X_t,  y_t,  tok)

    tr_ld  = DataLoader(tr_ds,  BATCH, shuffle=True,  num_workers=4)
    val_ld = DataLoader(val_ds, BATCH, shuffle=False, num_workers=2)
    tst_ld = DataLoader(tst_ds, BATCH, shuffle=False, num_workers=2)

    model    = SpecialistClassifier(n_labels).to(DEVICE)
    ckpt_path = f"{OUTPUT_DIR}/best_model.pt"

    pos_w = torch.ones(n_labels, dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(pos_w)

    optimizer = AdamW([
        {"params": model.bert.parameters(),  "lr": BERT_LR},
        {"params": model.head.parameters(),  "lr": HEAD_LR},
    ], eps=1e-8)
    total_steps = (len(tr_ld) // ACCUM) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler()

    best_f1, thrs = 0.0, np.full(n_labels, 0.5, dtype=np.float32)
    patience_cnt  = 0
    n_epochs      = EPOCHS

    # save initial checkpoint
    torch.save({"epoch": -1, "model": model.state_dict(),
                "thrs": thrs.tolist(), "f1": 0.0,
                "spec_sectors": spec_sectors}, ckpt_path)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(tr_ld, desc=f"Epoch {epoch+1}")):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            with torch.cuda.amp.autocast():
                loss = criterion(model(ids, mask), lbls) / ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer); scaler.update()
                scheduler.step(); optimizer.zero_grad()

        # validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for b in val_ld:
                p = torch.sigmoid(model(b["input_ids"].to(DEVICE),
                                        b["attention_mask"].to(DEVICE)))
                all_probs.append(p.cpu().numpy())
                all_labels.append(b["labels"].numpy())
        vp, vl = np.vstack(all_probs), np.vstack(all_labels)

        # tune thresholds (min 0.30 to prevent collapse)
        thrs_new = np.full(n_labels, 0.5, dtype=np.float32)
        for i in range(n_labels):
            if vl[:, i].sum() == 0: continue
            best_f, best_t = 0.0, 0.5
            for t in np.arange(0.30, 0.85, 0.02):
                f = f1_score(vl[:, i], (vp[:, i] > t).astype(int),
                             zero_division=0)
                if f > best_f + 0.005:
                    best_f, best_t = f, t
            thrs_new[i] = float(np.clip(best_t, 0.30, 0.85))

        val_f1 = f1_score(vl, (vp > thrs_new).astype(int),
                          average="macro", zero_division=0)
        print(f"  val_f1={val_f1:.4f}  best={best_f1:.4f}")

        if val_f1 > best_f1 or epoch == 0:
            best_f1 = max(val_f1, best_f1)
            thrs    = thrs_new
            patience_cnt = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "thrs": thrs.tolist(), "f1": best_f1,
                        "spec_sectors": spec_sectors}, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping.")
                break

    # test
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"]); model = model.to(DEVICE)
    thrs = np.array(ckpt["thrs"])
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for b in tst_ld:
            p = torch.sigmoid(model(b["input_ids"].to(DEVICE),
                                    b["attention_mask"].to(DEVICE)))
            all_p.append(p.cpu().numpy())
            all_l.append(b["labels"].numpy())
    tp, tl = np.vstack(all_p), np.vstack(all_l)
    preds  = (tp > thrs).astype(int)
    f1     = f1_score(tl, preds, average="macro", zero_division=0)
    print(f"\nTest F1-Macro (specialist): {f1:.4f}")

    # expand to 17 sectors for ensemble
    full_probs = expand(tp, spec_sectors)
    np.save(f"{OUTPUT_DIR}/test_probs.npy", full_probs)
    json.dump({"f1": round(f1, 4), "thrs": thrs.tolist(),
               "spec_sectors": spec_sectors},
              open(f"{OUTPUT_DIR}/metrics.json", "w"), ensure_ascii=False)
    tok.save_pretrained(f"{OUTPUT_DIR}/tokenizer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--extra-epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.resume, args.extra_epochs)


if __name__ == "__main__":
    main()
