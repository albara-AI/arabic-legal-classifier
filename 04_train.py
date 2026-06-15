# -*- coding: utf-8 -*-
"""
AL-BAYAN# | Step 3: Train BERT Model with Hard Negative Mining
==============================================================
Trains one of AraBERT / MarBERT / CaMeLBERT for multi-label
classification into 17 legal sectors.

Usage:
    python 04_train.py --model arabert
    python 04_train.py --model camelbert --resume --extra-epochs 10
"""

import os
import re
import gc
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

# ── Model registry ────────────────────────────────────────────
MODELS = {
    "arabert":   {"hf": "aubmindlab/bert-base-arabertv2",
                  "bert_lr": 1e-5, "head_lr": 5e-5},
    "marbert":   {"hf": "UBC-NLP/MARBERT",
                  "bert_lr": 1e-5, "head_lr": 5e-5},
    "camelbert": {"hf": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
                  "bert_lr": 1e-5, "head_lr": 5e-5},
}

# ── Hyperparameters ───────────────────────────────────────────
MAX_LEN    = 384
BATCH      = 4
ACCUM      = 8         # gradient accumulation → effective batch = 32
EPOCHS     = 30
PATIENCE   = 7
HNM_START  = 3         # epoch to begin Hard Negative Mining
HNM_W      = 3.0       # weight multiplier for hard samples
HNM_TOP    = 0.25      # fraction of samples considered hard
SEED       = 42

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

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Helpers ───────────────────────────────────────────────────
def parse_sectors(s):
    if pd.isna(s): return []
    return [p.strip() for p in re.split(r"[|¦,;/]", str(s)) if p.strip()]


def truncate(text, max_words=600):
    """Keep beginning + middle + end to preserve all document zones."""
    words = str(text).split()
    if len(words) <= max_words: return text
    c = max_words // 3
    m = len(words) // 2
    return " ".join(words[:c] + ["[...]"] +
                    words[m-c//2:m+c//2] + ["[...]"] + words[-c:])


# ── Dataset ───────────────────────────────────────────────────
class LegalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts   = texts
        self.labels  = labels
        self.tok     = tokenizer
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


# ── Model ─────────────────────────────────────────────────────
class Classifier(nn.Module):
    """BERT → Dropout → Linear(768→256) → GELU → Dropout → Linear(256→17)"""
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


# ── Focal Loss ────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """BCE with focal weighting (gamma=2) and label smoothing."""
    def __init__(self, pos_weight, gamma=2.0, smooth=0.05):
        super().__init__()
        self.gamma  = gamma
        self.smooth = smooth
        self.register_buffer("pw", pos_weight)

    def forward(self, logits, targets, sw=None):
        t   = targets * (1 - self.smooth) + 0.5 * self.smooth
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, t, pos_weight=self.pw, reduction="none")
        p   = torch.sigmoid(logits)
        pt  = p * targets + (1 - p) * (1 - targets)
        loss = (1 - pt).pow(self.gamma) * bce
        if sw is not None:
            loss = loss * sw.unsqueeze(1)
        return loss.mean()


# ── Class weights ─────────────────────────────────────────────
def class_weights(labels, beta=0.9999, max_w=12.0):
    """Effective Number of Samples weighting for imbalanced classes."""
    pos = labels.sum(0).astype(np.float64)
    eff = 1 - np.power(beta, pos)
    w   = (1 - beta) / (eff + 1e-8)
    w   = w / w.mean()
    return torch.tensor(np.clip(w, 0.3, max_w), dtype=torch.float32)


# ── Hard Negative Mining ──────────────────────────────────────
def update_hnm(model, dataset):
    """Mark hardest 25% of training samples with weight=3.0."""
    model.eval()
    losses = np.zeros(len(dataset), dtype=np.float32)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    with torch.no_grad():
        for b in loader:
            ids  = b["input_ids"].to(DEVICE)
            mask = b["attention_mask"].to(DEVICE)
            lbls = b["labels"].to(DEVICE)
            idxs = b["idx"].numpy()
            logits = model(ids, mask)
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, lbls, reduction="none")
            for i, idx in enumerate(idxs):
                losses[idx] = bce[i].mean().item()
    thresh = np.sort(losses)[-int(len(losses) * HNM_TOP)]
    w = np.ones(len(losses), dtype=np.float32)
    w[losses >= thresh] = HNM_W
    dataset.weights = w


# ── Threshold tuning ──────────────────────────────────────────
def tune_thresholds(probs, labels):
    """Find per-sector threshold maximizing F1 on validation set."""
    thrs = np.full(N, 0.5, dtype=np.float32)
    for i in range(N):
        yt, p = labels[:, i], probs[:, i]
        if yt.sum() == 0: continue
        best_f1, best_th = 0.0, 0.5
        for th in np.arange(0.15, 0.85, 0.02):
            f = f1_score(yt, (p > th).astype(int), zero_division=0)
            if f > best_f1 + 0.005:
                best_f1, best_th = f, th
        thrs[i] = float(np.clip(best_th, 0.15, 0.85))
    return thrs


# ── Evaluate ─────────────────────────────────────────────────
def evaluate(model, loader, criterion, thrs):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for b in loader:
            ids  = b["input_ids"].to(DEVICE)
            mask = b["attention_mask"].to(DEVICE)
            lbls = b["labels"]
            probs = torch.sigmoid(model(ids, mask)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(lbls.numpy())
    probs  = np.vstack(all_probs)
    labels = np.vstack(all_labels)
    preds  = (probs > thrs[np.newaxis, :]).astype(float)
    f1     = f1_score(labels, preds, average="macro", zero_division=0)
    return f1, probs, labels


# ── Training loop ─────────────────────────────────────────────
def train(model_key, resume=False, extra_epochs=10):
    cfg       = MODELS[model_key]
    out_dir   = f"models/{model_key}"
    ckpt_path = f"{out_dir}/best_model.pt"
    os.makedirs(out_dir, exist_ok=True)

    # load and prepare data
    df = pd.read_excel("data/augmented.xlsx", engine="openpyxl")
    df = df.drop_duplicates(subset=["اسم الملف"]).reset_index(drop=True)
    texts = df["text"].tolist()

    # build multi-label matrix [n × 17]
    labels = np.zeros((len(df), N), dtype=np.float32)
    for i, row in df.iterrows():
        for sec in parse_sectors(str(row["القطاعات"])):
            if sec in ALL_SECTORS:
                labels[i, ALL_SECTORS.index(sec)] = 1.0

    # split 80/10/10
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=SEED)

    tok      = AutoTokenizer.from_pretrained(cfg["hf"])
    train_ds = LegalDataset(X_tr,   y_tr,   tok)
    val_ds   = LegalDataset(X_val,  y_val,  tok)
    test_ds  = LegalDataset(X_test, y_test, tok)

    train_loader = DataLoader(train_ds, BATCH, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  BATCH, shuffle=False, num_workers=2)

    model    = Classifier(cfg["hf"]).to(DEVICE)
    pos_w    = class_weights(y_tr).to(DEVICE)
    criterion = FocalLoss(pos_w)

    # two learning rates: BERT layers vs classification head
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    params   = [
        {"params": [p for n, p in model.bert.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "lr": cfg["bert_lr"], "weight_decay": 0.01},
        {"params": [p for n, p in model.bert.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "lr": cfg["bert_lr"], "weight_decay": 0.0},
        {"params": model.head.parameters(),
         "lr": cfg["head_lr"], "weight_decay": 0.01},
    ]
    optimizer = AdamW(params, eps=1e-8)
    total_steps = (len(train_loader) // ACCUM) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler()

    best_f1, thrs = 0.0, np.full(N, 0.5, dtype=np.float32)
    patience_cnt  = 0
    start_epoch   = 0
    n_epochs      = EPOCHS

    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_f1     = ckpt["f1"]
        thrs        = np.array(ckpt["thrs"])
        n_epochs    = start_epoch + extra_epochs
        model = model.to(DEVICE)
        print(f"Resumed from epoch {start_epoch}, F1={best_f1:.4f}")

    # save initial model to avoid FileNotFoundError at evaluation
    torch.save({"epoch": -1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "f1": 0.0, "thrs": thrs.tolist()}, ckpt_path)

    for epoch in range(start_epoch, n_epochs):
        if epoch >= HNM_START:
            update_hnm(model, train_ds)

        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            sw   = batch["weight"].to(DEVICE)

            with torch.cuda.amp.autocast():
                loss = criterion(model(ids, mask), lbls, sw) / ACCUM

            scaler.scale(loss).backward()
            epoch_loss += loss.item()

            if (step + 1) % ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        val_f1, val_probs, val_labels, = evaluate(model, val_loader,
                                                    criterion, thrs)[:3]
        print(f"  loss={epoch_loss/len(train_loader):.4f}  "
              f"val_f1={val_f1:.4f}  best={best_f1:.4f}")

        if val_f1 > best_f1 or epoch == start_epoch:
            best_f1 = max(val_f1, best_f1)
            thrs    = tune_thresholds(val_probs, val_labels)
            patience_cnt = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "f1": best_f1, "thrs": thrs.tolist()}, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping.")
                break

    # final test evaluation
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(DEVICE)
    thrs  = np.array(ckpt["thrs"])

    test_f1, test_probs, test_labels = evaluate(
        model, test_loader, criterion, thrs)[:3]
    print(f"\nTest F1-Macro: {test_f1:.4f}")

    np.save(f"{out_dir}/test_probs.npy",  test_probs)
    np.save(f"{out_dir}/test_labels.npy", test_labels)
    json.dump({"model": model_key, "test_f1": round(test_f1, 4),
               "thrs": thrs.tolist()},
              open(f"{out_dir}/metrics.json", "w"), ensure_ascii=False)
    tok.save_pretrained(f"{out_dir}/tokenizer")


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="arabert",
                        choices=list(MODELS))
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--extra-epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.model, args.resume, args.extra_epochs)


if __name__ == "__main__":
    main()
