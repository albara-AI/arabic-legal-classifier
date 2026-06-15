# -*- coding: utf-8 -*-
"""
AL-BAYAN# | Step 6: Semantic Tag Generation (AraT5 + RAG Hybrid)
================================================================
Generates semantic tags for legal documents.
AraT5 generates → BiEncoder validates → fallback to RAG if confidence < 0.6.

Usage:
    python 07_generation.py --mode train
    python 07_generation.py --mode eval
    python 07_generation.py --mode tag --text "نص وثيقة قانونية"
"""

import re
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModel,
                           T5ForConditionalGeneration, T5Tokenizer)
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ── Settings ──────────────────────────────────────────────────
T5_MODEL      = "UBC-NLP/AraT5v2-base-1024"
BERT_MODEL    = "aubmindlab/bert-base-arabertv2"
DATA_FILE     = "data/augmented.xlsx"
OUTPUT_DIR    = "models/generation"
MAX_LEN_IN    = 512
MAX_LEN_OUT   = 64
BATCH         = 8
EPOCHS        = 10
LR            = 3e-5
TEMPERATURE   = 20.0   # InfoNCE sharpness
CONFIDENCE_TH = 0.60   # min cosine similarity to accept AraT5 output
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Text normalization ─────────────────────────────────────
def normalize(text):
    """Unify Arabic letter forms (alef variants, hamza, ta-marbuta)."""
    if pd.isna(text): return ""
    t = str(text)
    t = re.sub(r"[أإآ]", "ا", t)
    t = re.sub(r"ة$", "ه", t)
    t = re.sub(r"ى", "ي", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def clean(text):
    """Remove URLs, diacritics, non-Arabic characters."""
    t = str(text)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[\u0610-\u061A\u064B-\u065F]", "", t)
    t = re.sub(r"[^\u0600-\u06FF\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ── 2. AraT5 dataset ──────────────────────────────────────────
class T5Dataset(Dataset):
    """
    Seq2Seq format: input = 'صنّف: {text}' → target = '{semantic_tag}'
    """
    def __init__(self, texts, targets, tokenizer):
        self.texts    = [f"صنّف: {clean(normalize(t))}" for t in texts]
        self.targets  = [normalize(str(t)) for t in targets]
        self.tok      = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], max_length=MAX_LEN_IN,
                       padding="max_length", truncation=True,
                       return_tensors="pt")
        dec = self.tok(self.targets[i], max_length=MAX_LEN_OUT,
                       padding="max_length", truncation=True,
                       return_tensors="pt")
        labels = dec["input_ids"].clone()
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels":         labels.flatten(),
        }


# ── 3. BiEncoder (InfoNCE) ────────────────────────────────────
class BiEncoder(nn.Module):
    """
    Dual AraBERT encoders with shared weights.
    Trained with In-Batch Negatives (InfoNCE).
    Temperature=20 sharpens cosine similarity boundaries.
    """
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(BERT_MODEL)

    def encode(self, ids, mask):
        out = self.enc(ids, mask).pooler_output
        return nn.functional.normalize(out, dim=-1)  # L2 normalize

    def forward(self, q_ids, q_mask, t_ids, t_mask):
        q = self.encode(q_ids, q_mask)
        t = self.encode(t_ids, t_mask)
        # cosine similarity matrix × temperature
        sim = torch.matmul(q, t.T) * TEMPERATURE
        # InfoNCE: diagonal = positive pairs
        labels = torch.arange(sim.size(0), device=sim.device)
        return nn.functional.cross_entropy(sim, labels)


# ── 4. InfoNCE dataset ────────────────────────────────────────
class InfoNCEDataset(Dataset):
    """Pairs of (document_text, semantic_tag) for contrastive learning."""
    def __init__(self, texts, targets, tokenizer):
        self.texts   = [clean(normalize(t)) for t in texts]
        self.targets = [normalize(str(t)) for t in targets]
        self.tok     = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        def enc(t):
            e = self.tok(t, max_length=256, padding="max_length",
                         truncation=True, return_tensors="pt")
            return e["input_ids"].flatten(), e["attention_mask"].flatten()
        q_ids, q_mask = enc(self.texts[i])
        t_ids, t_mask = enc(self.targets[i])
        return {"q_ids": q_ids, "q_mask": q_mask,
                "t_ids": t_ids, "t_mask": t_mask}


# ── 5. Build retrieval index ──────────────────────────────────
def build_index(bi_encoder, targets, tokenizer):
    """
    Pre-compute embeddings for all semantic tags.
    Stored once; searched at inference time.
    """
    bi_encoder.eval()
    unique  = list(set(targets))
    embeds  = []
    with torch.no_grad():
        for tag in tqdm(unique, desc="Building index"):
            enc = tokenizer(tag, return_tensors="pt",
                            max_length=64, truncation=True,
                            padding="max_length").to(DEVICE)
            e   = bi_encoder.encode(enc["input_ids"],
                                     enc["attention_mask"])
            embeds.append(e.cpu().numpy())
    return unique, np.vstack(embeds)


# ── 6. ROUGE metric ───────────────────────────────────────────
def rouge_l(hyp, ref):
    """Simplified ROUGE-L for Arabic (normalize before comparison)."""
    hyp = normalize(hyp).split()
    ref = normalize(ref).split()
    if not hyp or not ref: return 0.0
    # LCS length
    m, n  = len(hyp), len(ref)
    dp    = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if hyp[i-1] == ref[j-1] \
                       else max(dp[i-1][j], dp[i][j-1])
    lcs  = dp[m][n]
    prec = lcs / m if m else 0
    rec  = lcs / n if n else 0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


# ── 7. Hybrid tagger ─────────────────────────────────────────
def hybrid_auto_tagger(text, sector, t5_model, t5_tok,
                        bi_encoder, bert_tok,
                        target_list, target_embeds, top_k=5):
    """
    1. AraT5 generates a candidate tag.
    2. BiEncoder computes cosine similarity with nearest known tag.
    3. If similarity >= CONFIDENCE_TH → accept AraT5 output.
       Else → return top-K retrieved tags (RAG fallback).
    """
    clean_text = clean(normalize(text))

    # AraT5 generation
    t5_model.eval()
    inp = t5_tok(f"صنّف: {clean_text}", return_tensors="pt",
                 max_length=MAX_LEN_IN, truncation=True).to(DEVICE)
    with torch.no_grad():
        out   = t5_model.generate(**inp, max_new_tokens=MAX_LEN_OUT,
                                   num_beams=4, early_stopping=True)
        gen   = t5_tok.decode(out[0], skip_special_tokens=True)

    # BiEncoder validation
    bi_encoder.eval()
    with torch.no_grad():
        enc = bert_tok(normalize(gen), return_tensors="pt",
                       max_length=64, truncation=True,
                       padding="max_length").to(DEVICE)
        q_emb = bi_encoder.encode(enc["input_ids"],
                                   enc["attention_mask"]).cpu().numpy()
    sims = np.dot(q_emb, target_embeds.T).flatten()
    best_idx  = sims.argmax()
    best_sim  = sims[best_idx]
    best_tag  = target_list[best_idx]

    if best_sim >= CONFIDENCE_TH:
        return {"tag": gen, "source": "generation",
                "confidence": float(best_sim)}

    # RAG fallback: return top-K retrieved tags
    top_k_idx  = sims.argsort()[-top_k:][::-1]
    top_k_tags = [target_list[i] for i in top_k_idx]
    return {"tag": top_k_tags[0], "source": "retrieval",
            "candidates": top_k_tags, "confidence": float(sims[top_k_idx[0]])}


# ── 8. Train AraT5 ───────────────────────────────────────────
def train_t5(df):
    """Fine-tune AraT5 on legal document → tag pairs."""
    t5_tok = T5Tokenizer.from_pretrained(T5_MODEL)
    t5     = T5ForConditionalGeneration.from_pretrained(T5_MODEL).to(DEVICE)
    ds     = T5Dataset(df["text"].tolist(),
                       df["enhanced_semantic_sentence"].tolist(), t5_tok)
    ld     = DataLoader(ds, BATCH, shuffle=True, num_workers=2)
    opt    = AdamW(t5.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        t5.train()
        total = 0
        for b in tqdm(ld, desc=f"T5 Epoch {epoch+1}"):
            ids  = b["input_ids"].to(DEVICE)
            mask = b["attention_mask"].to(DEVICE)
            lbls = b["labels"].to(DEVICE)
            loss = t5(input_ids=ids, attention_mask=mask, labels=lbls).loss
            loss.backward()
            opt.step(); opt.zero_grad()
            total += loss.item()
        print(f"  T5 Loss: {total/len(ld):.4f}")

    t5.save_pretrained(f"{OUTPUT_DIR}/t5")
    t5_tok.save_pretrained(f"{OUTPUT_DIR}/t5_tokenizer")
    return t5, t5_tok


# ── 9. Train BiEncoder ────────────────────────────────────────
def train_biencoder(df):
    """Train BiEncoder with InfoNCE contrastive loss."""
    bert_tok = AutoTokenizer.from_pretrained(BERT_MODEL)
    bi_enc   = BiEncoder().to(DEVICE)
    ds       = InfoNCEDataset(df["text"].tolist(),
                               df["enhanced_semantic_sentence"].tolist(),
                               bert_tok)
    ld  = DataLoader(ds, BATCH, shuffle=True, num_workers=2)
    opt = AdamW(bi_enc.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        bi_enc.train()
        total = 0
        for b in tqdm(ld, desc=f"BiEnc Epoch {epoch+1}"):
            loss = bi_enc(b["q_ids"].to(DEVICE), b["q_mask"].to(DEVICE),
                          b["t_ids"].to(DEVICE), b["t_mask"].to(DEVICE))
            loss.backward()
            opt.step(); opt.zero_grad()
            total += loss.item()
        print(f"  BiEnc Loss: {total/len(ld):.4f}")

    torch.save(bi_enc.state_dict(), f"{OUTPUT_DIR}/biencoder.pt")
    return bi_enc, bert_tok


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train",
                        choices=["train", "eval", "tag"])
    parser.add_argument("--text", type=str, default="")
    args = parser.parse_args()

    df = pd.read_excel(DATA_FILE, engine="openpyxl")
    df = df.dropna(subset=["enhanced_semantic_sentence"])

    if args.mode == "train":
        print("Training AraT5...")
        t5, t5_tok = train_t5(df)
        print("Training BiEncoder...")
        bi_enc, bert_tok = train_biencoder(df)

    elif args.mode == "tag" and args.text:
        # load models
        t5_tok  = T5Tokenizer.from_pretrained(f"{OUTPUT_DIR}/t5_tokenizer")
        t5      = T5ForConditionalGeneration.from_pretrained(
            f"{OUTPUT_DIR}/t5").to(DEVICE)
        bert_tok = AutoTokenizer.from_pretrained(BERT_MODEL)
        bi_enc   = BiEncoder().to(DEVICE)
        bi_enc.load_state_dict(
            torch.load(f"{OUTPUT_DIR}/biencoder.pt", map_location=DEVICE))
        tgt_list, tgt_emb = build_index(
            bi_enc, df["enhanced_semantic_sentence"].tolist(), bert_tok)

        result = hybrid_auto_tagger(args.text, "", t5, t5_tok,
                                     bi_enc, bert_tok, tgt_list, tgt_emb)
        print(f"\nTag: {result['tag']}")
        print(f"Source: {result['source']}")
        print(f"Confidence: {result['confidence']:.3f}")


if __name__ == "__main__":
    main()
