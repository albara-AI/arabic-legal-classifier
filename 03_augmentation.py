# -*- coding: utf-8 -*-
"""
AL-BAYAN# | Step 2: Data Augmentation
======================================
Expands rare sector documents using EDA techniques and AraBERT MLM.
Applies downsampling on dominant sector (legislation).

Usage:
    python 03_augmentation.py --input data/merged.xlsx --output data/augmented.xlsx
"""

import os
import re
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Settings ──────────────────────────────────────────────────
TARGET_MIN  = 120       # minimum documents per sector
EDA_COPIES  = 3         # augmented copies per rare document
MAX_RATIO   = 0.25      # maximum fraction for dominant sector
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Sector name unification map (old → new)
MERGE_MAP = {
    "طاقة وثروات": "أشغال وبنية تحتية",
    "صناعة وتجارة": "تجارة وشركات",
    "شؤون اجتماعية": "عمل وضمان اجتماعي",
    "اتصالات وتكنولوجيا": "إعلام ونشر",
}

ALL_SECTORS = [
    "أراضي وتنظيم", "مالية وضرائب", "تشريعات وقرارات عليا",
    "إدارة ووظيفة عامة", "عقوبات وجرائم", "أحوال شخصية",
    "عمل وضمان اجتماعي", "تجارة وشركات", "أشغال وبنية تحتية",
    "تعليم وبحث علمي", "صحة وسلامة عامة", "بيئة وزراعة",
    "أمن ودفاع", "سياحة وآثار", "إعلام ونشر",
    "نقل وسير", "قضاء وتنفيذ",
]

# Legal synonyms for Arabic (word → replacements)
SYNONYMS = {
    "قانون": ["تشريع", "نظام", "قرار"],
    "موظف": ["عامل", "مستخدم"],
    "شركة": ["مؤسسة", "منشأة"],
    "ضريبة": ["رسم", "عوائد"],
    "محكمة": ["قضاء", "هيئة قضائية"],
    "عقوبة": ["جزاء", "حكم"],
    "راتب": ["أجر", "مكافأة"],
    "أرض": ["عقار", "ملك"],
}

STOP_WORDS = {
    "في", "من", "إلى", "على", "عن", "مع", "المادة", "مادة",
    "رقم", "عدد", "الى", "وتعديلاته", "البند", "الفقرة",
}


# ── 1. Parse sector column ────────────────────────────────────
def parse_sectors(s):
    """Split 'sector1 | sector2' into a clean list of sector names."""
    if pd.isna(s) or not isinstance(s, str):
        return []
    parts = [p.strip() for p in re.split(r"[|¦,;/]", s) if p.strip()]
    result = []
    for sec in parts:
        mapped = MERGE_MAP.get(sec, sec)
        if mapped in ALL_SECTORS:
            result.append(mapped)
    return list(set(result))


# ── 2. Clean Arabic text ──────────────────────────────────────
def clean_text(text):
    """Remove noise: URLs, diacritics, digits, stop words, extra spaces."""
    if pd.isna(text) or not str(text).strip():
        return ""
    t = str(text)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[\u0610-\u061A\u064B-\u065F]", "", t)  # diacritics
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\u0600-\u06FF\s]", " ", t)
    words = [w for w in t.split() if w not in STOP_WORDS]
    return " ".join(words).strip()


# ── 3. EDA augmentation ───────────────────────────────────────
def eda_synonym(words, n=1):
    """Replace n words with legal synonyms."""
    result = words[:]
    candidates = [i for i, w in enumerate(result) if w in SYNONYMS]
    for i in random.sample(candidates, min(n, len(candidates))):
        result[i] = random.choice(SYNONYMS[result[i]])
    return result


def eda_delete(words, p=0.08):
    """Randomly delete each word with probability p."""
    result = [w for w in words if random.random() > p]
    return result if result else words


def eda_swap(words, n=1):
    """Swap positions of n word pairs."""
    result = words[:]
    for _ in range(n):
        i, j = random.sample(range(len(result)), 2)
        result[i], result[j] = result[j], result[i]
    return result


def augment_eda(text, copies=EDA_COPIES):
    """
    Generate `copies` augmented versions using EDA techniques in rotation:
    synonym → delete → swap → mix.
    """
    words = text.split()
    if len(words) < 5:
        return []
    funcs   = [eda_synonym, eda_delete, eda_swap]
    results = []
    for i in range(copies):
        fn   = funcs[i % len(funcs)]
        aug  = fn(words[:])
        results.append(" ".join(aug))
    return results


# ── 4. Downsample dominant sector ────────────────────────────
def downsample(df, sector="تشريعات وقرارات عليا", max_ratio=MAX_RATIO):
    """
    Reduce the dominant sector to max_ratio of total documents.
    Keeps all multi-label documents; removes only single-label ones.
    """
    total     = len(df)
    max_count = int(total * max_ratio)
    is_single = df["القطاعات"].apply(
        lambda s: parse_sectors(str(s)) == [sector])
    single_idx = df[is_single].index.tolist()
    sector_n   = (~df.index.isin(single_idx)).sum() + \
                 df[~is_single & df["القطاعات"].str.contains(
                     sector, na=False)].shape[0]

    if sector_n <= max_count:
        return df

    keep = max_count - (sector_n - len(single_idx))
    random.shuffle(single_idx)
    remove = set(single_idx[keep:])
    return df.drop(index=list(remove)).reset_index(drop=True)


# ── 5. Main augmentation loop ─────────────────────────────────
def augment_dataset(df):
    """
    For each rare sector (<TARGET_MIN docs), generate EDA copies
    until reaching the target minimum.
    """
    # count documents per sector
    counts = {s: 0 for s in ALL_SECTORS}
    for _, row in df.iterrows():
        for sec in parse_sectors(str(row["القطاعات"])):
            counts[sec] = counts.get(sec, 0) + 1

    rare = [s for s, c in counts.items() if 0 < c < TARGET_MIN]
    print(f"Rare sectors: {rare}")

    new_rows = []
    for sec in tqdm(rare, desc="Augmenting"):
        subset = df[df["القطاعات"].apply(
            lambda s: sec in parse_sectors(str(s)))]
        needed = TARGET_MIN - counts[sec]
        docs   = subset.to_dict("records")

        generated = 0
        while generated < needed:
            for doc in docs:
                if generated >= needed:
                    break
                augmented = augment_eda(clean_text(str(doc["text"])))
                for aug_text in augmented:
                    if generated >= needed:
                        break
                    new_rows.append({
                        "اسم الملف": doc["اسم الملف"] + f"_aug{generated}",
                        "القطاعات":  doc["القطاعات"],
                        "text":      aug_text,
                    })
                    generated += 1

    aug_df = pd.DataFrame(new_rows)
    result = pd.concat([df, aug_df], ignore_index=True)
    # clean all texts
    result["text"] = result["text"].apply(clean_text)
    return result


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/merged.xlsx")
    parser.add_argument("--output", default="data/augmented.xlsx")
    args = parser.parse_args()

    print("=" * 60)
    print("AL-BAYAN# | Data Augmentation")
    print("=" * 60)

    df = pd.read_excel(args.input, engine="openpyxl")
    print(f"Loaded: {len(df)} documents")

    # step 1: downsample dominant sector
    df = downsample(df)
    print(f"After downsampling: {len(df)}")

    # step 2: augment rare sectors
    df = augment_dataset(df)
    print(f"After augmentation: {len(df)}")

    df.to_excel(args.output, index=False, engine="openpyxl")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
