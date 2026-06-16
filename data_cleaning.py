# -*- coding: utf-8 -*-
"""
AL-BAYAN# | Data Cleaning — Arabic Text Preprocessing
======================================================
Extracted from: باك_اند (2).ipynb — CELL3
Full 10-step Arabic legal text cleaning pipeline.

Usage:
    from data_cleaning import preprocess_arabic
    clean_text = preprocess_arabic(raw_text)
"""

import re

# ── Titles to keep, then strip the name that follows ──────────
TITLES = [
    'السيد', 'الساده', 'الانسه', 'السيده', 'الدكتور', 'الدكتوره', 'المهندس',
    'المهندسه', 'المحامي', 'المحاميه', 'القاضي', 'القاضيه', 'المستدعي', 'ضده',
    'المتهم', 'المشتكي', 'المدعي', 'عليه', 'المميز', 'المجني', 'الوزير',
    'معالي', 'عطوفه', 'سعادة', 'سمو', 'جلالة', 'النائب', 'العين', 'المدعو',
    'المواطن', 'المكلف', 'رئيس', 'مدير', 'امين', 'عضو', 'دولة'
]

# ── Comprehensive stop words (legal + administrative + OCR noise) ──
STOP_WORDS = [
    'استنادا', 'بمقتضي', 'احكام', 'ماده', 'فقره', 'بند', 'قانون', 'نظام', 'قرار',
    'تعليمات', 'قرر', 'مجلس', 'وزراء', 'بالموافقه', 'المنشوره', 'الجريده', 'الرسميه',
    'عدد', 'تاريخ', 'لسنه', 'سنه', 'صفحه', 'بموجب', 'حسب', 'وفق', 'وذلك', 'اعلاه',
    'ادناه', 'المذكور', 'المشار', 'اليه', 'بشان', 'بخصوص', 'بملحق', 'صادر', 'عن', 'لدى',
    'يعمل', 'بها', 'اعتبار', 'من', 'نشر', 'تعديل', 'الغاء', 'استبدال', 'اضافه',
    'حذف', 'وارد', 'نص', 'عباره', 'كلمات', 'حلول', 'محل', 'تعدل', 'تلغي', 'تستبدل',
    'عام', 'سنه', 'عامه', 'جميع', 'كافه', 'فقط', 'غير', 'كان', 'يكون', 'كانت',
    'ذلك', 'تلك', 'هذا', 'هذه', 'هؤلاء', 'التي', 'الذي', 'الذين', 'اللذين', 'اللتين',
    'عند', 'بعد', 'قبل', 'خلال', 'اثناء', 'حيث', 'ان', 'انه', 'انها', 'انهم',
    'ص', 'ق', 'ط', 'ب', 'و', 'ف', 'ك', 'ل', 'م', 'ت', 'ن', 'ه', 'ي', 'أ'
]

# ── Common Arabic personal names (removed to reduce noise) ────
COMMON_NAMES = (
    r'\b(محمد|احمد|محمود|عبدالله|عبدالرحمن|خالد|عمر|علي|يوسف|ابراهيم'
    r'|حسين|حسن|مصطفي|فاطمه|عائشه|زينب|مريم|سارة|نور|ليلي|سلمي|هدي)\b'
)

# ── Allowed 2-letter words (functional/meaning particles) ─────
ALLOWED_2 = (
    r'(في|من|عن|مع|لو|ان|قد|هل|بل|لا|ما|يا|لم|لن|كي|هو|هي|هم'
    r'|او|اي|ام|بم|عم|اب|اخ|يد|دم|حق|حل|نص|رد|ضد|بت|صك|شك|عد|حد|سد)'
)

# ── Compile lists into regex patterns ──────────────────────────
TITLES_PATTERN    = r'(' + '|'.join(TITLES) + r')\s+((?:\w+\s*){1,3})'
STOPWORDS_PATTERN = r'\b(' + '|'.join(STOP_WORDS) + r')\b'


def preprocess_arabic(text: str) -> str:
    """
    Full Arabic legal text cleaning — 10 sequential steps:
      1. Unify letter forms + remove diacritics/tatweel
      2. Remove all digit forms (Arabic/Indic/Latin)
      3. Remove Latin letters and hidden control characters
      4. Strip titles+names and patronymic patterns (بن/بنت)
      5. Keep Arabic letters only (strip everything else)
      6. Remove legal/administrative stop words
      7. Remove abnormally long words (OCR errors > 15 chars)
      8. Remove words with repeated-letter noise (e.g. عسسسي)
      9. Remove 2-letter noise words and single letters
      10. Collapse whitespace

    Args:
        text: raw Arabic text (possibly OCR output)
    Returns:
        cleaned, normalized Arabic text
    """
    if not text:
        return ""

    # ── Step 1: unify letters, strip diacritics/tatweel ─────────
    text = re.sub(r'[\u0640]+', '', text)          # remove tatweel (ـ)
    text = re.sub(r'[\u064B-\u065F]+', '', text)   # remove diacritics
    text = re.sub(r'[إأآا]', 'ا', text)            # unify alef forms
    text = re.sub(r'ى', 'ي', text)                 # unify ya/alef-maksura
    text = re.sub(r'ة', 'ه', text)                 # unify ta-marbuta

    # ── Step 2: remove digits (Arabic/Indic/Latin) ───────────────
    text = re.sub(r'[0-9\u0660-\u0669\u06F0-\u06F9]+', ' ', text)

    # ── Step 3: remove Latin letters + hidden control chars ─────
    text = re.sub(r'[a-zA-Z]+', ' ', text)
    text = re.sub(r'[\u200e\u200f\u200b\u200c\u200d\u202a-\u202e]', ' ', text)

    # ── Step 4: handle titles, names, patronymics ────────────────
    text = re.sub(TITLES_PATTERN, r'\1 ', text)
    text = re.sub(r'\sعبد\s+\w+', ' ', text)
    text = re.sub(r'\w+\s+(?:بن|بنت)\s+\w+', ' ', text)
    text = re.sub(COMMON_NAMES, ' ', text, flags=re.IGNORECASE)

    # ── Step 5: keep Arabic letters only ─────────────────────────
    text = re.sub(r'[^\u0621-\u064A\s]', ' ', text)

    # ── Step 6: remove comprehensive stop words ──────────────────
    text = re.sub(STOPWORDS_PATTERN, ' ', text)

    # ── Step 7: remove overly long words (OCR noise > 15 chars) ──
    text = re.sub(r'\b[\u0621-\u064A]{16,}\b', ' ', text)

    # ── Step 8: remove words with 3+ repeated letters ────────────
    text = re.sub(r'\b[\u0621-\u064A]*([\u0621-\u064A])\1{2,}[\u0621-\u064A]*\b', ' ', text)

    # ── Step 9: remove 2-letter noise words and single letters ───
    text = re.sub(r'\b(?!' + ALLOWED_2 + r'\b)[\u0621-\u064A]{2}\b', ' ', text)
    text = re.sub(r'\b[\u0621-\u064A]\b', ' ', text)

    # ── Step 10: collapse whitespace ──────────────────────────────
    text = re.sub(r'\s+', ' ', text).strip()

    return text


if __name__ == "__main__":
    # quick self-test
    sample = "السيد محمد احمد، بموجب المادة 5 من القانون رقم 10 لسنة 2020"
    print("Before:", sample)
    print("After :", preprocess_arabic(sample))
