# AL-BAYAN# — Jordanian Official Gazette Intelligence System

<p align="center">
  <img src="https://img.shields.io/badge/F1--Macro-0.769-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Arabic_NLP-AraBERT-teal?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Sectors-17-purple?style=for-the-badge"/>
</p>

> **AI-powered framework** that transforms unstructured Jordanian Official Gazette PDFs into  
> semantically classified, searchable legal intelligence across **17 government sectors**.

---

## 📌 Project Overview

The Jordanian Official Gazette contains decades of Arabic legal documents — laws, regulations, royal decrees — stored as scanned PDFs with no classification or semantic indexing. AL-BAYAN# automates the full pipeline:

```
PDF Documents → OCR → Arabic NLP → Multi-Label Classification → Semantic Tag Generation
```

**Final Result: F1-Macro = 0.769** (↑ +38% from TF-IDF baseline of 0.558)

---

## 🏆 Results

| Stage | Method | F1-Macro |
|-------|--------|----------|
| Baseline | TF-IDF + Logistic Regression | 0.558 |
| AraBERT v1 | Global Threshold | 0.648 |
| AraBERT v2 | Per-label Threshold | 0.644 |
| AraBERT + HNM | Hard Negative Mining | 0.682 |
| CaMeLBERT + HNM | Best single model | 0.695 |
| **Ensemble × 3** | **Simple Average** | **0.769** |

### Per-Sector Performance (Best Ensemble)

| Sector | F1 | Precision | Recall |
|--------|----|-----------|--------|
| أمن ودفاع (Security) | 0.973 | 1.000 | 0.947 |
| تعليم (Education) | 0.906 | 0.960 | 0.857 |
| أراضي (Land) | 0.909 | 1.000 | 0.833 |
| سياحة (Tourism) | 0.941 | 1.000 | 0.889 |
| عمل (Labor) | 0.508 | 0.484 | 0.536 |
| تشريعات (Legislation) | 0.597 | 0.496 | 0.750 |

---

## 🚀 Pipeline

```
01_fetch_data.py     → Scrape PDFs from moj.gov.jo
02_ocr.py            → Extract Arabic text from scanned PDFs
03_augmentation.py   → EDA + AraBERT contextual augmentation
04_train.py          → Train AraBERT / MarBERT / CaMeLBERT
05_ensemble.py       → Combine 3 models (F1=0.769)
06_specialist.py     → Specialist model for weak sectors
07_generation.py     → AraT5 + BiEncoder + RAG tag generation
08_inference.py      → Classify new documents
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/albara-AI/AL-BAYAN.git
cd AL-BAYAN

# 2. Install
pip install -r requirements.txt

# 3. Fetch data
python 01_fetch_data.py --pages 20

# 4. OCR
python 02_ocr.py --input data/pdfs --output data/ocr_results.csv

# 5. Augment
python 03_augmentation.py

# 6. Train (run all 3 models)
python 04_train.py --model arabert
python 04_train.py --model marbert
python 04_train.py --model camelbert

# 7. Ensemble
python 05_ensemble.py --method all

# 8. Classify a document
python 08_inference.py --text "نص قانون العمل الأردني..."
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AL-BAYAN#                            │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│  Fetch   │   OCR    │  Augment │  Train   │   Ensemble     │
│  Data    │ Tesseract│   EDA    │  BERT×3  │  Avg Probs     │
│  moj.gov │  PSM-6   │ AraBERT  │  Focal   │  F1=0.769      │
│          │  DPI=300 │   MLM    │  Loss+   │                │
│          │          │          │   HNM    │                │
└──────────┴──────────┴──────────┴──────────┴────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │         Generation (Optional)          │
                    │  AraT5 → BiEncoder → RAG Fallback      │
                    └───────────────────────────────────────┘
```

---

## 🔧 Key Techniques

| Technique | Why | Where |
|-----------|-----|-------|
| **Hard Negative Mining** | Focus training on difficult samples | `04_train.py` |
| **Focal Loss** (γ=2) | Handle class imbalance | `04_train.py` |
| **Per-label Thresholds** | Different decision boundary per sector | `04_train.py`, `05_ensemble.py` |
| **Effective Num Samples** | Class weight for rare sectors | `04_train.py` |
| **Simple Average Ensemble** | Diversity beats single model | `05_ensemble.py` |
| **InfoNCE Contrastive** | Learn semantic similarity | `07_generation.py` |
| **Hybrid RAG** | AraT5 + retrieval fallback | `07_generation.py` |

---

## 📁 Project Structure

```
AL-BAYAN/
├── 01_fetch_data.py        # Web scraper for Official Gazette
├── 02_ocr.py               # PDF → Arabic text (Tesseract)
├── 03_augmentation.py      # EDA data augmentation
├── 04_train.py             # BERT multi-label classifier
├── 05_ensemble.py          # 3-model ensemble
├── 06_specialist.py        # Specialist for weak sectors
├── 07_generation.py        # AraT5 + BiEncoder tag generation
├── 08_inference.py         # Predict on new documents
├── requirements.txt
├── README.md
└── data/                   # (gitignored — add your data here)
    ├── pdfs/
    ├── merged.xlsx
    └── augmented.xlsx
```

---

## 📦 Requirements

```
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
tqdm>=4.64.0
openpyxl>=3.0.10
requests>=2.28.0
beautifulsoup4>=4.11.0
pytesseract>=0.3.10
opencv-python>=4.7.0
Pillow>=9.4.0
pdf2image>=1.16.0
sentencepiece>=0.1.99
```

---

## 💡 Future Development

| Idea | Description | Impact |
|------|-------------|--------|
| **Cross-lingual** | Extend to other Arab countries' gazettes | High |
| **Active Learning** | Human-in-the-loop for ambiguous documents | High |
| **REST API** | FastAPI endpoint for real-time classification | Medium |
| **GPT-4 Augmentation** | Higher quality data expansion | High |
| **Semantic Search** | Vector database for document retrieval | Medium |
| **Hierarchical Classification** | Sub-sector classification beyond 17 | High |

---

## 👥 Team

| Name 
|------

| **Albara Aljaber** 
| Mohamd Olimat 
| Ahmad Bustanji 
| Hothifa Howary
| Shaden Abd 

**Supervisor:** Dr. Majdi Maabreh  
**Institution:** Hashemite University — Faculty of Prince Al-Hussein bin Abdallah II for IT

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for Arabic NLP · Hashemite University 2025
</p>
