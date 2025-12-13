# MSIS-822 Final Project (Simple Version)

Goal: detect AI-generated Arabic abstracts vs human abstracts using the dataset:
`KFUPM-JRCAI/arabic-generated-abstracts` (Hugging Face).

This is a simple implementation focused on meeting the course requirements:
- download from Hugging Face with `datasets`
- Arabic preprocessing (normalisation, diacritics removal, stopwords, ISRI stemmer)
- EDA (basic stats + word clouds + top n-grams)
- features: TF-IDF + my 5 assigned stylometric features + BERT embeddings
- models: baseline NB + Logistic Regression + Linear SVM + simple NN on BERT embeddings
- evaluation: Accuracy / Balanced Accuracy / Precision / Recall / F1 / ROC-AUC + confusion matrices + error analysis
- interpretation: feature importance for the best traditional model

## Repository structure
- `notebooks/` : full pipeline notebook
- `src/` : preprocessing, feature extraction, modelling, visualisation utilities
- `data/` : raw/processed/external (kept empty in Git; dataset is downloaded at runtime)
- `reports/figures/` : saved plots (if enabled)
- `models/` : optional saved models (ignored by default)

## Quick start
1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Run the pipeline notebook:
- `notebooks/full_pipeline.ipynb`

Notes:
- The dataset is downloaded automatically by the notebook using `datasets`.
- BERT embeddings may be cached under `cache/` to speed up reruns (this folder is ignored by git).

## My assigned stylometric feature indices
Given i=1 and n=23:
Assigned 1-based indices = [1, 24, 47, 70, 93]

So I implement:
- `f1_total_chars`
- `f24_diff_punct_over_C`
- `f47_pronoun_count`
- `f70_first_person_count`
- `f93_arousal_mean`

To change them, edit:
- `src/data_preparation.py` (search for `ASSIGNED_FEATURES`)
