# Methodology notes (short)

- Data source: Hugging Face dataset KFUPM-JRCAI/arabic-generated-abstracts
- Labels: human = 0 (original_abstract), AI = 1 ({model}_generated_abstract)
- Split: 70/15/15 done early with stratification.
- Preprocessing: remove diacritics, normalise, stopwords, ISRI stemmer.
- Features:
  1) TF-IDF
  2) Assigned stylometry: f1, f24, f47, f70, f93 (based on i=1, n=23)
  3) BERT embeddings (asafaya/bert-base-arabic), then a simple feedforward net
