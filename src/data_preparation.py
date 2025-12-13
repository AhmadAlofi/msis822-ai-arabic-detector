import os
import re
import csv
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

from .utils import remove_diacritics, normalise_arabic, simple_tokenise

# -----------------------------
# Assigned stylometric features
# i = 1, n = 23  -> [1, 24, 47, 70, 93]
# -----------------------------
I_POS = 1
N_STUDENTS = 23
ASSIGNED_1BASED = [I_POS + k * N_STUDENTS for k in range(5)]  # [1,24,47,70,93]

# f1, f24, f47, f70, f93
# NOTE: f24 is "Number of different punctuation signs / C"
ASSIGNED_FEATURES = [
    "f1_total_chars",
    "f24_diff_punct_over_C",
    "f47_pronoun_count",
    "f70_first_person_count",
    "f93_arousal_mean",
]

PUNCT_CHARS = set(list(".,!?:;\"'()[]{}-،؟؛…"))

AR_PRONOUNS = set([
    "انا","نحن","انت","انتي","انتم","انتن","هو","هي","هم","هن",
    "إنا","إننا","اني","انني","إنني","لنا","لي","لك","لكم","لكن"
])

FIRST_PERSON = set(["انا","نحن","اني","انني","إنني","إنا","إننا","لي","لنا","عندي","عندنا"])

def get_ar_stopwords():
    """
    Uses NLTK Arabic stopwords. If missing locally, downloads once.
    """
    try:
        sw = set(stopwords.words("arabic"))
    except LookupError:
        nltk.download("stopwords")
        sw = set(stopwords.words("arabic"))
    except Exception:
        sw = set()

    # small extra list (just helpful, not trying to be perfect)
    sw |= set(["و","في","على","من","الى","إلى","عن","هذا","هذه","ذلك","تلك","هناك","كما","لكن","او","أو"])
    return sw

def preprocess_text(text: str, do_stem=True):
    """
    Arabic-specific pipeline:
    - normalisation
    - remove diacritics
    - stopwords
    - ISRI stemming (optional)
    """
    if not isinstance(text, str):
        return ""

    text = remove_diacritics(text)
    text = normalise_arabic(text)

    toks = simple_tokenise(text)
    sw = get_ar_stopwords()
    toks = [t for t in toks if t not in sw]

    if do_stem:
        st = ISRIStemmer()
        toks = [st.stem(t) for t in toks]

    return " ".join(toks)

def light_for_stylometry(text: str):
    # keep pronouns etc., so: no stemming, but do normalise + remove diacritics
    if not isinstance(text, str):
        return ""
    text = remove_diacritics(text)
    text = normalise_arabic(text)
    return text

def _diff_punct_over_C(text: str):
    """
    f24: Number of different punctuation signs / C
    """
    C = max(len(text), 1)
    used = set([ch for ch in text if ch in PUNCT_CHARS])
    return len(used) / C

def _pronoun_count(tokens):
    return sum(1 for t in tokens if t in AR_PRONOUNS)

def _first_person_count(tokens):
    return sum(1 for t in tokens if t in FIRST_PERSON)

def load_arousal_lexicon(path="data/external/arabic_vad.csv"):
    """
    Optional external file expected as CSV with columns: word, arousal
    If missing, fallback to a tiny built-in dict.
    """
    if os.path.exists(path):
        lex = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                w = row.get("word", "").strip()
                a = row.get("arousal", "").strip()
                if w and a:
                    w = light_for_stylometry(w)
                    try:
                        lex[w] = float(a)
                    except Exception:
                        pass
        if len(lex) > 50:
            return lex

    # fallback (tiny)
    return {
        "خوف": 0.9, "غضب": 0.85, "حماس": 0.75, "سعاده": 0.7, "حزن": 0.65,
        "قلق": 0.8, "ممتاز": 0.6, "سيء": 0.7
    }

AROUSAL_LEX = load_arousal_lexicon()

def _arousal_mean(tokens):
    vals = []
    for t in tokens:
        key = light_for_stylometry(t)
        if key in AROUSAL_LEX:
            vals.append(AROUSAL_LEX[key])
    return float(np.mean(vals)) if vals else 0.0

def extract_assigned_stylometry(text: str):
    """
    Returns ONLY the 5 required features: f1, f24, f47, f70, f93.
    """
    t = light_for_stylometry(text)
    tokens = simple_tokenise(t)

    f1  = float(len(t))                 # f1: total chars (C)
    f24 = float(_diff_punct_over_C(t))  # f24: different punctuation signs / C
    f47 = float(_pronoun_count(tokens)) # f47: number of pronouns
    f70 = float(_first_person_count(tokens)) # f70: 1st person count
    f93 = float(_arousal_mean(tokens))  # f93: emotional arousal mean

    return {
        "f1_total_chars": f1,
        "f24_diff_punct_over_C": f24,
        "f47_pronoun_count": f47,
        "f70_first_person_count": f70,
        "f93_arousal_mean": f93,
    }

def add_stylometry(df: pd.DataFrame, text_col="text"):
    feats = df[text_col].apply(extract_assigned_stylometry).apply(pd.Series)
    out = df.copy()
    for c in feats.columns:
        out[c] = feats[c].astype(float)
    return out

# -----------------------------
# BERT embeddings helper
# -----------------------------
def build_bert_embeddings(
    texts,
    model_name="asafaya/bert-base-arabic",
    batch_size=16,
    max_len=128,
    device=None
):
    """
    Returns numpy array [N, D] using mean pooling.
    This is intentionally simple and not overly optimised.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.to(device)
    mdl.eval()

    all_vecs = []

    def mean_pool(last_hidden, mask):
        mask = mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
            all_vecs.append(pooled.detach().cpu().numpy())

    return np.vstack(all_vecs)
