import re

ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")

def remove_diacritics(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return ARABIC_DIACRITICS_RE.sub("", text)

def normalise_arabic(text: str) -> str:
    """Very standard, intentionally simple normalisation."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # unify alifs
    text = re.sub("[إأآا]", "ا", text)
    # hamza on waw/yaa
    text = re.sub("[ؤ]", "و", text)
    text = re.sub("[ئ]", "ي", text)
    # taa marbuta / alif maqsura
    text = re.sub("ة", "ه", text)
    text = re.sub("ى", "ي", text)
    # remove tatweel
    text = re.sub("ـ+", "", text)
    # keep Arabic letters + spaces + basic punctuation
    text = re.sub(r"[^\u0600-\u06FF\s\.\,\!\?\:\;\،\؟\؛\"\'\(\)\[\]\{\}\-]", " ", text)
    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def simple_tokenise(text: str):
    # split on spaces, keep it simple
    return [t for t in text.split() if t]
