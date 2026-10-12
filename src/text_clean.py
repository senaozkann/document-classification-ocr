import re
import unicodedata

# OCR'ın sık ürettiği tipografik karakterleri sadeleştir
_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "•": "-", "·": "-",
    " ": " ", "­": "",
})
_WORDISH = re.compile(r"[A-Za-z0-9]")
_NOISE_TOKEN = re.compile(r"^[^\w$%@&#+]+$")          # sadece sembol: | ~ ` ' . , _ vb.
_EDGE_JUNK_L = re.compile(r"^[|_~`'\"\s]+")
_EDGE_JUNK_R = re.compile(r"[|_~`\s]+$")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
_MONEY = re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d+)?")
_NUM = re.compile(r"\d+")


def alnum_ratio(s):
    s = s.replace(" ", "")
    return sum(c.isalnum() for c in s) / len(s) if s else 0.0


def clean_text(raw, min_line_alnum=0.4):
    """OCR metnini temizler; büyük/küçük harf, e-posta, tutar gibi bilgiler korunur
    (bilgi çıkarımı aşamasında gerekecek)."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    t = unicodedata.normalize("NFKC", raw).translate(_CHAR_MAP)
    t = "".join(c for c in t if c == "\n" or c.isprintable())
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)               # satır sonu tirelemesini birleştir
    lines = []
    for line in t.split("\n"):
        tokens = [tok for tok in line.split() if not _NOISE_TOKEN.match(tok)]
        line = " ".join(tokens)
        line = _EDGE_JUNK_R.sub("", _EDGE_JUNK_L.sub("", line))
        if len(line) < 2 or not _WORDISH.search(line):
            continue
        if alnum_ratio(line) < min_line_alnum:              # çoğu sembol olan çöp satır
            continue
        lines.append(line)
    return "\n".join(lines)


def normalize_for_model(clean):
    """Sınıflandırıcı için: küçük harf + e-posta/URL/tutar/sayıları ortak simgeye çevir.
    Böylece model 'john@acme.com' yerine 'bir e-posta adresi var' bilgisini öğrenir."""
    t = clean.lower()
    t = _EMAIL.sub(" emailaddr ", t)
    t = _URL.sub(" urladdr ", t)
    t = _MONEY.sub(" moneyamt ", t)
    t = _NUM.sub("0", t)
    t = re.sub(r"[^a-z0\s]", " ", t)
    return " ".join(t.split())


def garbage_ratio(raw):
    """Ham OCR çıktısındaki anlamsız token oranı (harf/rakam içermeyen ya da
    harf-rakam-sembol karışık kısa parçalar)."""
    toks = raw.split() if isinstance(raw, str) else []
    if not toks:
        return 1.0
    bad = 0
    for tok in toks:
        if not _WORDISH.search(tok):
            bad += 1
        elif len(tok) <= 3 and alnum_ratio(tok) < 0.67:
            bad += 1
    return bad / len(toks)
