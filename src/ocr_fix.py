"""Sözlük tabanlı OCR hata düzeltme (danışman isteği, 15 Ekim: "Dater" → "Date" gibi hatalar).

Üç adım, sırayla:
1. OCR karışıklık kuralları: Tesseract'ın sık karıştırdığı harf grupları (rn↔m, 1↔l, 0↔o, vv↔w, cl↔d ...).
   Kelime sözlükte yoksa ve bir kural uygulanınca sözlükte bir kelimeye dönüşüyorsa o kelime alınır.
2. Alan adı kuralı: satır başındaki kelime bilinen bir alan adına (Date, From, To, Total ...) 1 harf
   uzaklıktaysa düzeltilir. "Dater" sözlükte geçerli bir kelime olduğu için sadece bu kural yakalayabilir.
3. SymSpell: sözlükte olmayan kelime, sıklık sözlüğündeki en yakın kelimeye çevrilir
   (5 harfe kadar en fazla 1, daha uzunlarda en fazla 2 harf farkı).

Rakam, sembol içeren kelimeler ve 3 harften kısa kelimeler değiştirilmez.
Kişi/firma adları da sözlükte olmadığı için yanlış "düzeltilebilir"; bu yüzden düzeltilmiş metin
sadece sınıflandırma için kullanılır, içerik çıkarımı (ad, firma) orijinal metinden yapılır.
"""
import re
from collections import Counter
from importlib import resources

from symspellpy import SymSpell, Verbosity

_SYM = None
_WORDS = None
OCR_CONFUSIONS = [("rn", "m"), ("m", "rn"), ("1", "l"), ("l", "1"), ("0", "o"), ("vv", "w"), ("cl", "d"),
                  ("ii", "u"), ("li", "h"), ("5", "s"), ("8", "b"), ("|", "l"), ("l", "i"), ("i", "l")]
FIELD_NAMES = ["date", "from", "to", "subject", "name", "address", "total", "amount", "invoice", "phone",
               "fax", "pages", "re", "cc", "title", "company", "signature"]
_TOKEN = re.compile(r"[A-Za-z0-9|]+")


def _load():
    global _SYM, _WORDS
    if _SYM is None:
        _SYM = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        path = resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
        _SYM.load_dictionary(str(path), term_index=0, count_index=1)
        _WORDS = _SYM.words
    return _SYM


def in_dictionary(word):
    _load()
    return word.lower() in _WORDS


def _match_case(src, dst):
    if src.isupper():
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


def _confusion_fix(low):
    for a, b in OCR_CONFUSIONS:
        if a in low:
            cand = low.replace(a, b)
            if cand in _WORDS:
                return cand
    return None


def correct_word(word, line_start=False):
    """(düzeltilmiş kelime, kural adı) — değişiklik yoksa (word, None)."""
    sym = _load()
    low = word.lower()
    if line_start:
        for f in FIELD_NAMES:
            if len(f) >= 3 and low != f and len(low) == len(f) + 1 and low.startswith(f):
                return _match_case(word, f), "alan adı"
    if len(low) < 3 or low in _WORDS:
        return word, None
    fixed = _confusion_fix(low)
    if fixed:
        return _match_case(word, fixed), "OCR karışıklığı"
    if not low.isalpha():
        return word, None
    max_d = 1 if len(low) <= 5 else 2
    sug = sym.lookup(low, Verbosity.TOP, max_edit_distance=max_d)
    if sug and sug[0].term != low:
        return _match_case(word, sug[0].term), "SymSpell"
    return word, None


def correct_text(text, log=None):
    """Metindeki kelimeleri düzeltir. `log` bir Counter ise (orijinal, düzeltme, kural) sayılır."""
    if not text:
        return text
    out_lines = []
    for line in text.split("\n"):
        pieces, last, first = [], 0, True
        for m in _TOKEN.finditer(line):
            w = m.group(0)
            new, rule = correct_word(w, line_start=first and m.start() <= 2)
            first = False
            pieces.append(line[last:m.start()])
            pieces.append(new)
            last = m.end()
            if rule and log is not None:
                log[(w, new, rule)] += 1
        pieces.append(line[last:])
        out_lines.append("".join(pieces))
    return "\n".join(out_lines)


def dict_ratio(text):
    """Harflerden oluşan (≥3) kelimelerin sözlükte bulunma oranı."""
    _load()
    toks = [t.lower() for t in re.findall(r"[A-Za-z]+", text or "") if len(t) >= 3]
    return sum(t in _WORDS for t in toks) / len(toks) if toks else 0.0
