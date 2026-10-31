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


# ---------------------------------------------------------------- iki aşamalı sözlük (22 Ekim danışman isteği)
# Genel İngilizce sözlük bilimsel terimleri tanımadığı için onları "düzeltiyordu" (DNA → DAN, tumor → tutor).
# Çözüm: ikinci bir ALAN SÖZLÜĞÜ. Train belgelerinde sık geçen ama genel sözlükte olmayan, tek bir belge
# türünde yoğunlaşan kelimeler (DNA, PhD, NIH, tumor ...) alan sözlüğüne alınır.
#   1. aşama: kelime alan sözlüğündeyse dokunulmaz (korunur)
#   2. aşama: alan kelimesine 1 harf uzaklıktaysa ona düzeltilir (ör. "DNAs" değil, "tumar" → "tumor")
#   3. kalanlar genel sözlükle düzeltilir (eski yöntem)
# Yaygın OCR hataları ("wih", "fom", "folowing") ve birleşik kelimeler ("ofthe") alan sözlüğüne alınmaz (_looks_like_ocr_error).

def build_domain_dictionary(texts, labels, min_docs=5, min_share=0.6):
    """Alan sözlüğü: {kelime: belge sayısı}. Sadece train metinleri verilmelidir.

    Şartlar: (1) genel sözlükte yok, (2) en az `min_docs` belgede geçiyor,
    (3) geçtiği belgelerin en az `min_share` kadarı tek bir türde (alan terimi; OCR hatası değil).
    """
    sym = _load()
    per_word = {}
    for t, lab in zip(texts, labels):
        for w in {w.lower() for w in re.findall(r"[A-Za-z]+", t or "") if len(w) >= 3}:
            if w not in _WORDS:
                per_word.setdefault(w, Counter())[lab] += 1
    out = {}
    for w, c in per_word.items():
        n = sum(c.values())
        if n >= min_docs and max(c.values()) / n >= min_share and not _looks_like_ocr_error(w, sym):
            out[w] = n
    return out


def _looks_like_ocr_error(w, sym, common=5e7):
    """Alan terimi değil, sık bir kelimenin OCR hatası mı?
    (1) Genel sözlükte ÇOK SIK geçen bir kelimeye 1 harf uzaklıkta: wih → with, folowing → following.
        (DNA → dan, NIH → nil gibi yakın kelimeler o kadar sık değildir; DNA alan terimi olarak kalır.)
    (2) İki sık kelimenin birleşmesi: ofthe = of + the, inthe = in + the (biochem = bio + chem değil).
    """
    sug = sym.lookup(w, Verbosity.TOP, max_edit_distance=1)
    if sug and sug[0].distance == 1 and sug[0].count >= common:
        return True
    return any(_WORDS.get(w[:i], 0) >= common and _WORDS.get(w[i:], 0) >= common for i in range(2, len(w) - 1))


class TwoStageFixer:
    """Genel sözlük + alan sözlüğü ile düzeltme. `correct_text` ile aynı arayüz."""

    def __init__(self, domain):
        _load()
        self.domain = dict(domain)
        self.dsym = SymSpell(max_dictionary_edit_distance=1, prefix_length=7)
        for w, c in self.domain.items():
            self.dsym.create_dictionary_entry(w, c)

    def correct_word(self, word, line_start=False):
        low = word.lower()
        if low in self.domain:
            return word, None                                    # alan kelimesi korunur
        if len(low) >= 5 and low.isalpha() and low not in _WORDS:
            sug = self.dsym.lookup(low, Verbosity.TOP, max_edit_distance=1)
            if sug and sug[0].term != low:
                return _match_case(word, sug[0].term), "alan sözlüğü"
        return correct_word(word, line_start)

    def correct_text(self, text, log=None):
        if not text:
            return text
        out_lines = []
        for line in text.split("\n"):
            pieces, last, first = [], 0, True
            for m in _TOKEN.finditer(line):
                w = m.group(0)
                new, rule = self.correct_word(w, line_start=first and m.start() <= 2)
                first = False
                pieces.append(line[last:m.start()])
                pieces.append(new)
                last = m.end()
                if rule and log is not None:
                    log[(w, new, rule)] += 1
            pieces.append(line[last:])
            out_lines.append("".join(pieces))
        return "\n".join(out_lines)

    def valid_ratio(self, text):
        """Genel ya da alan sözlüğünde bulunan kelime oranı."""
        toks = [t.lower() for t in re.findall(r"[A-Za-z]+", text or "") if len(t) >= 3]
        return sum(t in _WORDS or t in self.domain for t in toks) / len(toks) if toks else 0.0


def dictionary_info():
    """Genel sözlüğün kaynağı ve boyutu (sunum için)."""
    _load()
    return {"dosya": "frequency_dictionary_en_82_765.txt (symspellpy paketi)",
            "kelime sayısı": len(_WORDS),
            "kaynak": "SymSpell; SCOWL kelime listesi ile Google Books Ngram sıklıklarının kesişimi"}
