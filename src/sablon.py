"""Belge türü şablonları (danışman isteği, 15 Ekim): her türün hemen her belgesinde bulunan ortak ifadeler.

OCR metninde bir türün şablon ifadeleri bulunuyorsa bu, makine öğrenmesi modelinin kararını destekleyen
ikinci bir kanıttır (ör. faturada "invoice", "total", para tutarı; mektupta "Dear", "Sincerely").
Veri seti İngilizce olduğu için ifadeler İngilizcedir; Türkçe karşılıkları yorumda verilmiştir.
"""
import re

# her ifade: (kısa ad, düzenli ifade) · metin küçük harfe çevrilmiş temiz OCR metnine uygulanır
TEMPLATES = {
    "email": [  # e-posta başlık alanları
        ("from:", r"^\s*from\s*[:;]"),
        ("to:", r"^\s*to\s*[:;]"),
        ("subject:", r"^\s*(?:subject|subj)\s*[:;]"),
        ("sent/date:", r"^\s*(?:sent|date)\s*[:;]"),
        ("cc:", r"^\s*b?cc\s*[:;]"),
        ("@ adresi", r"[\w.+-]+@[\w-]+"),
        ("message", r"\bmessage\b"),
    ],
    "invoice": [  # fatura: numara, tutar, miktar, ödeme
        ("invoice", r"\binvoice\b"),
        ("total", r"\btotal\b"),
        ("amount", r"\bamount\b"),
        ("para tutarı", r"[$€£]\s?\d|\b\d{1,3}(?:,\d{3})*\.\d{2}\b"),
        ("qty/quantity", r"\b(?:qty|quantity)\b"),
        ("price", r"\bprice\b"),
        ("bill/ship/sold to", r"\b(?:bill|ship|sold|remit)\s*to\b"),
        ("payment/due", r"\b(?:payment|payable|due|terms)\b"),
    ],
    "resume": [  # CV bölüm başlıkları
        ("education", r"\beducation\b"),
        ("experience", r"\b(?:experience|employment)\b"),
        ("university/college", r"\b(?:university|college|institute)\b"),
        ("degree", r"\b(?:b\.?s\.?|m\.?s\.?|ph\.?d\.?|bachelor|master|doctor|degree)\b"),
        ("born/birth", r"\b(?:born|birth|married)\b"),
        ("research", r"\bresearch\b"),
        ("publications/references", r"\b(?:publications|references|honors|awards)\b"),
    ],
    "form": [  # form: faks kapağı, doldurulacak alanlar
        ("fax/facsimile", r"\b(?:fax|facsimile|telecopy|telecopier)\b"),
        ("transmission", r"\btransmi(?:ssion|tted|t)\b"),
        ("pages", r"\bpages?\b"),
        ("name:", r"\bname\s*[:;]"),
        ("date:", r"\bdate\s*[:;]"),
        ("signature/approved", r"\b(?:signature|signed|approv(?:ed|al))\b"),
        ("request/number", r"\b(?:request|number|no\.)\b"),
    ],
    "letter": [  # mektup / resmi yazı (dilekçe): hitap, kapanış, ek
        ("dear", r"^\s*dear\b"),
        ("sincerely/regards", r"\b(?:sincerely|yours\s+truly|very\s+truly|regards|respectfully|cordially)\b"),
        ("re:", r"^\s*re\s*[:;]"),
        ("enclosed/enclosure", r"\benclos(?:ed|ure)\b"),
        ("thank you", r"\bthank\s+you\b"),
        ("letter", r"\bletter\b"),
        ("please", r"\bplease\b"),
    ],
    "news_article": [  # gazete haberi
        ("said", r"\bsaid\b"),
        ("ajans (AP/UPI/Reuters)", r"\((?:ap|upi)\)|\breuters\b|\bassociated press\b"),
        ("percent", r"\bpercent\b"),
        ("according to", r"\baccording to\b"),
        ("spokesman/officials", r"\b(?:spokesman|spokeswoman|officials?)\b"),
        ("yesterday/today", r"\b(?:yesterday|today|last week)\b"),
        ("times/post/journal", r"\b(?:times|post|journal|tribune|herald|news)\b"),
    ],
    "scientific_publication": [  # bilimsel makale bölümleri
        ("abstract", r"\babstract\b"),
        ("introduction", r"\bintroduction\b"),
        ("methods/results", r"\b(?:methods?|results)\b"),
        ("references", r"\breferences\b"),
        ("et al.", r"\bet\s+al\b"),
        ("fig./table", r"\b(?:fig|figure|table)\b"),
        ("journal/vol.", r"\b(?:journal|vol|volume)\b"),
    ],
}

_COMPILED = {c: [(n, re.compile(p, re.I | re.M)) for n, p in pats] for c, pats in TEMPLATES.items()}


def match_template(text, classes=None):
    """{sınıf: [eşleşen ifade adları]}"""
    text = text or ""
    return {c: [n for n, rx in _COMPILED[c] if rx.search(text)] for c in (classes or TEMPLATES)}


def template_scores(text, classes=None):
    """{sınıf: eşleşen ifade oranı (0-1)}"""
    m = match_template(text, classes)
    return {c: len(v) / len(TEMPLATES[c]) for c, v in m.items()}


def template_vector(text, classes):
    """Model için özellik vektörü: her ifade için 0/1 + her sınıf için eşleşme oranı."""
    m = match_template(text, classes)
    bits = [float(n in m[c]) for c in classes for n, _ in _COMPILED[c]]
    return bits + [len(m[c]) / len(TEMPLATES[c]) for c in classes]


def template_feature_names(classes):
    return [f"{c}:{n}" for c in classes for n, _ in _COMPILED[c]] + [f"{c}:oran" for c in classes]


def rule_predict(text, classes):
    """Sadece şablona göre tahmin: en yüksek eşleşme oranına sahip sınıf (eşleşme yoksa None)."""
    s = template_scores(text, classes)
    best = max(s, key=s.get)
    return best if s[best] > 0 else None
