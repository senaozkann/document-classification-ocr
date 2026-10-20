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
        # 21 Ekim: CV'yi bilimsel makaleden ayıran, CV'ye özgü ifadeler eklendi (university korunuyor)
        ("biographical sketch/cv", r"\b(?:biographical\s+sketch|curriculum\s+vitae|resume)\b"),
        ("position title", r"\bposition\s+title\b"),
        ("professor", r"\b(?:assistant|associate|full|visiting)?\s*professor\b"),
        ("yıl aralığı", r"\b(?:19|20)\d{2}\s*[-–]\s*(?:(?:19|20)?\d{2}\b|present)"),
        ("present", r"\bpresent\b"),
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

# İlk sürüm (20 Ekim): karşılaştırma için saklanır. CV şablonunda sadece ilk 7 ifade vardı.
_NEW_RESUME = {"biographical sketch/cv", "position title", "professor", "yıl aralığı", "present"}
TEMPLATES_V1 = {c: [(n, p) for n, p in pats if not (c == "resume" and n in _NEW_RESUME)]
                for c, pats in TEMPLATES.items()}

_CACHE = {}


def _compiled(templates):
    key = id(templates)
    if key not in _CACHE:
        _CACHE[key] = {c: [(n, re.compile(p, re.I | re.M)) for n, p in pats] for c, pats in templates.items()}
    return _CACHE[key]


_COMPILED = _compiled(TEMPLATES)


def match_template(text, classes=None, templates=None):
    """{sınıf: [eşleşen ifade adları]}"""
    comp = _compiled(templates or TEMPLATES)
    text = text or ""
    return {c: [n for n, rx in comp[c] if rx.search(text)] for c in (classes or comp)}


def fit_weights(texts, labels, classes, templates=None):
    """Her ifadeye veriden ağırlık: (kendi sınıfında geçtiği belge oranı) − (diğer sınıflardaki en yüksek oran).

    Örnek: "education" CV'lerin %59'unda, makalelerin %1'inde geçer → ağırlık ≈ 0,58.
    "university" CV'lerin %93'ünde, makalelerin %36'sında geçer → ağırlık ≈ 0,57'den düşük kalır ve
    tek başına makaleyi CV'ye çeviremez. Negatif çıkanlar küçük bir pozitif değere (0,01) çekilir.
    """
    templates = templates or TEMPLATES
    labels = list(labels)
    matches = [match_template(t, list(templates), templates) for t in texts]
    rate = {}
    for c in templates:
        for n, _ in templates[c]:
            rate[(c, n)] = {k: (sum(1 for m, l in zip(matches, labels) if l == k and n in m[c]) /
                                max(1, sum(1 for l in labels if l == k))) for k in classes}
    weights = {}
    for c in classes:
        weights[c] = {}
        for n, _ in templates[c]:
            r = rate[(c, n)]
            others = max((r[k] for k in classes if k != c), default=0.0)
            weights[c][n] = round(max(0.01, r[c] - others), 4)
    return weights


def template_scores(text, classes=None, weights=None, templates=None):
    """{sınıf: şablon puanı (0-1)}. weights verilmezse her ifade eşit sayılır (eşleşen ifade oranı)."""
    templates = templates or TEMPLATES
    m = match_template(text, classes, templates)
    if weights is None:
        return {c: len(v) / len(templates[c]) for c, v in m.items()}
    return {c: sum(weights[c].get(n, 0) for n in v) / max(1e-9, sum(weights[c].values())) for c, v in m.items()}


def template_vector(text, classes):
    """Model için özellik vektörü: her ifade için 0/1 + her sınıf için eşleşme oranı."""
    m = match_template(text, classes)
    bits = [float(n in m[c]) for c in classes for n, _ in _COMPILED[c]]
    return bits + [len(m[c]) / len(TEMPLATES[c]) for c in classes]


def template_feature_names(classes):
    return [f"{c}:{n}" for c in classes for n, _ in _COMPILED[c]] + [f"{c}:oran" for c in classes]


def rule_predict(text, classes, weights=None, templates=None):
    """Sadece şablona göre tahmin: en yüksek puanlı sınıf (hiç eşleşme yoksa None)."""
    s = template_scores(text, classes, weights, templates)
    best = max(s, key=s.get)
    return best if s[best] > 0 else None
