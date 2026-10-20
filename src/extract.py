"""Belge türüne özel içerik çıkarımı (kural tabanlı + spaCy NER).

Girdi: OCR'dan gelen temizlenmiş metin (text_clean.clean_text çıktısı) ve tahmin edilen tür.
Çıktı: {"alanlar": {...}, "ozet": "Bu bir CV: ..."}
"""
import re

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:  # spaCy yoksa sadece kurallar çalışır
    _NLP = None

# ---------------------------------------------------------------- ortak kalıplar
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3}[\s.-]?\d{2,4}(?:[\s.-]?\d{2})?")
URL = re.compile(r"\b(?:https?://|www\.)\S+|\b(?:github|linkedin)\.com/\S+", re.I)
MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE = re.compile(
    rf"\b(?:\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}|{MONTHS}\s+\d{{1,2}},?\s+\d{{2,4}}|\d{{1,2}}\s+{MONTHS}\s+\d{{2,4}}|\d{{4}}-\d{{2}}-\d{{2}})\b",
    re.I)
MONEY = re.compile(r"(?:[$€£₺]\s?\d[\d,]*(?:\.\d{2})?|\d[\d,]*\.\d{2}\s?(?:usd|eur|tl|try)?)", re.I)

DEGREE_WORDS = ("bachelor", "master", "phd", "ph.d", "b.s", "b.sc", "m.s", "m.sc", "b.a", "m.a", "mba",
                "degree", "engineering", "science", "lisans", "mühendisliği", "muhendisligi")
SCHOOL_WORDS = ("university", "universit", "college", "institute", "school", "academy", "üniversite", "universite")
SECTION_HEADS = ("education", "experience", "work experience", "employment", "skills", "technical skills",
                 "projects", "references", "publications", "certifications", "languages", "interests",
                 "objective", "summary", "awards", "activities")


def _lines(text):
    return [l.strip() for l in text.splitlines() if l.strip()]


def _ents(text, labels):
    if _NLP is None or not text:
        return []
    doc = _NLP(text[:20000])
    out = []
    for e in doc.ents:
        t = " ".join(e.text.split())
        if e.label_ in labels and len(t) > 1 and t not in out:
            out.append(t)
    return out


def _first(pattern, text, group=0):
    m = pattern.search(text)
    return m.group(group).strip() if m else None


def _section(lines, names):
    """Başlığı `names` içinde olan bölümün satırlarını döndürür (bir sonraki başlığa kadar)."""
    out, inside = [], False
    for l in lines:
        key = re.sub(r"[^a-z ]", "", l.lower()).strip()
        is_head = key in SECTION_HEADS and len(l) < 40
        if is_head:
            if inside:
                break
            inside = key in names
            continue
        if inside:
            out.append(l)
    return out


_NOT_NAME = SECTION_HEADS + SCHOOL_WORDS + DEGREE_WORDS + ("resume", "curriculum", "vitae", "cv", "page",
                                                           "street", "avenue", "road", "phone", "email", "address")


def _looks_like_name(line):
    """2-4 kelimelik, harflerden oluşan, büyük harfle başlayan, başlık/okul/adres olmayan satır."""
    if any(ch.isdigit() for ch in line) or any(ch in line for ch in "@:/|()"):
        return False
    words = line.replace(",", " ").replace(".", " ").split()
    if not 2 <= len(words) <= 4 or not all(w.isalpha() for w in words) or sum(len(w) >= 2 for w in words) < 2:
        return False
    low = line.lower()
    if any(k in low.split() or k in low for k in _NOT_NAME if len(k) > 3) or low in _NOT_NAME:
        return False
    return all(w[0].isupper() for w in words)


# ---------------------------------------------------------------- türe özel çıkarıcılar
def extract_resume(text):
    lines = _lines(text)
    persons = [p for p in _ents("\n".join(lines[:8]), {"PERSON"}) if _looks_like_name(p)]
    name = persons[0] if persons else next((l for l in lines[:6] if _looks_like_name(l)), None)

    edu = _section(lines, {"education"}) or lines
    school = next((l for l in edu if any(w in l.lower() for w in SCHOOL_WORDS)), None)
    if school is None:
        orgs = [o for o in _ents("\n".join(edu), {"ORG"}) if any(w in o.lower() for w in SCHOOL_WORDS)]
        school = orgs[0] if orgs else None
    degree = next((l for l in edu if any(w in l.lower() for w in DEGREE_WORDS) and l != school), None)
    if school:
        school = re.sub(r"^\s*education\s*[:;-]\s*", "", school, flags=re.I) or None

    skills_lines = _section(lines, {"skills", "technical skills"})
    skills = None
    if skills_lines:
        items = re.split(r"[,;•·|]|\s-\s", " ".join(skills_lines[:4]))
        skills = [s.strip(" :.") for s in items if 1 < len(s.strip()) < 30][:10] or None

    return {
        "ad_soyad": name,
        "universite": school,
        "bolum_derece": degree,
        "e_posta": _first(EMAIL, text),
        "telefon": _first(PHONE, text),
        "beceriler": skills,
    }


_NOT_COMPANY = ("invoice", "quantity", "description", "price", "amount", "total", "date", "bill to",
                "ship to", "page", "tax", "qty", "item")
_CORP = re.compile(r"\b(?:inc|corp|corporation|co|company|ltd|llc|plc|group|gmbh|a\.?ş|ltd\.?\s*şti)\b\.?", re.I)


def _invoice_company(lines):
    top = [l for l in lines[:12] if not any(w in l.lower() for w in _NOT_COMPANY) and re.search(r"[A-Za-z]{3}", l)]
    corp = next((l for l in top if _CORP.search(l)), None)
    if corp:
        return corp
    orgs = [o for o in _ents("\n".join(top), {"ORG"}) if not any(w in o.lower() for w in _NOT_COMPANY)]
    if orgs:
        return orgs[0]
    # antetli kâğıt: ilk 3 satırdan büyük harf ağırlıklı olan
    def upper_ratio(l):
        letters = [c for c in l if c.isalpha()]
        return sum(c.isupper() for c in letters) / len(letters) if letters else 0
    return next((l for l in top[:3] if upper_ratio(l) > 0.6 and len(l) > 4), None)


def extract_invoice(text):
    text = re.sub(r"(\d)[,.]\s+(\d{3})\b", r"\1,\2", text)   # OCR: '1, 250.00' -> '1,250.00'
    lines = _lines(text)
    inv_no = _first(re.compile(r"invoice\s*(?:no\.?|number|num|#)\s*[:.]?\s*((?=[A-Z0-9/-]*\d)[A-Z0-9][A-Z0-9/-]{1,20})", re.I), text, 1)
    total = None
    for pat in (r"(?:total\s+amount\s+due|amount\s+due|balance\s+due|grand\s+total|total\s+due|invoice\s+total|total)\b[^\n$\d]{0,20}([$€£]?\s?\d[\d,]*(?:\.\d{2})?)",
                r"\bamount\b[^\n$\d]{0,15}([$€£]?\s?\d[\d,]*(?:\.\d{2})?)"):
        m = re.findall(pat, text, re.I)
        m = [x.strip().rstrip(",.") for x in m if re.search(r"\d\.\d{2}$|\d{3,}", x.strip().rstrip(",."))]
        if m:
            total = m[-1]
            break
    company = _invoice_company(lines)
    return {
        "fatura_no": inv_no,
        "tarih": _first(DATE, text),
        "firma": company,
        "toplam_tutar": total,
        "tutarlar": MONEY.findall(text)[:5] or None,
    }


def _header(text, names):
    pat = re.compile(rf"^\s*(?:{names})\s*[:;]\s*(.+)$", re.I | re.M)
    m = pat.search(text)
    return m.group(1).strip() if m else None


def extract_email(text):
    lines = _lines(text)
    sender = _header(text, "from|sender")
    to = _header(text, "to|recipient")
    subject = _header(text, "subject|re|subj")
    date = _header(text, "date|sent") or _first(DATE, text)
    head_pat = re.compile(r"^\s*(?:from|to|cc|bcc|subject|subj|re|date|sent|sender|recipient)\s*[:;]", re.I)
    body = [l for l in lines if not head_pat.match(l)]
    return {
        "gonderen": sender,
        "alici": to,
        "konu": subject,
        "tarih": date,
        "e_postalar": EMAIL.findall(text)[:5] or None,
        "govde_ozeti": " ".join(body[:3])[:200] or None,
    }


def extract_form(text):
    lines = _lines(text)
    title = next((l for l in lines[:8] if l.isupper() and len(l.split()) >= 2), lines[0] if lines else None)
    pairs = {}
    for l in lines:
        m = re.match(r"^([A-Za-z][A-Za-z ./&()'-]{1,40}?)\s*[:;]\s*(.+)$", l)
        if m:
            key, val = m.group(1).strip(), m.group(2).strip(" _.")
            if val and key.lower() not in pairs and len(val) < 120:
                pairs[key.lower()] = val
        if len(pairs) >= 10:
            break
    return {
        "baslik": title,
        "alanlar": pairs or None,
        "tarih": _first(DATE, text),
    }


_CLOSING = re.compile(r"^\s*(?:sincerely|yours\s+(?:truly|sincerely|faithfully|very\s+truly)|very\s+truly\s+yours|"
                      r"respectfully(?:\s+yours)?|best\s+regards|kind\s+regards|regards|cordially)\b", re.I)
_SALUTATION = re.compile(r"^\s*dear\s+(.{2,60}?)\s*[:,;]?\s*$", re.I | re.M)


def extract_letter(text):
    """Mektup / resmi yazı (dilekçe): kurum (antet), tarih, alıcı, hitap, konu, imza sahibi."""
    lines = _lines(text)
    salute = _first(_SALUTATION, text, 1)
    sal_idx = next((i for i, l in enumerate(lines) if _SALUTATION.match(l)), None)
    # imza: kapanış ifadesinden sonraki ilk isim gibi görünen satır
    signer = None
    close_idx = next((i for i, l in enumerate(lines) if _CLOSING.match(l)), None)
    if close_idx is not None:
        signer = next((l for l in lines[close_idx + 1:close_idx + 6] if _looks_like_name(l)), None)
        if signer is None:
            persons = _ents("\n".join(lines[close_idx + 1:close_idx + 6]), {"PERSON"})
            signer = persons[0] if persons else None
    # alıcı adresi: tarih ile "Dear" arasındaki blokta isim gibi görünen ilk satır
    head = [l for l in lines[:4] if not DATE.search(l)]
    org = next((l for l in head if _CORP.search(l)), None)
    if org is None:
        orgs = _ents("\n".join(head), {"ORG"})
        org = orgs[0] if orgs else None
    recipient = None
    if sal_idx:
        date_idx = next((i for i, l in enumerate(lines[:sal_idx]) if DATE.search(l)), -1)
        window = lines[max(date_idx + 1, sal_idx - 8):sal_idx]
        recipient = next((l for l in window if _looks_like_name(l) and l != org), None)
    return {
        "kurum": org,
        "tarih": _first(DATE, text),
        "alici": recipient,
        "hitap": salute,
        "konu": _header(text, "re|subject|subj"),
        "imza": signer,
    }


_DATELINE = re.compile(r"^([A-Z][A-Z .,'-]{2,40})\s*(?:\((?:AP|UPI|Reuters|AFP)\))?\s*[-—–]{1,2}\s", re.M)


def extract_news(text):
    """Gazete haberi: başlık, yer (dateline), tarih, haberde geçen kişi ve kurumlar."""
    lines = _lines(text)
    title = next((l for l in lines[:6] if 3 <= len(l.split()) <= 15 and sum(c.isalpha() for c in l) > 10), None)
    return {
        "baslik": title,
        "yer": _first(_DATELINE, text, 1),
        "tarih": _first(DATE, text),
        "kisiler": _ents(text, {"PERSON"})[:4] or None,
        "kurumlar": _ents(text, {"ORG"})[:4] or None,
    }


def extract_scientific(text):
    """Bilimsel makale: başlık, yazarlar, özet (Abstract), anahtar kelimeler."""
    lines = _lines(text)
    title = next((l for l in lines[:6] if len(l.split()) >= 4 and sum(c.isalpha() for c in l) > 15), None)
    authors = [p for p in _ents("\n".join(lines[:12]), {"PERSON"})][:5] or None
    m = re.search(r"\babstract\b[\s:.-]*(.{20,600}?)(?:\n\s*\n|\b(?:introduction|key\s*words?|keywords)\b|$)",
                  text, re.I | re.S)
    abstract = " ".join(m.group(1).split())[:300] if m else None
    kw = _first(re.compile(r"\bkey\s*words?\s*[:.-]\s*(.+)", re.I), text, 1)
    return {
        "baslik": title,
        "yazarlar": authors,
        "ozet": abstract,
        "anahtar_kelimeler": kw,
        "kaynakca_var": bool(re.search(r"\b(?:references|bibliography|literature cited)\b", text, re.I)),
    }


EXTRACTORS = {"resume": extract_resume, "invoice": extract_invoice, "email": extract_email, "form": extract_form,
              "letter": extract_letter, "news_article": extract_news, "scientific_publication": extract_scientific}
TR_NAMES = {"resume": "CV", "invoice": "fatura", "email": "e-posta", "form": "form",
            "letter": "resmi yazı / mektup", "news_article": "gazete haberi", "scientific_publication": "bilimsel makale"}


def summarize(label, fields):
    f = fields
    if label == "resume":
        parts = [f.get("ad_soyad"), f.get("universite"), f.get("bolum_derece")]
        if f.get("beceriler"):
            parts.append("Beceriler: " + ", ".join(f["beceriler"][:5]))
    elif label == "invoice":
        parts = [f"Fatura no {f['fatura_no']}" if f.get("fatura_no") else None, f.get("firma"), f.get("tarih"),
                 f"Toplam {f['toplam_tutar']}" if f.get("toplam_tutar") else None]
    elif label == "email":
        who = " → ".join(x for x in (f.get("gonderen"), f.get("alici")) if x) or None
        parts = [who, f"Konu: {f['konu']}" if f.get("konu") else None, f.get("tarih")]
    elif label == "letter":
        parts = [f.get("kurum"), f.get("tarih"), f"Alıcı: {f.get('hitap') or f.get('alici')}" if (f.get("hitap") or f.get("alici")) else None,
                 f"Konu: {f['konu']}" if f.get("konu") else None, f"İmza: {f['imza']}" if f.get("imza") else None]
    elif label == "news_article":
        parts = [f.get("baslik"), f.get("yer"), f.get("tarih"),
                 "Kişiler: " + ", ".join(f["kisiler"][:3]) if f.get("kisiler") else None]
    elif label == "scientific_publication":
        parts = [f.get("baslik"), "Yazarlar: " + ", ".join(f["yazarlar"][:3]) if f.get("yazarlar") else None,
                 f"Anahtar kelimeler: {f['anahtar_kelimeler']}" if f.get("anahtar_kelimeler") else None]
    else:
        kv = list((f.get("alanlar") or {}).items())[:4]
        parts = [f.get("baslik")] + [f"{k} = {v}" for k, v in kv]
    parts = [p for p in parts if p]
    return f"Bu bir {TR_NAMES.get(label, label)}" + (": " + " · ".join(parts) if parts else " (içerik alanları okunamadı)")


def extract(text, label):
    fields = EXTRACTORS.get(label, extract_form)(text or "")
    return {"alanlar": fields, "ozet": summarize(label, fields)}


# ---------------------------------------------------------------- konu özeti (danışmanın "ikinci aşama" hedefi)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

_GENERIC = {"new", "please", "thank", "thanks", "dear", "regards", "sincerely", "page", "shows", "also",
            "would", "could", "like", "just", "know", "let", "use", "used", "make", "need", "want", "good",
            "subject", "cc", "bcc", "sent", "fax", "tel", "phone", "date", "re", "fwd", "attached"}
_STOP = sorted(ENGLISH_STOP_WORDS | _GENERIC)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


class TopicSummarizer:
    """Belgenin ne hakkında olduğunu anlatan anahtar kelimeler + 2-3 cümlelik çıkarımsal özet.

    Kelime ağırlıkları (TF-IDF) eğitim belgelerinden öğrenilir: bir belgede sık geçen ama
    tüm belgelerde yaygın olmayan kelimeler o belgenin konusunu temsil eder. OCR çöpü genelde
    tek bir belgede geçtiği için min_df=2 ile sözlüğe girmez.
    """

    def __init__(self, min_df=2, max_df=0.5):
        self.vec = TfidfVectorizer(stop_words=_STOP, ngram_range=(1, 2), min_df=min_df, max_df=max_df,
                                   sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b")

    def fit(self, texts):
        self.vec.fit([t for t in texts if isinstance(t, str)])
        self.vocab_ = self.vec.vocabulary_
        self.names_ = self.vec.get_feature_names_out()
        return self

    def keywords(self, text, k=6):
        if not text:
            return []
        row = self.vec.transform([text]).toarray().ravel()
        order = np.argsort(row)[::-1]
        out = []
        for i in order:
            if row[i] <= 0 or len(out) >= k:
                break
            term = self.names_[i]
            if any(term in o or o in term for o in out):   # "invoice" ile "invoice number" tekrarını önle
                continue
            out.append(term)
        return out

    def _sentences(self, text):
        sents = []
        for s in _SENT_SPLIT.split(text):
            s = " ".join(s.split())
            words = s.split()
            alpha = sum(w.isalpha() for w in words)
            if len(words) >= 5 and alpha / len(words) >= 0.6 and len(s) <= 300:
                sents.append(s)
        return sents

    def summary(self, text, n=2):
        sents = self._sentences(text or "")
        if not sents:
            return []
        m = self.vec.transform(sents)
        scores = np.asarray(m.sum(axis=1)).ravel() / np.sqrt(np.maximum(m.getnnz(axis=1), 1))
        top = sorted(np.argsort(scores)[::-1][:n])
        return [sents[i] for i in top if scores[i] > 0]

    def describe(self, text, label, k=6, n=2):
        kws = self.keywords(text, k)
        sents = self.summary(text, n)
        tur = TR_NAMES.get(label, label)
        line = f"Bu bir {tur}."
        if kws:
            line += f" Konusu / anahtar kelimeler: {', '.join(kws)}."
        return {"anahtar_kelimeler": kws, "ozet_cumleler": sents, "konu_ozeti": line}
