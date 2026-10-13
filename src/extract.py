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


EXTRACTORS = {"resume": extract_resume, "invoice": extract_invoice, "email": extract_email, "form": extract_form}
TR_NAMES = {"resume": "CV", "invoice": "fatura", "email": "e-posta", "form": "form"}


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
    else:
        kv = list((f.get("alanlar") or {}).items())[:4]
        parts = [f.get("baslik")] + [f"{k} = {v}" for k, v in kv]
    parts = [p for p in parts if p]
    return f"Bu bir {TR_NAMES[label]}" + (": " + " · ".join(parts) if parts else " (içerik alanları okunamadı)")


def extract(text, label):
    fields = EXTRACTORS[label](text or "")
    return {"alanlar": fields, "ozet": summarize(label, fields)}
