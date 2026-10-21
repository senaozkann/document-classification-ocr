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
# v2 (21 Ekim, elle değerlendirme): belge numarası (50145702) ve yayın sayfa aralığı (30502-30509) telefon sanılıyordu.
# Artık ya "tel/phone" etiketi ya da telefon biçimi (alan kodu + 3 + 4 hane, ayraçlı) gerekiyor.
PHONE_LABELED = re.compile(r"\b(?:tel|telephone|phone|mobile|cell|gsm)\b\.?\s*(?:no\.?|number)?\s*[:#.]?\s*(\+?[\d(][\d\s().-]{6,18}\d)", re.I)
PHONE_FORMATTED = re.compile(r"(?<![\d-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\d-])")


def find_phone(text):
    m = PHONE_LABELED.search(text)
    if m and sum(c.isdigit() for c in m.group(1)) >= 7:
        return " ".join(m.group(1).split())
    m = PHONE_FORMATTED.search(text)
    return m.group(0).strip() if m else None
URL = re.compile(r"\b(?:https?://|www\.)\S+|\b(?:github|linkedin)\.com/\S+", re.I)
MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE = re.compile(
    rf"\b(?:\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}|{MONTHS}\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}|\d{{1,2}}\s+{MONTHS}\s+(?:19|20)?\d{{2}}|\d{{4}}-\d{{2}}-\d{{2}})\b",
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
        # v2: form tipi CV'lerde başlık açıklamayla birlikte gelir: "EDUCATION/TRAINING (Begin with ...)"
        m = _LONG_HEAD.match(l)
        if not is_head and m:
            is_head, key = True, m.group(1).lower()
        if is_head:
            if inside:
                break
            inside = key in names
            continue
        if inside:
            out.append(l)
    return out


_LONG_HEAD = re.compile(r"^\s*(EDUCATION|EXPERIENCE|EMPLOYMENT|PUBLICATIONS|SKILLS)\b(?:\s*/\s*[A-Z]+)?\s*[(:]")

_NOT_NAME = SECTION_HEADS + SCHOOL_WORDS + DEGREE_WORDS + ("resume", "curriculum", "vitae", "cv", "page",
                                                           "street", "avenue", "road", "phone", "email", "address",
                                                           # v2: CV başlıkları ve form alanları ad sanılıyordu
                                                           "biographical", "sketch", "personal", "data", "principal",
                                                           "investigator", "program", "director", "grant", "number",
                                                           "position", "title", "research", "associate", "professor",
                                                           "senior", "department", "chief", "staff", "form", "birth")
def _is_name_token(t):
    """"Sena" · "ÖZKAN" · "M." · "Jean-Paul" (Türkçe harfler dahil)."""
    if len(t) == 2 and t[0].isalpha() and t[0].isupper() and t[1] == ".":
        return True
    core = t.replace("-", "")
    return len(t) >= 2 and core.isalpha() and t[0].isupper() and (t[1:].replace("-", "").islower() or t.isupper())
_NAME_SUFFIX = re.compile(r"^(?:M\.?D\.?|Ph\.?D\.?|Jr\.?|Sr\.?)$")
_NAME_LABEL = re.compile(r"^\s*(?:full\s+)?name\b\s*[:;]?\s*(.*)$", re.I)
_NOT_NAME_EXTRA = ("institution", "location", "field", "study", "year", "conferred", "birthdate", "assistant",
                   "fellow", "scientist", "lecturer", "instructor", "manager", "engineer", "head", "member")


def _name_candidate(segment):
    """Satır parçasının başındaki ad-soyadı döndürür: büyük harfle başlayan 2-4 kelime (baş harf "M." olabilir).
    Başlık/kurum kelimesine gelince durur: "Cecilia M. Giachelli Research Associate" → "Cecilia M. Giachelli"."""
    toks, out = segment.strip().split(), []
    for t in toks[:6]:
        core = t.rstrip(",")
        if not _is_name_token(core) or core.lower().strip(".") in _NOT_NAME + _NOT_NAME_EXTRA:
            break
        out.append(core)
        if t.endswith(",") or len(out) == 4:
            break
    if sum(len(w.strip(".")) >= 2 for w in out) < 2:
        return None
    name = " ".join(out)
    rest = toks[len(out):]
    if rest and _NAME_SUFFIX.match(rest[0].strip(",")):     # "Glenn R. Cunningham, M.D."
        name += ", " + rest[0].strip(",")
    return name


def _resume_name(lines):
    # 1) "Name" / "Full Name" etiketi: aynı satırda ya da sonraki 3 satırda
    for i, l in enumerate(lines[:20]):
        m = _NAME_LABEL.match(l)
        if not m:
            continue
        for cand in [m.group(1)] + lines[i + 1:i + 4]:
            for seg in re.split(r"\s*\|\s*|\s{2,}", cand or ""):
                n = _name_candidate(seg)
                if n:
                    return n
    # 2) üst satırlarda satır başında ad (form tipi CV: "Glenn R. Cunningham, M.D. VA Medical Center ...")
    for l in lines[:12]:
        n = _name_candidate(re.split(r"\s*\|\s*", l)[0])
        if n:
            return n
    # 3) eski yöntem: spaCy kişi adı ya da ad gibi görünen satır
    persons = [p for p in _ents("\n".join(lines[:8]), {"PERSON"}) if _looks_like_name(p)]
    return persons[0] if persons else next((l for l in lines[:6] if _looks_like_name(l)), None)


# açık derece: noktalı kısaltmalar (B.A., Ph.D., BS.) ya da "BA in ..."; "MA. 02138" gibi eyalet kodları sayılmaz
_STRONG_DEGREE = re.compile(r"\b(?:B\.\s?A\.?|B\.?\s?S\.|B\.\s?Sc\.?|M\.\s?S\.?|M\.\s?Sc\.?|M\.\s?A\.|Ph\.?\s?D\.?|M\.\s?D\.|"
                            r"BSc|MSc|MBA|(?:BA|BS|MS|MA)(?=\s+in\s))|\b(?:bachelor|master|doctorate)\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


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
    name = _resume_name(lines)

    edu = _section(lines, {"education"}) or lines
    schools = [l for l in edu if any(w in l.lower() for w in SCHOOL_WORDS)]
    # v2: eğitim tablosu satırını (yıl ya da derece içeren okul satırı) tercih et; unvan satırındaki kurumu değil
    school = next((l for l in schools if _YEAR.search(l) or _STRONG_DEGREE.search(l)), schools[0] if schools else None)
    if school is None:
        orgs = [o for o in _ents("\n".join(edu), {"ORG"}) if any(w in o.lower() for w in SCHOOL_WORDS)]
        school = orgs[0] if orgs else None
    # v2: önce açık derece kısaltması (B.A., Ph.D. ...) olan satır; sonra genel kelimeler (uzun açıklama ve tablo başlığı hariç)
    name_key = name.split(",")[0] if name else None
    degree = next((l for l in edu if _STRONG_DEGREE.search(l) and l != school
                   and not (name_key and name_key in l)), None)   # ad satırındaki "M.D." derece satırı değil
    if degree is None and school and _STRONG_DEGREE.search(school):   # "BA in Biochemistry (1978, University of Dublin)"
        degree = school
    if degree is None:
        degree = next((l for l in edu if any(w in l.lower() for w in DEGREE_WORDS) and l != school
                       and len(l.split()) <= 15 and not re.search(r"\binstitution\b|\byear\b", l, re.I)), None)
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
        "telefon": find_phone(text),
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


def _invoice_amounts(lines):
    """v2: tablo satırlarında miktar ve birim fiyat da para gibi görünür (10.00, 50.00). Her satırın en sağdaki
    tutarı (TUTAR sütunu) alınır; belgede en az iki "$"lı tutar varsa sadece onlar tutulur."""
    vals = []
    for l in lines:
        found = [m.strip().rstrip(",.") for m in MONEY.findall(l)]
        if found:
            vals.append(found[-1])
    dollars = [v for v in vals if v[:1] in "$€£₺"]
    vals = dollars if len(dollars) >= 2 else vals
    out = list(dict.fromkeys(vals))
    return out[:5] or None


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
        "tutarlar": _invoice_amounts(lines),
    }


_LABEL_ONLY = re.compile(r"^\s*(?:from|to|cc|ce|co|bcc|subject|subj|re|date|sent|sender|recipient)\s*[:;]\s*$", re.I)


def _header(text, names):
    # v2: değer aynı satırda olmalı ([ \t]*); eskiden \s* satır sonunu da atladığı için "From:" satırının
    # değeri bir alt satırdaki "Date:" sanılıyordu. Değer başka bir etiketse kabul edilmez.
    pat = re.compile(rf"^[ \t]*(?:{names})[ \t]*[:;][ \t]*(\S[^\n]*)$", re.I | re.M)
    for m in pat.finditer(text):
        val = m.group(1).strip()
        if not _LABEL_ONLY.match(val):
            return val
    return None


# OCR "Cc:" etiketini sık sık "Ce:" ya da "co:" diye okuyor
_EMAIL_LABELS = {"from": "from", "sender": "from", "to": "to", "recipient": "to", "cc": "cc", "ce": "cc", "co": "cc",
                 "bcc": "cc", "subject": "subject", "subj": "subject", "re": "subject", "date": "date", "sent": "date"}
_HEAD_PAT = re.compile(r"^\s*(?:from|to|cc|ce|co|bcc|subject|subj|re|date|sent|sender|recipient)\s*[:;]", re.I)
_WEEKDAY_TIME = re.compile(r"\b(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\b.*\d{1,2}:\d{2}|"
                           r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b", re.I)


def _label_block(lines):
    """OCR etiket sütununu (From/Date/Subject/To/Cc) değerlerden ayrı bir blok olarak okuduğunda
    etiketleri sırayla değerlerle eşleştirir. Hizalama tarih satırına göre yapılır."""
    for i in range(len(lines)):
        run = []
        j = i
        while j < len(lines) and _LABEL_ONLY.match(lines[j]):
            run.append(_EMAIL_LABELS[re.match(r"\s*([a-z]+)", lines[j], re.I).group(1).lower()])
            j += 1
        if len(run) < 2:
            continue
        rest = lines[j:j + len(run) + 6]
        if "date" in run:
            d = next((k for k, l in enumerate(rest) if DATE.search(l) or _WEEKDAY_TIME.search(l)), None)
            start = d - run.index("date") if d is not None else None
        else:
            start = None
        if start is None or start < 0:
            start = next((k for k, l in enumerate(rest) if not re.fullmatch(r"[\d\s.-]+", l)), 0)
        vals = rest[start:start + len(run)]
        if len(vals) < len(run):
            return {}, None
        out = {}
        for lab, v in zip(run, vals):
            if lab == "subject":
                v = re.sub(r"^\s*re\s*[:;]\s*", "RE: ", v, flags=re.I)
            out.setdefault(lab, v.strip())
        return out, j + start + len(run)        # gövdenin başladığı satır
    return {}, None


def extract_email(text):
    lines = _lines(text)
    # 1) etiket bloğu + değer bloğu (iki sütunlu başlık)
    block, body_start = _label_block(lines)
    sender, to = block.get("from"), block.get("to")
    subject, date = block.get("subject"), block.get("date")
    # 2) etiket ve değer aynı satırda
    sender = sender or _header(text, "from|sender")
    to = to or _header(text, "to|recipient")
    subject = subject or _header(text, "subject|subj|re")
    date = date or _header(text, "date|sent")
    # 3) etiketler hiç okunmamış (Outlook çıktısı): gönderen, gün+saat satırı, alıcılar, "Subject:"
    if not sender:
        subj_i = next((k for k, l in enumerate(lines[:15]) if re.match(r"\s*(?:subject|subj)\s*[:;]", l, re.I)), None)
        d_i = next((k for k, l in enumerate(lines[:6]) if _WEEKDAY_TIME.search(l) and DATE.search(l)), None)
        if d_i is not None and d_i >= 1 and (subj_i is None or d_i < subj_i):
            sender = lines[d_i - 1].strip(" '‘’")
            date = date or lines[d_i]
            if not to and subj_i is not None and subj_i > d_i + 1:
                to = " ".join(l.strip(" '‘’") for l in lines[d_i + 1:subj_i])[:200]
            body_start = max(body_start or 0, (subj_i + 1) if subj_i is not None else d_i + 1)
    date = date or _first(DATE, text)
    # gövde: başlık satırlarından sonra
    if body_start is None:
        head_idx = [k for k, l in enumerate(lines[:20]) if _HEAD_PAT.match(l)]
        body_start = head_idx[-1] + 1 if head_idx else 0
    body = [l for l in lines[body_start:] if not _HEAD_PAT.match(l)]
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


_NOT_TITLE = re.compile(r"\b(?:printed\s+in|journal|vol\.?|volume|pp\.|fax|copyright|received|accepted|reprint(?:ed)?|"
                        r"sonderdruck|page|inc\.?|wiley|elsevier|press)\b|©|\(c\)", re.I)
_AFFIL = re.compile(r"\b(?:university|universit|college|institute|department|dept|school|hospital|laborator|"
                    r"centre|center|company|corporation|inc)\b", re.I)
_VOCAB = None


def _vocab():
    """İngilizce kelime listesi (SymSpell sıklık sözlüğü); yoksa boş küme → sözlük kontrolü atlanır."""
    global _VOCAB
    if _VOCAB is None:
        try:
            from importlib import resources
            path = resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
            _VOCAB = {l.split()[0] for l in path.read_text().splitlines() if l.strip()}
        except Exception:
            _VOCAB = set()
    return _VOCAB


def _wordlike_ratio(line):
    words = re.findall(r"[A-Za-z]{2,}", line)
    if not words:
        return 0.0
    vocab = _vocab()
    if not vocab:   # sözlük yoksa: sesli harf içeren, makul uzunlukta kelime oranı
        return sum(bool(re.search(r"[aeiouy]", w, re.I)) and len(w) <= 15 for w in words) / len(words)
    return sum(w.lower() in vocab for w in words) / len(words)


def _is_title_line(l):
    words = l.split()
    letters = sum(c.isalpha() for c in l)
    alpha_words = re.findall(r"[A-Za-z]+", l)
    mean_len = sum(map(len, alpha_words)) / max(1, len(alpha_words))   # OCR çöpü kısa parçalardan oluşur
    return (len(words) >= 3 and letters >= 15 and letters / max(1, len(l.replace(" ", ""))) >= 0.85
            and mean_len >= 4 and not _NOT_TITLE.search(l) and not _AFFIL.search(l) and _wordlike_ratio(l) >= 0.7)


def _looks_like_author_line(l):
    """"Pauline M. Tang, Judith Finkelstein" gibi: virgülle ayrılmış, büyük harfle başlayan ad grupları."""
    parts = [p.strip() for p in re.split(r",|\band\b", l) if p.strip()]
    return len(parts) >= 2 and sum(bool(re.match(r"^[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){1,3}$", p)) for p in parts) >= 2


def _scientific_title(lines):
    """v2: dergi künyesi, faks başlığı, "Printed in Great Britain" gibi satırları atlar; başlık iki satıra
    bölünmüşse birleştirir."""
    for i, l in enumerate(lines[:15]):
        if _is_title_line(l) and len(l.split()) >= 4:
            title, j = l, i + 1
            while j < min(i + 3, len(lines)) and _is_title_line(lines[j]) and not re.match(r"^\s*by\b", lines[j], re.I) \
                    and not _looks_like_author_line(lines[j]) and not title.rstrip().endswith((".", ":")):
                title += " " + lines[j]
                j += 1
            prev = lines[i - 1] if i > 0 else ""
            if prev and prev.isupper() and l.isupper() and 1 <= len(prev.split()) <= 6 and not _NOT_TITLE.search(prev) \
                    and sum(c.isalpha() for c in prev) >= 5 and _wordlike_ratio(prev) >= 0.7:
                title = prev + " " + title           # "CHARACTERIZATION" + "OF AIRBORNE PARTICULATE MATTER"
            return title.strip(), j
    return None, 0


def _scientific_authors(lines, after):
    """Başlıktan sonraki 3 satırda: "BY ..." satırı ya da virgül/"and" ile ayrılmış adlar (kurum satırı değil)."""
    for l in lines[after:after + 3]:
        if re.fullmatch(r"[\d\s.,()-]+", l) or _AFFIL.search(l):
            continue
        if re.search(r"[\"*]\s*$", l) or re.match(r"^\W*(?:[IVX]+\.|\d+\.|\.)\s", l):   # alt başlık: "I. Vapor Phase Analysis*"
            continue
        m = re.match(r"^\s*by\s+(.+)$", l, re.I)
        cand = m.group(1) if m else l
        cand = re.split(r",\s*(?:PH|M\.?\s?D|Ph\.?\s?D|F\.?A\.?C\.?P)\b", cand)[0]   # unvanları at
        if m or re.search(r",|\band\b", cand):
            parts = [p.strip(" .") for p in re.split(r",|\band\b", cand) if len(p.strip(" .")) > 1]
            if parts and sum(bool(re.search(r"[A-Za-z]{2,}", p)) for p in parts) >= max(1, len(parts) // 2):
                return parts[:5]
        caps = [w for w in cand.split() if w[:1].isupper() or re.fullmatch(r"\d\.", w)]   # OCR: "J." → "3."
        stop = {"in", "of", "the", "for", "on", "at", "to", "from", "with", "by"}
        if 2 <= len(cand.split()) <= 5 and len(caps) == len(cand.split()) \
                and not any(w.lower() in stop for w in cand.split()):   # tek yazar: "DAVID J. LEWIS"
            return [cand.strip()]
    return None


def extract_scientific(text):
    """Bilimsel makale: başlık, yazarlar, özet (Abstract), anahtar kelimeler."""
    lines = _lines(text)
    title, after = _scientific_title(lines)
    authors = _scientific_authors(lines, after) if title else None
    if title is None:   # eski yöntem
        title = next((l for l in lines[:6] if len(l.split()) >= 4 and sum(c.isalpha() for c in l) > 15), None)
    if authors is None and title is None:
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
        parts = [f.get("ad_soyad"), f.get("universite"),
                 f.get("bolum_derece") if f.get("bolum_derece") != f.get("universite") else None]
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
