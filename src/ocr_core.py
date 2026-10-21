import cv2
import numpy as np
import pytesseract


def estimate_skew(gray, max_angle=5.0, step=0.25):
    """Projection-profile skew estimate: the angle whose row sums are sharpest."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    _, th = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.medianBlur(th, 3)  # drop isolated noise pixels
    h, w = th.shape
    best_angle, best_score = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + 1e-9, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(r.sum(axis=1, dtype=np.float64)))
        if score > best_score:
            best_angle, best_score = float(a), score
    return best_angle


def deskew(gray):
    """Rotate the page so text lines are horizontal."""
    angle = estimate_skew(gray)
    if abs(angle) < 0.3:
        return gray, 0.0
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def preprocess(img_bgr_or_gray, upscale_to=2000, do_deskew=True, binarize=True):
    """Grayscale -> upscale (RVL-CDIP is max 1000px) -> denoise -> deskew -> Otsu."""
    img = img_bgr_or_gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.medianBlur(gray, 3)  # salt-and-pepper scan noise, before upscaling
    h, w = gray.shape
    scale = upscale_to / max(h, w)
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    angle = 0.0
    if do_deskew:
        gray, angle = deskew(gray)
    if binarize:
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, angle


def run_ocr(img, lang="eng", psm=3, min_conf=0):
    """Tesseract OCR -> words, normalized boxes (0-1000, LayoutLM format), line text."""
    h, w = img.shape[:2]
    d = pytesseract.image_to_data(img, lang=lang, config=f"--oem 1 --psm {psm}",
                                  output_type=pytesseract.Output.DICT)
    words, boxes, confs, lines = [], [], [], {}
    for i, txt in enumerate(d["text"]):
        txt = txt.strip()
        conf = float(d["conf"][i])
        if not txt or conf < min_conf:
            continue
        x, y, bw, bh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        words.append(txt)
        confs.append(conf)
        boxes.append([int(1000 * x / w), int(1000 * y / h),
                      int(1000 * (x + bw) / w), int(1000 * (y + bh) / h)])
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        lines.setdefault(key, []).append(txt)
    text = "\n".join(" ".join(v) for _, v in sorted(lines.items()))
    return {
        "text": text,
        "words": words,
        "boxes": boxes,
        "confs": confs,
        "n_words": len(words),
        "mean_conf": float(np.mean(confs)) if confs else 0.0,
    }


def detect_table(gray):
    """Morfolojik işlemlerle yatay/dikey çizgileri bulup belgede tablo olup olmadığını tahmin eder.

    OCR tabloları bozarak okuduğu için (danışman/asistan görüşmesi) tablo bilgisi
    görüntüden ayrıca çıkarılır ve sınıflandırmada ek özellik olarak kullanılabilir.
    """
    img = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 25)))
    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(th, cv2.MORPH_OPEN, vk)
    n_h = len(cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    n_v = len(cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    inter = cv2.bitwise_and(cv2.dilate(horiz, np.ones((3, 3), np.uint8)), cv2.dilate(vert, np.ones((3, 3), np.uint8)))
    n_x = len(cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    line_ratio = float((horiz > 0).mean() + (vert > 0).mean())
    return {
        "n_hlines": n_h,
        "n_vlines": n_v,
        "n_intersections": n_x,
        "line_ratio": round(line_ratio, 5),
        "has_table": bool(n_h >= 3 and n_v >= 2 and n_x >= 4),
    }


def rotate_bound(gray, angle):
    """Görüntüyü kırpmadan döndürür (köşeler beyazla doldurulur)."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=255)


def _good_words(o, min_conf=70):
    return sum(c >= min_conf for c in o["confs"])


def upscale_if_small(gray, ocr, target_px=24):
    """Yazı küçükse görüntüyü büyütüp yeniden okur (21 Ekim).

    Tesseract harf yüksekliği ~20-30 piksel olduğunda en iyi çalışır. RVL-CDIP taramalarında kelime yüksekliği
    7-12 piksel; internetten alınan küçük resimlerde daha da az. Kelime kutularının ortanca yüksekliği
    hedefin altındaysa görüntü büyütülür; güvenli (>= 70) okunan kelime sayısı artarsa büyütülmüş okuma seçilir.
    """
    hs = [b[3] - b[1] for b in ocr["boxes"]]
    if not hs:
        med = 0.0
    else:
        med = float(np.median(hs)) * gray.shape[0] / 1000.0
    if med >= target_px * 0.8 and _good_words(ocr) > 0:
        return gray, ocr, 1.0
    f = 2.0   # denemelerde x2 en dengelisi: x3 bazı tablo sütunlarını kaybettirdi
    big = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    o2 = run_ocr(big)
    return (big, o2, f) if _good_words(o2) > 1.1 * _good_words(ocr) else (gray, ocr, 1.0)


def ocr_with_orientation(img, upscale=True):
    """Yan, eğik ya da ters dönmüş sayfayı düzeltip OCR yapar.

    21 Ekim: elle değerlendirmede ~77° dönük bir fatura hiç okunamamıştı ("WoRsN"). Yazı satırlarının açısı,
    satır toplamlarının en keskin olduğu açı aranarak (-90°…+90°) bulunur; satır yönü 180° belirsiz olduğu için
    sayfa o açıyla ve ters yönüyle okunur. Açı küçükse Tesseract'ın yön tespiti (OSD) baş aşağı sayfaya bakar.
    Aday hâller arasından en çok ve en emin kelimeyi veren seçilir; asıl hâlden en az %20 iyi değilse sayfaya
    dokunulmaz, böylece düz belgeler bozulmaz.
    Dönüş: (görüntü, ocr sonucu, döndürme açısı)
    """
    import re
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ocr = run_ocr(gray)
    score = lambda o: o["n_words"] * o["mean_conf"]
    angles = []
    a = estimate_skew(gray, max_angle=90, step=1.0)
    if abs(a) >= 3:
        angles += [a, a + 180]
    else:
        try:
            osd = pytesseract.image_to_osd(gray, config="--psm 0 -c min_characters_to_try=5")
            if int(re.search(r"Rotate: (\d+)", osd).group(1)) == 180:
                angles.append(180)
        except Exception:
            pass
    best = (gray, ocr, 0)
    for ang in angles:
        r = rotate_bound(gray, ang)
        o = run_ocr(r)
        if score(o) > 1.2 * score(best[1]):
            best = (r, o, ang)
    if upscale:
        g2, o2, _ = upscale_if_small(best[0], best[1])
        best = (g2, o2, best[2])
    return best
