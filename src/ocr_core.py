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
