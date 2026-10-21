"""Tüm veri setini aynı ayarlarla yeniden OCR'lar (21 Ekim: x2 büyütme deneyi).

Her belgeye aynı işlem uygulanır: gri görüntü → `scale` kat büyütme (bikübik) → Tesseract (LSTM, psm 3).
Belgeye özel hiçbir ayar yoktur. Sonuçlar parça parça CSV'ye yazılır; yarıda kalırsa kaldığı yerden devam eder.
"""
import os

os.environ.setdefault("OMP_THREAD_LIMIT", "1")   # her süreç tek çekirdek: paralel süreçler birbirini yavaşlatmasın

import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd

from ocr_core import run_ocr


def _work(args):
    data_dir, rel_path, scale = args
    img = cv2.imread(str(Path(data_dir) / rel_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"path": rel_path, "text": "", "mean_conf": 0.0, "n_words": 0, "hata": "okunamadı"}
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    try:
        o = run_ocr(img)
        return {"path": rel_path, "text": o["text"], "mean_conf": round(o["mean_conf"], 2),
                "n_words": o["n_words"], "hata": ""}
    except Exception as e:  # tek belge hatası tüm işi durdurmasın
        return {"path": rel_path, "text": "", "mean_conf": 0.0, "n_words": 0, "hata": str(e)[:200]}


def run_batch(paths, data_dir, out_csv, scale=2.0, workers=2, chunk=40, log=print):
    out_csv = Path(out_csv)
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["path"])
    todo = [p for p in paths if p not in set(done["path"])]
    log(f"Toplam {len(paths)} · daha önce yapılan {len(paths) - len(todo)} · kalan {len(todo)}")
    t0 = time.time()
    with Pool(workers) as pool:
        for k in range(0, len(todo), chunk):
            part = todo[k:k + chunk]
            rows = pool.map(_work, [(str(data_dir), p, scale) for p in part])
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            tmp = out_csv.with_suffix(".tmp.csv")
            done.to_csv(tmp, index=False)
            tmp.replace(out_csv)                      # yarım yazılmış dosya kalmasın
            n = k + len(part)
            el = time.time() - t0
            log(f"{n}/{len(todo)} · {el / 60:.1f} dk · kalan ~{el / n * (len(todo) - n) / 60:.0f} dk")
    return done
