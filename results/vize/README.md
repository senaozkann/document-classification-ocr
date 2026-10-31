# Vize ekleri (notebook 13 · 31 Ekim 2026)

22 Ekim danışman görüşmesindeki isteklere göre. Çıktıların tamamı Drive'da `bitirme_ocr/vize/` altında; burada özet tablolar var.

- `test_tamami_ocr_ve_siniflandirma.csv`: 400 test belgesinin tamamı; OCR'ın okuyamadığı belge (temizlik sonrası < 5 kelime) ile modelin hatası ayrı. Model: 04'ün en iyi modeli (kelime TF-IDF + LR).
- `sablon_adim_adim.csv`: CV şablonuna ifadeler birer birer eklenince sadece şablonla sınıflandırma (test). *research* eklemek sonucu düşürdü (67,2 → 62,7).
- `sozluk_asamalari.csv`: ham OCR → genel sözlük (SymSpell, 82.834 kelime) → iki aşamalı sözlük (genel + 380 kelimelik alan sözlüğü, sadece train'den).
  Genel sözlük 88.654 düzeltmenin 6.927'sinde (%7,8) alan kelimesini bozdu (DNA → DAN, NIH → NIL, vitro → nitro). İki aşamalıda 0. Sınıflandırma değişmedi.
  İlk denemede alan sözlüğüne OCR hataları da girdi (wih, fom, folowing, ofthe: 1.060 kelime); sık bir kelimeye 1 harf uzak olanlar ve birleşik kelimeler çıkarılınca 380 kelime kaldı.
