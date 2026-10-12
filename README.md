# document-classification-ocr

OCR-based document text extraction and document classification using machine learning.

**OCR Tabanlı Belge Sınıflandırma** —

Taranmış belge görüntüleri OCR ile metne çevrilir, metin temizlenir ve belge türü makine öğrenmesiyle tahmin edilir.

## Sınıflar

| Sınıf | RVL-CDIP adı |
|---|---|
| E-posta | `email` |
| Form | `form` |
| Fatura | `invoice` |
| CV | `resume` |

## Pipeline

```
Belge görüntüsü
   │
   ▼  OCR: Tesseract 5 (LSTM), ham görüntü → kelimeler, konum kutuları, güven skorları
   ▼  Metin temizleme: OCR gürültüsü, tireleme, boşluk; e-posta/tutar/sayı normalizasyonu
   ▼  Özellik çıkarımı: TF-IDF
   ▼  Sınıflandırma: Multinomial Naive Bayes · Logistic Regression · Linear SVM
   ▼
Belge türü
```

## Veri seti

[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) (Harley ve ark., ICDAR 2015) — 400.000 taranmış belge, 16 sınıf.
Projede orijinal train/val/test ayrımını koruyan dengeli alt küme ([`jinhybr/rvl_cdip_400_train_val_test`](https://huggingface.co/datasets/jinhybr/rvl_cdip_400_train_val_test)) kullanılır:

| Bölüm | Sınıf başına | Toplam (4 sınıf) |
|---|---|---|
| Train | 400 | 1.600 |
| Validation | 100 | 400 |
| Test | 100 | 400 |

Görüntüler lisans ve boyut nedeniyle repoya eklenmez; `01_data_preparation.ipynb` otomatik indirir.

## Notebook'lar

| Notebook | Açıklama | Çıktı |
|---|---|---|
| [`01_data_preparation`](notebooks/01_data_preparation.ipynb) | Veri setini indirir, 4 sınıfı filtreler, görüntüleri kaydeder | `metadata.csv` |
| [`02_ocr_pipeline`](notebooks/02_ocr_pipeline.ipynb) | Tesseract OCR (ham görüntü, v2), kelime kutuları, OCR istatistikleri | `ocr_dataset.csv` |
| [`03_ocr_analysis_cleaning`](notebooks/03_ocr_analysis_cleaning.ipynb) | Ön işleme pilot deneyi, OCR kalite analizi, en iyi/kötü okunan belgeler, metin temizleme, etiket kontrolleri | `cleaned_dataset.csv` |
| [`04_classification`](notebooks/04_classification.ipynb) | TF-IDF (kelime + karakter) → NB / LR / Linear SVM, accuracy · precision · recall · F1, confusion matrix, sınıf bazlı analiz, şüpheli etiketler | `best_model.joblib` |
| `05_end_to_end_demo` *(yakında)* | Yeni belge → OCR → tür tahmini | — |

## Bulgular (devam ediyor)

### Ön işlemenin OCR'a etkisi (pilot deney, 40 validation belgesi)

| Ayar | Ortalama OCR güveni | Sözlükte bulunan kelime oranı |
|---|---|---|
| **Ham görüntü (seçilen)** | **65,9** | **0,591** |
| Median filtre + büyütme | 44,9 | 0,448 |
| + eğiklik düzeltme | 45,5 | 0,460 |
| + Otsu ikilileştirme | 38,3 | 0,405 |

Klasik ön işleme adımları, RVL-CDIP'in düşük çözünürlüklü taramalarında ince harf çizgilerini bozarak OCR'ı kötüleştirdi; bu yüzden OCR ham görüntü üzerinde çalıştırıldı.

### Sınıf bazında OCR (2.400 belge, ham görüntü)

| Sınıf | Ort. kelime | Ort. OCR güveni | Boş metin (< 5 kelime) |
|---|---|---|---|
| email | 116,0 | 76,8 | 3 |
| form | 140,3 | 61,1 | 11 |
| invoice | 100,7 | 53,1 | 19 |
| resume | 314,1 | 79,0 | 6 |

Düz metin ağırlıklı e-posta ve CV'ler iyi okunurken tablo, kutucuk ve el yazısı içeren fatura ve formlarda OCR güveni belirgin şekilde düşüktür.

## Kaynak kod

| Dosya | İçerik |
|---|---|
| [`src/ocr_core.py`](src/ocr_core.py) | `preprocess()` görüntü ön işleme, `run_ocr()` Tesseract OCR |
| [`src/text_clean.py`](src/text_clean.py) | `clean_text()`, `normalize_for_model()`, `garbage_ratio()` |

## Çalıştırma (Google Colab)

1. Colab'da **File → Open notebook → GitHub** sekmesinden bu repoyu açın.
2. Notebook'ları sırayla (01 → 02 → 03 → 04) **Runtime → Run all** ile çalıştırın.
3. Tüm çıktılar Google Drive'da `MyDrive/bitirme_ocr/` altına kaydedilir; oturum kapanırsa kaldığı yerden devam eder.

OCR adımı CPU'da çalışır (2.400 belge ≈ 30–60 dk); GPU gerekmez.
