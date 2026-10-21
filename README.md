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

Sınıf listesi Drive'daki `siniflar.json` ile değiştirilebilir (06). Form yerine aday sınıflar: `letter` (resmi yazı / dilekçe), `news_article`, `scientific_publication`.

## Pipeline

```
Belge görüntüsü
   │
   ▼  OCR: Tesseract 5 (LSTM), ham görüntü → kelimeler, konum kutuları, güven skorları
   ▼  Metin temizleme: OCR gürültüsü, tireleme, boşluk; e-posta/tutar/sayı normalizasyonu
   ▼  Özellik çıkarımı: TF-IDF
   ▼  Sınıflandırma: Multinomial Naive Bayes · Logistic Regression · Linear SVM
   ▼  İçerik çıkarımı: türe özel kurallar + spaCy NER
   ▼  Konu özeti: anahtar kelimeler + 2 cümlelik çıkarımsal özet · tablo tespiti
   ▼
Belge türü + içerik + konu  (ör. "Bu bir CV: ad, üniversite, bölüm · Konusu: ...")
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
| [`04_classification`](notebooks/04_classification.ipynb) | TF-IDF (kelime + karakter) → NB / LR / Linear SVM, accuracy · precision · recall · F1, confusion matrix, sınıf bazlı analiz, şüpheli etiketler, tablo özelliği deneyi | `best_model.joblib` |
| [`05_end_to_end_demo`](notebooks/05_end_to_end_demo.ipynb) | Yeni belge (resim/PDF) → OCR → tür tahmini → türe özel içerik çıkarımı + konu özeti + tablo tespiti, elle değerlendirme, Gradio demo | `cikarim_sonuclari.csv` |
| [`06_sinif_secimi`](notebooks/06_sinif_secimi.ipynb) | 16 sınıfa bakış; form yerine aday sınıflar (letter, news_article, scientific_publication) indirilir, OCR'dan geçirilir ve aynı modelle karşılaştırılır; seçim `siniflar.json`'a yazılır, 03–05 bu listeyle çalışır | `sinif_secimi_sonuclari.csv`, `siniflar.json` |
| [`07_sablon_ve_sozluk`](notebooks/07_sablon_ve_sozluk.ipynb) | Her türün ortak ifadelerinden şablon (tek başına ve TF-IDF'e destek olarak); sözlükle OCR hata düzeltme (OCR karışıklık kuralları + SymSpell) ve sınıflandırmaya etkisi | `sablon_sonuclari.csv`, `sozluk_sonuclari.csv` |
| [`08_gurultu_giderme`](notebooks/08_gurultu_giderme.ipynb) | Median, Gauss, non-local means, Fourier (alçak geçiren, katsayı eşikleme) ve küçük leke silme: hangisi metni en az bozuyor? | `gurultu_sonuclari.csv` |
| [`09_elle_degerlendirme`](notebooks/09_elle_degerlendirme.ipynb) | İçerik çıkarımının elle değerlendirilmesi: belge resmi + çıkarılan alanlar, Doğru / Yanlış / Kaçırıldı / Belgede yok; precision, recall; 20 belgelik inceleme PDF'i | `cikarim_sonuclari_v2.csv`, `cikarim_inceleme_20belge.pdf` |
| [`11_ocr_x2`](notebooks/11_ocr_x2.ipynb) | Tüm veri setinin 2 kat büyütülmüş görüntüden yeniden OCR'lanması; OCR kalitesi ve sınıflandırma karşılaştırması (karar validation'da) | `data/ocr_dataset_x2.csv`, `ocr_x2_kalite.csv`, `ocr_x2_siniflandirma.csv` |
| [`10_son_model`](notebooks/10_son_model.ipynb) | Son model (karakter TF-IDF + tablo özelliği, kalibre Linear SVM), "emin değilim" çıktısı, çıkarım kurallarının önce/sonra karşılaştırması, dönük sayfa düzeltme | `models/final_model.joblib`, `son_model_karsilastirma.csv`, `cikarim_once_sonra.csv` |

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

### Sınıflandırma (test seti, 400 belge)

| Model | Özellik | Accuracy | Macro-F1 |
|---|---|---|---|
| **Linear SVM** | karakter TF-IDF | **0,9075** | **0,908** |
| Logistic Regression | karakter TF-IDF | 0,9075 | 0,908 |
| Linear SVM | kelime TF-IDF | 0,9025 | 0,904 |
| Logistic Regression | kelime TF-IDF | 0,9000 | 0,901 |
| Naive Bayes | karakter TF-IDF | 0,8975 | 0,898 |
| Naive Bayes | kelime TF-IDF | 0,8875 | 0,889 |

Sınıf bazında F1 (Linear SVM, karakter): resume 0,980 · email 0,918 · invoice 0,873 · form 0,862.
En sık hata, formların fatura olarak tahmin edilmesidir (100 formdan 12'si). Karakter TF-IDF, OCR hatalarına dayanıklılığı sayesinde her algoritmada kelime TF-IDF'ten biraz daha iyi sonuç vermiştir. İlk dört model arasındaki farklar 400 belgelik test setinde istatistiksel olarak anlamlı değildir.

## Kaynak kod

| Dosya | İçerik |
|---|---|
| [`src/ocr_core.py`](src/ocr_core.py) | `preprocess()` görüntü ön işleme, `run_ocr()` Tesseract OCR, `detect_table()` tablo tespiti |
| [`src/text_clean.py`](src/text_clean.py) | `clean_text()`, `normalize_for_model()`, `garbage_ratio()` |
| [`src/extract.py`](src/extract.py) | Türe özel içerik çıkarımı (regex + spaCy NER) ve `TopicSummarizer` konu özeti |
| [`src/sablon.py`](src/sablon.py) | Belge türü şablonları: her türün ortak ifadeleri, kural tabanlı tahmin, model için şablon özellikleri |
| [`src/ocr_fix.py`](src/ocr_fix.py) | Sözlükle OCR düzeltme: OCR karışıklık kuralları, alan adı kuralı, SymSpell |

## Çalıştırma (Google Colab)

1. Colab'da **File → Open notebook → GitHub** sekmesinden bu repoyu açın.
2. Notebook'ları sırayla (01 → 02 → 03 → 04 → 05) **Runtime → Run all** ile çalıştırın.
3. Tüm çıktılar Google Drive'da `MyDrive/bitirme_ocr/` altına kaydedilir; oturum kapanırsa kaldığı yerden devam eder.

OCR adımı CPU'da çalışır (2.400 belge ≈ 30–60 dk); GPU gerekmez.
