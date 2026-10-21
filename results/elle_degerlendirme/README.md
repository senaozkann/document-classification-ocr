# İçerik çıkarımının elle değerlendirilmesi (21 Ekim 2026)

Test setinden her türden 5 belge (20 belge, 110 alan). Her alan için karar:
Doğru (d) · Yanlış (y) · Kaçırıldı (k) · Belgede yok (b). Küçük OCR harf hataları doğru sayıldı.

| Dosya | İçerik |
|---|---|
| `elle_kararlar.txt` | Belge başına kararlar (belge no + harfler, alan sırası notebook 09 ile aynı) |
| `cikarim_degerlendirme_v2.csv` | 110 alanın her biri: belge, tür, alan, sistemin değeri, karar |
| `cikarim_sonuclari_tur_v2.csv` | Türe göre precision / recall / doğruluk |
| `cikarim_sonuclari_v2.csv` | Alana göre precision / recall / doğruluk |

Toplam: precision 0,525 · recall 0,744 · doğruluk 0,636.

## Önce / sonra (21 Ekim, çıkarım kuralları v2 · notebook 10)

Aynı 20 belge, aynı OCR metni. Değeri değişen 35 alan resme bakılarak yeniden işaretlendi; değişmeyenler eski kararını korudu.

| Tür | Precision önce → sonra | Recall önce → sonra | Doğruluk önce → sonra |
|---|---|---|---|
| E-posta | 0,60 → 0,95 | 0,86 → 1,00 | 0,67 → 0,97 |
| Fatura | 0,57 → 0,71 | 0,53 → 0,59 | 0,48 → 0,56 |
| CV | 0,47 → 1,00 | 0,89 → 0,87 | 0,67 → 0,93 |
| Bilimsel makale | 0,40 → 0,89 | 0,80 → 0,80 | 0,72 → 0,88 |
| **Toplam** | **0,525 → 0,897** | **0,744 → 0,825** | **0,636 → 0,845** |

**Uyarı:** Kurallar bu 20 belgedeki hatalara bakılarak düzeltildi; sonuç iyimserdir. Gerçek başarıyı görmek için kurallar
yeni (görülmemiş) belgelerde ölçülmelidir.

Dosyalar: `cikarim_degerlendirme_v2_once_sonra.csv` (karar = önce, karar_v2 = sonra), `cikarim_once_sonra_ozet.csv`.
