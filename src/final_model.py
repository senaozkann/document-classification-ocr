"""Son model: karakter TF-IDF + tablo özellikleri → olasılık veren Linear SVM (Platt kalibrasyonu).

04'teki ek deneyde tablo özelliği (asistan önerisi) Linear SVM'in test başarısını artırdı. Bu modül o
deneyi demoda kullanılabilecek tek bir modele dönüştürür:

    metin (normalize_for_model çıktısı) ──► karakter TF-IDF ─┐
                                                              ├─► Linear SVM (kalibre) ─► olasılıklar
    görüntü ──► detect_table ──► 5 tablo özelliği (ölçekli) ─┘

"Emin değilim": en yüksek olasılık `threshold`'un altındaysa model karar vermez; en olası iki türü
gösterir. Eşik validation setinde seçilir (test setine bakılmaz).
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

TAB_COLS = ["n_hlines", "n_vlines", "n_intersections", "line_ratio", "has_table"]
UNSURE = "emin_degilim"


def char_tfidf():
    # 04'teki karakter TF-IDF ile aynı ayarlar
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_df=0.9,
                           sublinear_tf=True, max_features=100_000)


def tab_array(rows):
    """detect_table sözlükleri ya da TAB_COLS sütunlu DataFrame → (n, 5) dizi; sayılar log ölçekte."""
    if hasattr(rows, "loc"):
        t = rows[TAB_COLS].astype(float).values.copy()
    else:
        t = np.array([[float(r[c]) for c in TAB_COLS] for r in rows])
    t[:, :3] = np.log1p(t[:, :3])
    return t


class TextTableModel:
    def __init__(self, C=0.5, table_weight=0.2, threshold=0.0, seed=42):
        self.C, self.table_weight, self.threshold, self.seed = C, table_weight, threshold, seed

    def _X(self, texts, tabs, fit=False):
        if fit:
            self.vec_ = char_tfidf().fit(texts)
            self.scaler_ = StandardScaler().fit(tab_array(tabs))
        X_text = self.vec_.transform(texts)
        if self.table_weight == 0:
            return X_text.tocsr()
        X_tab = csr_matrix(self.table_weight * self.scaler_.transform(tab_array(tabs)))
        return hstack([X_text, X_tab]).tocsr()

    def fit(self, texts, tabs, y, calibrate=True):
        X = self._X(texts, tabs, fit=True)
        svm = LinearSVC(C=self.C, random_state=self.seed)
        self.clf_ = CalibratedClassifierCV(svm, cv=5, method="sigmoid").fit(X, y) if calibrate else svm.fit(X, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, texts, tabs):
        return self.clf_.predict_proba(self._X(texts, tabs))

    def predict(self, texts, tabs):
        return self.clf_.predict(self._X(texts, tabs))

    def decide(self, texts, tabs, threshold=None):
        """Tahmin + 'emin değilim': en yüksek olasılık eşiğin altındaysa UNSURE döner."""
        t = self.threshold if threshold is None else threshold
        P = self.predict_proba(texts, tabs)
        top = P.argmax(axis=1)
        labels = self.classes_[top].astype(object)
        labels[P.max(axis=1) < t] = UNSURE
        return labels, P


def coverage_table(y_true, P, classes, thresholds):
    """Her eşik için: kapsama (karar verilen belge oranı), karar verilenlerdeki doğruluk, kaçan hata sayısı."""
    y_true = np.asarray(y_true)
    pred = np.asarray(classes)[P.argmax(axis=1)]
    conf = P.max(axis=1)
    rows = []
    for t in thresholds:
        keep = conf >= t
        n_err_all = int((pred != y_true).sum())
        n_err_kept = int((pred[keep] != y_true[keep]).sum())
        rows.append({
            "esik": t,
            "kapsama": keep.mean(),
            "karar_verilenlerde_dogruluk": (pred[keep] == y_true[keep]).mean() if keep.any() else np.nan,
            "emin_degilim_sayisi": int((~keep).sum()),
            "yakalanan_hata": n_err_all - n_err_kept,
            "kalan_hata": n_err_kept,
            "gereksiz_kararsizlik": int(((~keep) & (pred == y_true)).sum()),
        })
    return rows
