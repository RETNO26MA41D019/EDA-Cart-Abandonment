# 🛒 Deteksi Dini Pengabaian Keranjang Belanja (*E-Commerce Cart Abandonment*)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RETNO26MA41D019/EDA-Cart-Abandonment/blob/main/Deteksi%20(Cart_Abandonment)_.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green)

Proyek **Exploratory Data Analysis (EDA)** dan pemodelan klasifikasi prediktif untuk mendeteksi secara dini potensi pengabaian sesi atau keranjang belanja (*cart/session abandonment*) pada platform *e-commerce* berdasarkan jejak klik (*clickstream*) dan metrik perilaku pengunjung.

---

## 📌 Latar Belakang & Tujuan Utama

Dalam industri *e-commerce*, tingginya tingkat pengabaian keranjang belanja menyebabkan hilangnya potensi pendapatan secara langsung. Pada dataset yang dianalisis, **84,53%** sesi kunjungan berakhir tanpa transaksi pembelian (`Revenue = False`).

* **Permasalahan Bisnis:** Ketidakseimbangan kelas (*class imbalance*) yang tajam antara sesi tanpa pembelian (`0`) dan sesi konversi pembelian (`1`), sehingga diperlukan deteksi perilaku navigasi secara *real-time* sebelum pengunjung meninggalkan situs.
* **Tujuan Proyek:**
  1. Mengidentifikasi pola perilaku navigasi halaman, kualitas interaksi, dan faktor temporal yang membedakan pembeli dan non-pembeli.
  2. Membangun *pipeline* pra-pemrosesan data yang bersih dari kebocoran data (*data leakage*), menangani distribusi menceng (*right-skewed*), serta melakukan rekayasa fitur (*feature engineering*).
  3. Mengembangkan dan membandingkan model klasifikasi dini (*Baseline* vs *Cost-Sensitive*) untuk memprediksi potensi konversi/pengabaian sesi.

---

## 📊 Deskripsi Dataset

* **Sumber Dataset:** [Online Shoppers Purchasing Intention Dataset (Kaggle)](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)
* **Dimensi Awal:** `12.330 baris` (sesi unik selama 1 tahun) × `18 kolom` (atribut).
* **Dimensi Bersih (Setelah Deduplikasi):** `12.205 baris` × `18 kolom` (`0% missing values`, menghapus `125` baris duplikat penuh).
* **Variabel Target (`Revenue`):**
  * `0` (`False`): Sesi berakhir tanpa pembelian — **84,53%** (kelas mayoritas).
  * `1` (`True`): Sesi berakhir dengan pembelian — **15,47%** (kelas minoritas).

### Ringkasan Atribut Dataset
| Kategori Fitur | Nama Kolom | Keterangan |
| :--- | :--- | :--- |
| **Perilaku Halaman** | `Administrative`, `Informational`, `ProductRelated` + `*_Duration` | Jumlah halaman yang dikunjungi & total durasi (detik) per kategori halaman. |
| **Metrik Google Analytics** | `BounceRates`, `ExitRates`, `PageValues` | Rasio pentalan, rasio keluar halaman, dan nilai komersial rata-rata halaman. |
| **Temporal** | `Month`, `SpecialDay`, `Weekend` | Bulan kunjungan, kedekatan dengan hari spesial, dan indikator akhir pekan. |
| **Teknis & Segmen** | `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType` | Perangkat pengguna, asal lalu lintas, wilayah, dan segmen pengunjung. |

---

## 🗂️ Struktur Analisis dalam Notebook

1. **Bab 1 — Pendahuluan & Pemahaman Masalah:** Perumusan masalah bisnis, definisi klasifikasi biner, dan pemilihan metrik evaluasi yang sesuai untuk *imbalanced data* (`ROC-AUC`, `Recall`, dan `F1-Score`).
2. **Bab 2 — Memahami Data:** Pengumpulan data terprogram melalui `kagglehub`, tinjauan kualitas data, dan penyelarasan kebutuhan bisnis.
3. **Bab 3 — Memuat & Memeriksa Data:** Inspeksi struktur dimensi, tipe data, dan statistik awal numerik maupun kategorikal.
4. **Bab 4 — Penanganan Data Hilang & Duplikat:** Audit *missing values* (`0%`) menggunakan `missingno`, deduplikasi (`125` baris), penyeragaman format biner, serta simulasi perbandingan metode imputasi (`Simple Median`, `KNN Imputer`, dan `MICE`).
5. **Bab 5 — Eksplorasi Karakteristik Data:** Analisis ukuran pemusatan, penyebaran, *skewness* positif ekstrem (`> 1,9`), dan *kurtosis* leptokurtik (`> 4,6`) melalui visualisasi *Histogram + KDE*.
6. **Bab 6 — Transformasi, Penskalaan, & Rekayasa Fitur:**
   * Transformasi logaritmik (`np.log1p`) untuk meredam kemencengan distribusi.
   * Perbandingan penskalaan (`StandardScaler` & `MinMaxScaler`).
   * Pembuatan **5 fitur baru**: `Total_Pages`, `Total_Duration`, `Avg_Time_Per_Page`, `Product_Page_Ratio`, dan `Has_PageValue`.
7. **Bab 7 — Uji Hipotesis Bivariat:** Uji non-parametrik **Mann-Whitney U** untuk fitur numerik dan **Chi-Square ($\chi^2$)** untuk fitur kategorikal terhadap variabel target `Revenue`.
8. **Bab 8 — Deteksi & Penanganan Pencilan (*Outliers*):** Identifikasi nilai ekstrem menggunakan metode **IQR** serta strategi mitigasi tanpa menghapus sinyal pelanggan bernilai tinggi.
9. **Pemodelan & Evaluasi:** Pelatihan model **Logistic Regression (*Baseline*)** dan **XGBoost Classifier (*Cost-Sensitive*)** menggunakan *Stratified Split* dan `ColumnTransformer`.

---

## 🔍 Temuan Utama (*Key Insights*)

* **Indikator Intensi Belanja Terkuat:** Uji Mann-Whitney U membuktikan seluruh fitur numerik berbeda signifikan ($p < 0,05$) antara pembeli dan non-pembeli. **`PageValues`** ($p \approx 0$) dan **`ExitRates`** ($p = 3,03 \times 10^{-169}$) menjadi pembeda terkuat (median `PageValues` pembeli **16,76** vs non-pembeli **0,00**).
* **Faktor Temporal & Lalu Lintas:** Uji Chi-Square menunjukkan faktor bulan (`Month`), sumber trafik (`TrafficType`), dan tipe pengunjung (`VisitorType`) berpengaruh signifikan terhadap konversi, sedangkan wilayah geografis (`Region`, $p = 0,288$) tidak berpengaruh signifikan.
* **Performa Model Klasifikasi:**

| Model | ROC-AUC | F1-Score (Kelas 1) | Recall | Precision |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.9268 | **0.6646** | **0.8508** | 0.5453 |
| **XGBoost Classifier (Cost-Sensitive)** | **0.9343** | 0.6638 | 0.8089 | **0.5628** |

> **Rekomendasi Implementasi:** **XGBoost Classifier (`ROC-AUC = 0.9343`)** direkomendasikan sebagai model utama karena memberikan kemampuan diskriminasi terbaik dan *Precision* lebih tinggi (`56,28%`), sehingga intervensi promosi otomatis (seperti *pop-up voucher* atau gratis ongkir) lebih tepat sasaran dan efisien secara biaya.

---

## 🛠️ Teknologi & Pustaka yang Digunakan

* **Bahasa Pemrograman:** Python 3
* **Manipulasi & Analisis Data:** `pandas`, `numpy`, `scipy`, `kagglehub`
* **Visualisasi Data:** `matplotlib`, `seaborn`, `missingno`
* **Machine Learning & Interpretabilitas:** `scikit-learn`, `xgboost`, `shap`

---

## 🚀 Cara Menjalankan Proyek

1. **Jalankan Langsung di Google Colab (Direkomendasikan):**
   Klik tombol **"Open in Colab"** di bagian atas halaman ini, lalu jalankan seluruh sel secara berurutan (`Runtime` > `Run all`).
2. **Jalankan secara Lokal:**
   ```bash
   git clone [https://github.com/RETNO26MA41D019/EDA-Cart-Abandonment.git](https://github.com/RETNO26MA41D019/EDA-Cart-Abandonment.git)
   cd EDA-Cart-Abandonment
   pip install kagglehub missingno xgboost shap scikit-learn seaborn matplotlib pandas numpy scipy
   jupyter notebook "Deteksi (Cart_Abandonment)_.ipynb"
