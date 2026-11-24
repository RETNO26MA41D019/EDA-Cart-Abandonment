# 🛒 Deteksi Dini Pengabaian Keranjang Belanja (*E-Commerce Cart Abandonment*)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RETNO26MA41D019/EDA-Cart-Abandonment/blob/main/Deteksi%20(Cart_Abandonment)_.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green)

Proyek **Exploratory Data Analysis (EDA)** dan pemodelan klasifikasi prediktif untuk mendeteksi secara dini potensi pengabaian sesi atau keranjang belanja (*cart/session abandonment*) pada platform *e-commerce* berdasarkan jejak klik (*clickstream*) dan metrik perilaku pengunjung.

Nama: Retno Setioningrum
NIM : 26MA41D019

## 📌 Latar Belakang & Tujuan Utama

Dalam industri *e-commerce*, tingkat pengabaian sesi atau keranjang belanja (*cart abandonment*) menjadi tantangan bisnis utama karena menyebabkan hilangnya potensi pendapatan secara langsung. Pada dataset yang dianalisis, terdapat ketimpangan kelas (*imbalanced dataset*) di mana mayoritas sesi pengunjung tidak berakhir pada transaksi pembelian.

* **Permasalahan Bisnis:** Tingginya proporsi sesi pengunjung tanpa pembelian:
  * **Dataset Mentah (`12.330` sesi):** **84,53%** (`10.422` sesi) berakhir tanpa pembelian (`Revenue = False`), dan hanya **15,47%** (`1.908` sesi) berakhir pada transaksi pembelian (`Revenue = True`).
  * **Dataset Bersih Setelah Deduplikasi (`12.205` sesi):** **84,37%** (`10.297` sesi) berakhir tanpa pembelian dan **15,63%** (`1.908` sesi) berakhir pada transaksi pembelian.
* **Tujuan Utama:**
  1. Mengidentifikasi pola perilaku navigasi pengguna, kualitas interaksi halaman, dan faktor temporal yang membedakan pembeli (`Revenue = 1`) dengan pengunjung yang meninggalkan sesi (`Revenue = 0`).
  2. Membangun *pipeline* pra-pemrosesan data yang bebas kebocoran data (*data leakage*), menangani kemencengan distribusi (*skewness*), serta melakukan rekayasa fitur (*feature engineering*).
  3. Mengembangkan dan membandingkan model klasifikasi biner (*Baseline* vs *Cost-Sensitive*) untuk memprediksi potensi konversi/pengabaian sesi secara *real-time*.

---

## 📊 Ringkasan & Kualitas Dataset

* **Sumber Dataset:** [Online Shoppers Purchasing Intention Dataset (Kaggle)](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset) (`online_shoppers_intention.csv`).
* **Dimensi Data:**
  * **Data Mentah:** `12.330 baris` × `18 kolom`.
  * **Audit *Missing Values*:** **0,0%** nilai kosong (`0` *missing values*) di seluruh 18 kolom.
  * **Audit Duplikat:** Ditemukan **`125` baris duplikat penuh (`1,01%`)** dari sesi *bounce* identik tanpa aktivitas halaman.
  * **Data Bersih (Setelah Deduplikasi):** `12.205 baris` × `18 kolom`.

### Komposisi Atribut Dataset (18 Kolom)
| Kategori | Nama Fitur | Tipe & Deskripsi |
| :--- | :--- | :--- |
| **Perilaku Halaman (6)** | `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration` | Jumlah halaman yang dikunjungi (`int64`) & total durasi akses dalam detik (`float64`) per kategori halaman. |
| **Metrik Analitik (3)** | `BounceRates`, `ExitRates`, `PageValues` | Rasio pentalan, rasio keluar halaman, dan nilai komersial halaman (`float64`). |
| **Temporal & Segmen (4)** | `Month`, `SpecialDay`, `VisitorType`, `Weekend` | Bulan kunjungan (`10` bulan unik), kedekatan hari spesial, segmen pengunjung (`3` kategori), dan indikator akhir pekan (`0`/`1`). |
| **Teknis & Trafik (4)** | `OperatingSystems`, `Browser`, `Region`, `TrafficType` | Kode kategori nominal sistem operasi (`1–8`), peramban (`1–13`), wilayah (`1–9`), dan sumber trafik (`1–20`). |
| **Target Klasifikasi (1)** | `Revenue` | `0` = Tidak Membeli (*Abandonment*), `1` = Membeli (*Conversion*). |

---

## 🗂️ Alur Analisis dalam Notebook

1. **Bab 1 — Pendahuluan & Pemahaman Masalah:** Perumusan masalah bisnis, definisi klasifikasi biner, dan penetapan metrik evaluasi untuk *imbalanced dataset* (`ROC-AUC`, `Recall`, dan `F1-Score`).
2. **Bab 2 — Memahami Data:** Pengumpulan data terprogram melalui `kagglehub`, tinjauan kualitas data, dan penyelarasan kebutuhan bisnis.
3. **Bab 3 — Memuat & Memeriksa Data:** Inspeksi dimensi (`12.330 × 18`), tipe data (`df.info()`), dan ringkasan statistik awal numerik maupun kategorikal.
4. **Bab 4 — Penanganan Data Hilang & Duplikat:** Pemeriksaan *missing values* (`0,0%`) dengan `missingno`, penghapusan `125` baris duplikat (`1,01%`) menjadi `12.205` baris bersih, konversi boolean ke biner (`0`/`1`), serta simulasi perbandingan 3 metode imputasi (`Simple Median`, `KNN Imputer`, dan `MICE`) pada `5%` *missing* buatan.
5. **Bab 5 — Eksplorasi Karakteristik Data:** Analisis ukuran pemusatan, penyebaran, *skewness* positif ekstrem (`1,947` hingga `7,540`), dan *kurtosis* leptokurtik (`4,641` hingga `136,681`) melalui tabel statistik lengkap dan visualisasi *Histogram + KDE*.
6. **Bab 6 — Transformasi, Penskalaan, & Rekayasa Fitur:**
   * Transformasi logaritmik (`np.log1p`) untuk menurunkan kemencengan distribusi (*skewness*).
   * Perbandingan penskalaan `StandardScaler` ($\mu=0, \sigma=1$) dan `MinMaxScaler` ($[0, 1]$).
   * Rekayasa **5 fitur baru**: `Total_Pages`, `Total_Duration`, `Avg_Time_Per_Page`, `Product_Page_Ratio`, dan `Has_PageValue`.
7. **Bab 7 — Uji Hipotesis Bivariat:** Uji non-parametrik **Mann-Whitney U** untuk 10 fitur numerik dan uji **Chi-Square ($\chi^2$)** untuk 7 fitur kategorikal terhadap `Revenue` ($\alpha = 0,05$).
8. **Bab 8 — Deteksi & Penanganan Pencilan (*Outliers*):** Identifikasi nilai ekstrem dengan metode **IQR** serta strategi mitigasi menggunakan transformasi logaritmik dan model berbasis pohon tanpa menghapus sinyal pelanggan bernilai tinggi.
9. **Pemodelan & Evaluasi:** Pelatihan model **Logistic Regression (*Baseline*)** dan **XGBoost Classifier (*Cost-Sensitive*)** menggunakan *Stratified Train-Test Split* dan `ColumnTransformer`.

---

## 🔍 Temuan Utama (*Key Insights*)

* **Pola Trafik & Segmen Pengunjung (Data Mentah `12.330` Sesi):**
  * **Dominasi Pengunjung Lama:** **`Returning_Visitor`** mendominasi sebesar **85,57%** (`10.551` sesi).
  * **Puncak Trafik Musiman:** Bulan **Mei (`May`)** mencatat jumlah kunjungan tertinggi sebanyak **27,28%** (`3.364` sesi).
  * **Aktivitas Hari Kerja:** Sebanyak **76,74%** (`9.462` sesi) terjadi pada hari kerja (`Weekend = False`).
* **Karakteristik Perilaku Navigasi (Data Bersih `12.205` Sesi):**
  * Pengunjung paling banyak mengakses halaman produk (`ProductRelated` rata-rata **32,046 halaman**, median **18,000**) dengan rata-rata durasi **1.206,982 detik** (~20,1 menit; median **608,943 detik**).
  * Lebih dari 75% sesi memiliki nilai `PageValues = 0,000` (persentil ke-25 hingga ke-75 = `0,000`), sementara rata-ratanya sebesar **5,950** dan maksimumnya mencapai **361,764**.
* **Pembeda Utama Pembeli vs Non-Pembeli (Uji Statistik $\alpha = 0,05$):**
  * Seluruh 10 fitur numerik berbeda signifikan ($p < 0,05$) berdasarkan **Uji Mann-Whitney U**, dengan **`PageValues`** ($p \approx 0,0000$; median pembeli **16,7581** vs non-pembeli **0,0000**) dan **`ExitRates`** ($p = 3,0299 \times 10^{-169}$; median pembeli **0,0160** vs non-pembeli **0,0284**) sebagai pembeda paling kuat.
  * Berdasarkan **Uji Chi-Square ($\chi^2$)**, 6 dari 7 fitur kategorikal berhubungan signifikan dengan `Revenue` (terkuat pada `Month` $p = 1,5668 \times 10^{-75}$ dan `TrafficType` $p = 1,2633 \times 10^{-64}$), sedangkan **`Region`** ($p = 0,28826 > 0,05$) tidak signifikan.
* **Performa Evaluasi Model Klasifikasi:**

| Model | ROC-AUC | F1-Score (Kelas 1) | Recall | Precision |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.9268 | **0.6646** | **0.8508** | 0.5453 |
| **XGBoost Classifier (Cost-Sensitive)** | **0.9343** | 0.6638 | 0.8089 | **0.5628** |

> **Rekomendasi Implementasi:** **XGBoost Classifier (`ROC-AUC = 0.9343`)** dipilih sebagai model utama karena memiliki daya diskriminasi global tertinggi dan *Precision* lebih baik (**56,28%**), sehingga intervensi promosi *real-time* (seperti *pop-up* diskon atau *voucher* gratis ongkir) lebih akurat dan efisien secara biaya dengan tingkat deteksi pembeli (*Recall*) tetap di atas **80%** (**80,89%**).

---

## 🛠️ Teknologi & Pustaka yang Digunakan

* **Bahasa Pemrograman:** Python 3
* **Manipulasi & Statistik Data:** `pandas`, `numpy`, `scipy`, `kagglehub`
* **Visualisasi & Audit Kualitas Data:** `matplotlib`, `seaborn`, `missingno`
* **Machine Learning & Interpretabilitas:** `scikit-learn`, `xgboost`, `shap`

---

## 🚀 Cara Menjalankan Notebook

1. **Buka Langsung di Google Colab:**
   Klik lencana **"Open in Colab"** di bagian atas halaman ini, lalu jalankan seluruh sel secara berurutan (`Runtime` > `Run all`).
2. **Jalankan secara Lokal:**
   ```bash
   git clone [https://github.com/RETNO26MA41D019/EDA-Cart-Abandonment.git](https://github.com/RETNO26MA41D019/EDA-Cart-Abandonment.git)
   cd EDA-Cart-Abandonment
   pip install kagglehub missingno xgboost shap scikit-learn seaborn matplotlib pandas numpy scipy
   jupyter notebook "Deteksi (Cart_Abandonment)_.ipynb"
