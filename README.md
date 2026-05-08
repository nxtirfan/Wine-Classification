# Wine Quality Classification with Feature Engineering and Gradient Boosting

Proyek ini membahas klasifikasi multikelas kualitas wine menggunakan dataset fisikokimia wine merah.

Pipeline mencakup:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Scaling
- Penanganan Imbalanced Data dengan SMOTE
- Perbandingan Model Machine Learning
- Evaluasi menggunakan Out-of-Fold Prediction
- Feature Importance Analysis

## Dataset

Dataset terdiri dari fitur fisikokimia wine, seperti:

- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- sulfur dioxide
- density
- pH
- sulphates
- alcohol

Target yang diprediksi adalah:

- `quality` (klasifikasi multikelas)

## Metodologi

### 1. Exploratory Data Analysis

EDA dilakukan untuk:

- memeriksa missing value,
- melihat distribusi kelas,
- menganalisis korelasi fitur,
- dan mendeteksi potensi outlier.

### 2. Feature Engineering

Tiga fitur rasio ditambahkan:

- `acidity_ratio`
- `sulfur_ratio`
- `alcohol_density_ratio`

Fitur dibuat untuk membantu model menangkap hubungan relatif antar komponen kimia wine.

### 3. Feature Scaling

`StandardScaler` digunakan sebelum SMOTE karena SMOTE berbasis jarak Euclidean.

### 4. Penanganan Imbalanced Data

SMOTE diterapkan di dalam pipeline cross validation untuk mencegah data leakage.

### 5. Model yang Dibandingkan

- Random Forest
- Random Forest + SMOTE
- Gradient Boosting

### 6. Evaluasi

Evaluasi dilakukan menggunakan:

- Cross Validation Accuracy
- Out-of-Fold Accuracy
- F1 Macro
- Classification Report
- Confusion Matrix

## Hasil

Gradient Boosting menghasilkan performa terbaik dibanding model lainnya.

Model mampu menangani overlap antar kelas dengan lebih baik dibanding pendekatan Random Forest berbasis bagging.

## Struktur Notebook

1. Pendahuluan
2. Inisialisasi dan Pemuatan Data
3. Exploratory Data Analysis
4. Persiapan Data
5. Feature Engineering
6. Feature Scaling
7. Penanganan Imbalanced Data dengan SMOTE
8. Perbandingan dan Pemilihan Model
9. Evaluasi Model dengan OOF Prediction
10. Prediksi Data Uji dan File Submission
11. Feature Importance
12. Kesimpulan

## Library Utama

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn

## Catatan

Pipeline dirancang untuk:

- menghindari data leakage,
- menjaga validitas evaluasi,
- dan menghasilkan proses klasifikasi yang lebih realistis pada dataset imbalanced multiclass.
