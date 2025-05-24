# Laporan Proyek Machine Learning - Ade Ripaldi Nuralim

## Domain Proyek

![image](https://raw.githubusercontent.com/revaile/wine/refs/heads/main/assets/1.png)


Proyek ini berada dalam domain agroindustri dan industri minuman beralkohol, khususnya pada proses produksi wine (anggur). Industri wine secara global merupakan salah satu industri bernilai tinggi yang sangat bergantung pada kualitas produk. Kualitas wine dipengaruhi oleh berbagai faktor kimia dan fisik yang terjadi selama proses fermentasi, penyimpanan, dan produksi. Namun, pengujian kualitas wine umumnya dilakukan oleh para ahli wine (wine tasters) secara sensorik, yang bersifat subjektif, memakan waktu, dan mahal.

Dalam beberapa tahun terakhir, pendekatan data-driven seperti Machine Learning mulai digunakan untuk mempercepat dan mengotomatiskan proses penilaian kualitas wine. Dengan memanfaatkan data kimia wine seperti kadar alkohol, keasaman, kandungan sulfur, dan pH, model prediktif dapat dibuat untuk mengklasifikasikan wine ke dalam kategori kualitas tertentu (baik atau buruk). Model ini membantu industri mengurangi biaya uji sensorik, mempercepat siklus produksi, serta menjaga konsistensi mutu produk.

Menurut Cortez et al. (2009), dalam jurnal "Modeling wine preferences by data mining from physicochemical properties," pendekatan Machine Learning dapat digunakan untuk memodelkan hubungan antara parameter kimia wine dan penilaian kualitas oleh para ahli. Hasil studi tersebut membuka peluang besar bagi industri wine untuk mengadopsi metode otomasi berbasis data mining dan pembelajaran mesin guna meningkatkan efisiensi produksi.

Referensi:

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547-553. https://doi.org/10.1016/j.dss.2009.05.016

UCI Machine Learning Repository – Wine Quality Data Set. https://archive.ics.uci.edu/ml/datasets/wine+quality

## Business Understanding

### Problem Statements:
1. Bagaimana cara memprediksi kualitas wine (baik atau buruk) berdasarkan fitur kimia? Kualitas wine seringkali ditentukan oleh panelis ahli, yang merupakan proses memakan waktu dan mahal. Membangun model prediktif dapat mengotomatisasi proses ini.
  
3. Model Machine Learning apa yang memberikan performa terbaik dalam memprediksi kualitas wine? Terdapat berbagai algoritma klasifikasi yang dapat digunakan, dan perlu diidentifikasi mana yang memberikan performa paling optimal untuk masalah prediksi kualitas wine ini.

### Goals:
1. Membuat model Machine Learning yang dapat memprediksi kualitas wine (good/bad). Model ini diharapkan dapat mengklasifikasikan wine menjadi dua kategori kualitas: "good" (kualitas baik) dan "bad" (kualitas buruk).
   
3. Mengidentifikasi model Machine Learning terbaik (dengan metrik evaluasi yang sesuai) untuk prediksi kualitas wine. Kami akan membandingkan beberapa model untuk menemukan yang paling efektif.

### Solution Statement:
Untuk mencapai tujuan di atas, proyek ini akan mengajukan beberapa pendekatan solusi:

1. **Solusi 1**: Menggunakan dua atau lebih algoritma klasifikasi: Di sini saya akan menggunakan tiga algoritma klasifikasi yang berbeda: Logistic Regression, K-Nearest Neighbors (KNN), dan Random Forest. Perbandingan performa di antara model-model ini akan membantu mengidentifikasi model terbaik untuk masalah ini.
  
2. **Solusi 2**: Melakukan improvement pada model baseline dengan hyperparameter tuning: Setelah mendapatkan model baseline dari setiap algoritma, kami akan melakukan hyperparameter tuning untuk mengoptimalkan performa model terbaik. Metrik evaluasi seperti akurasi, presisi, recall, dan F1-Score akan digunakan untuk mengukur peningkatan performa dan memilih model terbaik.

## Data Understanding

## Informasi Dataset Wine Quality

| Jenis | Keterangan |
| ------ | ------ |
| Title | _Wine Quality Dataset_ |
| Source | [Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data) |
| Maintainer | [M Yasser H](https://www.kaggle.com/yasserh) |
| License | Other (specified in description) |
| Visibility | Publik |
| Tags | _Earth and Nature, Food, Tabular, Beginer, Classification, Alcohol_ |
| Usability | 10.00 |

## Informasi Dataset Wine Quality

| Id | Fixed Acidity | Volatile Acidity | Citric Acid | Residual Sugar | Chlorides | Free Sulfur Dioxide | Total Sulfur Dioxide | Density | pH  | Sulphates | Alcohol | Quality |
|----|----------------|------------------|-------------|----------------|-----------|----------------------|-----------------------|---------|-----|------------|---------|---------|
| 0  | 7.4            | 0.70             | 0.00        | 1.9            | 0.076     | 11.0                 | 34.0                  | 0.9978  | 3.51| 0.56       | 9.4     | 5       |
| 1  | 7.8            | 0.88             | 0.00        | 2.6            | 0.098     | 25.0                 | 67.0                  | 0.9968  | 3.20| 0.68       | 9.8     | 5       |
| 2  | 7.8            | 0.76             | 0.04        | 2.3            | 0.092     | 15.0                 | 54.0                  | 0.9970  | 3.26| 0.65       | 9.8     | 5       |
| 3  | 11.2           | 0.28             | 0.56        | 1.9            | 0.075     | 17.0                 | 60.0                  | 0.9980  | 3.16| 0.58       | 9.8     | 6       |
| 4  | 7.4            | 0.70             | 0.00        | 1.9            | 0.076     | 11.0                 | 34.0                  | 0.9978  | 3.51| 0.56       | 9.4     | 5       |



Tabel 1. EDA Deskripsi Variabel

Dilihat dari _Tabel 1. EDA Deskripsi Variabel_ dataset ini telah di *bersihkan* dan *normalisasi* terlebih dahulu oleh pembuat, sehingga mudah digunakan dan ramah bagi pemula. 
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 4001 sample dengan 9 fitur.
- Dataset memiliki 7 fitur bertipe float64 dan 2 fitur bertipe object.
- Terdapat 1 missing value dalam dataset.
### Variable - variable pada dataset
- `A_id` : Identifikasi unik untuk setiap buah.
- `Size` : Ukuran buah.
- `Weight` : Berat buah.
- `Sweetness` : Tingkat kemanisan buah.
- `Crunchiness` : Tekstur yang menunjukkan kerenyahan buah.
- `Juiciness` : Tingkat kesegaran buah.
- `Ripeness` : Tahap kematangan buah.
- `Acidity` : Tingkat keasaman buah.
- `Quality` : Kualitas buah secara keseluruhan, baik atau buruk.

Dari ke 9 fitur dapat dilihat bahwa fitur `A_id` tidak mempengaruhi kualitas buah hingga akan di hapus.

Data yang digunakan dalam proyek ini berasal dari dataset "Wine Quality" yang tersedia di Kaggle (https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009).
Dataset ini berisi fitur-fitur fisikokimia dari wine dan juga rating kualitas yang diberikan oleh panelis ahli. Dataset ini telah dimodifikasi untuk tujuan klasifikasi, dimana rating kualitas diubah menjadi dua kategori: "good" (kualitas baik) dan "bad" (kualitas buruk).

Variabel-variabel pada dataset kualitas wine adalah sebagai berikut:


| Nama Fitur            | Deskripsi                                                                 |
|-----------------------|---------------------------------------------------------------------------|
| fixed acidity         | Jumlah asam tartarat dalam wine.                                          |
| volatile acidity      | Jumlah asam asetat, dapat menyebabkan rasa tidak enak.                    |
| citric acid           | Jumlah asam sitrat, menambah kesegaran pada wine.                         |
| residual sugar        | Jumlah gula sisa setelah proses fermentasi selesai.                       |
| chlorides             | Jumlah garam klorida dalam wine.                                          |
| free sulfur dioxide   | Jumlah SO₂ bebas, berfungsi mencegah pertumbuhan mikroba.                 |
| total sulfur dioxide  | Jumlah total SO₂ (termasuk bentuk bebas dan terikat).                     |
| density               | Kepadatan wine, berhubungan dengan kandungan alkohol dan gula.            |
| pH                    | Tingkat keasaman atau kebasaan wine.                                      |
| sulphates             | Jumlah kalium sulfat, aditif yang dapat memengaruhi kandungan SO₂.        |
| alcohol               | Kandungan alkohol dalam wine (dalam % vol).                               |
| quality               | Rating kualitas wine (target variabel). Telah dikategorikan menjadi:      |
|                       | - `good` : jika kualitas >= 7                                             |
|                       | - `bad`  : jika kualitas < 7                                              |

### Exploratory Data Analysis:

Dari eksplorasi data awal, dapat dilihat bahwa:

- Dataset memiliki 1143 baris dan 13 kolom.
- Tidak ada nilai missing pada semua kolom, yang berarti data sudah cukup bersih.
- Sebaran kelas pada variabel quality menunjukkan adanya ketidakseimbangan kelas. Mayoritas wine diklasifikasikan sebagai "bad" (1382 sampel), sedangkan hanya sebagian kecil yang "good" (217 sampel). Hal ini perlu ditangani pada tahap data preparation untuk menghindari bias pada model.

### Analisis Distribusi Fitur 

![image](https://raw.githubusercontent.com/revaile/wine/refs/heads/main/assets/unvariate.png)

Distribusi: fixed acidity

- Bentuk: Distribusi ini tampak sedikit miring ke kanan (positively skewed), menunjukkan bahwa sebagian besar nilai keasaman tetap terkonsentrasi pada sisi kiri (nilai lebih rendah) dari rentangnya.
- Pemusatan: Puncak distribusi berada di sekitar 7-8, menunjukkan bahwa keasaman tetap yang paling sering ditemui adalah di rentang tersebut.
- Rentang: Nilai fixed acidity bervariasi dari sekitar 4.5 hingga 16.

Distribusi: volatile acidity

- Bentuk: Sangat miring ke kanan (positively skewed). Ini menunjukkan bahwa sebagian besar anggur memiliki tingkat keasaman volatil yang rendah, dengan hanya sedikit yang memiliki tingkat sangat tinggi.
- Pemusatan: Puncak distribusi berada di sekitar 0.3-0.4.
- Rentang: Nilai volatile acidity berkisar dari sekitar 0.1 hingga 1.6.

Distribusi: citric acid

- Bentuk: Distribusi ini cenderung miring ke kanan, dengan konsentrasi massa probabilitas di dekat 0. Namun, ada puncak sekunder di sekitar 0.3-0.4, menunjukkan mungkin ada dua kelompok data atau bahwa data tersebar luas di awal.
- Pemusatan: Puncak utama di dekat 0.
- Rentang: Nilai citric acid bervariasi dari 0 hingga sekitar 1.0.

Distribusi: residual sugar

- Bentuk: Sangat miring ke kanan (positively skewed). Mayoritas anggur memiliki kandungan gula sisa yang rendah.
- Pemusatan: Puncak distribusi tajam di sekitar 2, menunjukkan konsentrasi tinggi pada nilai gula sisa yang rendah.
- Rentang: Nilai residual sugar berkisar dari sekitar 0.5 hingga lebih dari 15.

Distribusi: chlorides

- Bentuk: Sangat miring ke kanan (positively skewed), dengan konsentrasi tinggi pada nilai klorida yang sangat rendah.
- Pemusatan: Puncak distribusi berada di sekitar 0.05-0.07.
- Rentang: Nilai chlorides berkisar dari sekitar 0.01 hingga 0.6.

Distribusi: free sulfur dioxide

- Bentuk: Miring ke kanan, meskipun tidak sekuat beberapa distribusi sebelumnya.
- Pemusatan: Puncak distribusi berada di sekitar 6-10.
- Rentang: Nilai free sulfur dioxide berkisar dari 0 hingga sekitar 70.

Distribusi: total sulfur dioxide

- Bentuk: Miring ke kanan. Konsentrasi data lebih banyak pada nilai total sulfur dioksida yang lebih rendah.
- Pemusatan: Puncak distribusi berada di sekitar 20-40.
- Rentang: Nilai total sulfur dioxide bervariasi dari sekitar 0 hingga lebih dari 300.

Distribusi: density

- Bentuk: Cukup simetris dan tampak mendekati distribusi normal, atau sedikit miring ke kanan. Ini menunjukkan bahwa densitas anggur cenderung mengelompok di sekitar nilai rata-rata.
- Pemusatan: Puncak distribusi berada di sekitar 0.996 - 0.997.
- Rentang: Nilai density memiliki rentang yang sangat sempit, dari sekitar 0.990 hingga 1.004.

Distribusi: pH

- Bentuk: Mirip dengan distribusi normal, sedikit miring ke kanan atau hampir simetris.
- Pemusatan: Puncak distribusi berada di sekitar 3.2-3.4.
- Rentang: Nilai pH bervariasi dari sekitar 2.8 hingga 4.0.

Distribusi: sulphates

- Bentuk: Miring ke kanan (positively skewed). Mayoritas anggur memiliki tingkat sulfat yang rendah.
- Pemusatan: Puncak distribusi berada di sekitar 0.5-0.6.
- Rentang: Nilai sulphates berkisar dari sekitar 0.25 hingga 2.0.

Distribusi: alcohol

- Bentuk: Miring ke kanan, dengan konsentrasi data lebih banyak pada kadar alkohol yang lebih rendah.
- Pemusatan: Puncak distribusi berada di sekitar 9-10.
- Rentang: Nilai alcohol bervariasi dari sekitar 8.5 hingga 15.

### Analisis Distribusi Kualitas Wine

![image](https://raw.githubusercontent.com/revaile/wine/refs/heads/main/assets/distribusi.png)

Berdasarkan bar chart "Distribusi Kualitas Wine", terlihat jelas bahwa mayoritas sampel wine terkonsentrasi pada kualitas 5 dan 6, dengan jumlah sampel yang sangat dominan mendekati 500 untuk kualitas 5 dan sedikit di bawahnya untuk kualitas 6. Sementara itu, kualitas 7 memiliki jumlah sampel yang lebih sedikit, dan kualitas ekstrem seperti 3, 4, dan 8 hanya diwakili oleh sangat sedikit sampel, mengindikasikan ketidakseimbangan distribusi kelas yang signifikan dalam dataset ini.

### Matriks Korelasi

![image](https://raw.githubusercontent.com/revaile/wine/refs/heads/main/assets/matrix.png)

Matriks Korelasi menunjukkan hubungan antar fitur dan quality wine. Alcohol (0.48) dan sulphates (0.26) berkorelasi positif dengan kualitas. Sebaliknya, volatile acidity (-0.41), chlorides (-0.12), total sulfur dioxide (-0.18), dan density (-0.18) berkorelasi negatif. Fitur lainnya memiliki korelasi lemah atau tidak signifikan dengan quality.

## Data Preparation

Tahap ini bertujuan untuk mempersiapkan data sebelum digunakan dalam pemodelan Machine Learning. Langkah-langkah yang dilakukan meliputi:

### 1. Pembagian Fitur dan Target
- Variabel `quality` dipisahkan sebagai variabel target (`y`).
- Kolom lainnya digunakan sebagai fitur (`X`).

### 2. Encoding Variabel Target
- ubah data ke numerik

### 3. Penanganan Imbalance Data (SMOTE)
- Karena distribusi kelas target tidak seimbang (lebih banyak "bad" daripada "good"), digunakan teknik oversampling SMOTE (Synthetic Minority Over-sampling Technique).
- SMOTE menghasilkan data sintetis dari kelas minoritas untuk menyeimbangkan dataset.

### 4. Pembagian Data Latih dan Uji
- Dataset dibagi menggunakan `train_test_split`:
  - 80% untuk data latih
  - 20% untuk data uji
- Pembagian menggunakan `stratify` untuk menjaga proporsi kelas tetap seimbang di data latih dan uji.

### 5. Standardisasi Fitur
- Semua fitur numerik dinormalisasi menggunakan `StandardScaler`.
- Standardisasi ini membuat fitur memiliki rata-rata 0 dan standar deviasi 1.
- Tujuan: agar model yang sensitif terhadap skala (seperti KNN dan Logistic Regression) bekerja lebih baik.

## Modeling

Pada tahap pemodelan, kami akan melatih tiga model klasifikasi yang berbeda: Logistic Regression, K-Nearest Neighbors (KNN), dan Random Forest. Masing-masing model ini memiliki karakteristik dan asumsi yang berbeda, sehingga perbandingan performanya akan memberikan wawasan tentang model terbaik untuk masalah ini.

1. Logistic Regression
- Penjelasan: Logistic Regression adalah algoritma klasifikasi linier yang digunakan untuk memprediksi probabilitas sebuah instance termasuk dalam kelas tertentu. Meskipun namanya mengandung "regresi", ini adalah model klasifikasi. Model ini bekerja dengan mencocokkan data ke fungsi logistik (sigmoid) yang menghasilkan probabilitas antara 0 dan 1.
- Parameter yang digunakan:
random_state=42: Digunakan untuk memastikan hasil yang reproduktif.
Kelebihan: Cepat, mudah diinterpretasikan, dan performa baik untuk dataset yang terpisah secara linier.
- Kekurangan: Asumsi linieritas, mungkin tidak berkinerja baik pada dataset yang sangat kompleks atau non-linier.
  
2. K-Nearest Neighbors (KNN)
- Penjelasan: KNN adalah algoritma berbasis instance non-parametrik yang mengklasifikasikan titik data baru berdasarkan mayoritas kelas dari 'k' tetangga terdekatnya di ruang fitur.
- Parameter yang digunakan:
n_neighbors=5: Jumlah tetangga terdekat yang dipertimbangkan untuk klasifikasi.
- Kelebihan: Sederhana, tidak memerlukan asumsi distribusi data, efektif untuk data yang kecil.
Kekurangan: Komputasi mahal untuk dataset besar (karena perlu menghitung jarak ke semua titik data), sensitif terhadap data pencilan, dan performa menurun dengan dimensi fitur yang tinggi.

3. Random Forest
- Penjelasan: Random Forest adalah algoritma ensemble yang membangun banyak pohon keputusan selama pelatihan dan mengeluarkan kelas yang merupakan mode dari kelas-kelas (klasifikasi) atau rata-rata prediksi (regresi) dari pohon-pohon individu. Ini mengurangi overfitting yang sering terjadi pada pohon keputusan tunggal.
- Parameter yang digunakan:
n_estimators=100: Jumlah pohon dalam hutan.
random_state=42: Digunakan untuk memastikan hasil yang reproduktif.
- Kelebihan: Sangat akurat, dapat menangani fitur non-linier dan interaksi antar fitur, tidak rentan terhadap overfitting dibandingkan pohon keputusan tunggal.
Kekurangan: Kurang mudah diinterpretasikan dibandingkan pohon keputusan tunggal atau Logistic Regression, membutuhkan lebih banyak sumber daya komputasi.

Pemilihan Model Terbaik

Dari hasil akurasi awal, dapat dilihat bahwa Random Forest menunjukkan performa akurasi tertinggi dibandingkan Logistic Regression dan KNN. Oleh karena itu, Random Forest akan dipilih sebagai model terbaik dan akan menjadi fokus untuk proses hyperparameter tuning selanjutnya.

Improvement Model (Hyperparameter Tuning dengan GridSearchCV)

Untuk meningkatkan performa model Random Forest, kami akan melakukan hyperparameter tuning menggunakan GridSearchCV. Metode ini secara sistematis mencoba semua kombinasi hyperparameter yang ditentukan untuk menemukan kombinasi terbaik yang menghasilkan performa model tertinggi.

Parameter yang akan di-tuning:

- n_estimators: Jumlah pohon dalam hutan.
- max_features: Jumlah fitur yang dipertimbangkan saat mencari pemisahan terbaik.
- min_samples_leaf: Jumlah minimum sampel yang diperlukan untuk menjadi simpul daun.
- min_samples_split: Jumlah minimum sampel yang diperlukan untuk membagi sebuah simpul.
- Penjelasan proses improvement:
- GridSearchCV akan melatih model Random Forest berulang kali dengan berbagai kombinasi hyperparameter yang ditentukan dalam param_grid. Setiap kombinasi akan dievaluasi menggunakan validasi silang (cross-validation) untuk memastikan robustnya performa. Model dengan kombinasi hyperparameter terbaik berdasarkan metrik evaluasi (dalam kasus ini, akurasi) akan dipilih sebagai model final.

## Evaluasi

Dalam tahap evaluasi, metrik yang digunakan adalah `accuracy`
Accuracy didapatkan dengan menghitung persentase dari jumlah prediksi yang benar dibagi dengan jumlah seluruh prediksi. Rumus:

$$\text{Accuracy} = \frac{\text{TP + TN}}{\text{TN + TP + FN + FP}} \times 100\%$$

*Penjelasan*
- TP (True Positive): Jumlah data positif yang diprediksi dengan benar sebagai positif.
- TN (True Negative): Jumlah data negatif yang diprediksi dengan benar sebagai negatif.
- FP (False Positive): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (Kesalahan Tipe I).
- FN (False Negative): Jumlah data positif yang diprediksi secara tidak benar sebagai negatif (Kesalahan Tipe II).

Rumus ini memecah akurasi menjadi rasio antara data yang diklasifikasikan dengan benar (TP dan TN) dengan jumlah total data. Mengalikan dengan 100% mengubah rasio menjadi persentase.

Berikut hasil accuracy 5 buah model yang latih:

| Model | Accuracy |
| ------ | ------ | 
| RandomForest  | 71.18% |
| Logistic Regression | 64.63% |
| KNN | 57.21% |


Tabel 3. Hasil Accuracy

![Plot Accuracy](https://raw.githubusercontent.com/revaile/wine/refs/heads/main/assets/eval.png)

Dilihat dari Tabel Hasil Accuracy dan Gambar Visualisasi Accuracy Model, dapat diketahui bahwa model dengan algoritma Random Forest memiliki nilai akurasi tertinggi yaitu 71.18%. Oleh karena itu, model ini dipilih sebagai model terbaik untuk digunakan dalam memprediksi kualitas Wine.

Model Random Forest dipilih karena memiliki performa yang paling unggul dibandingkan model lainnya seperti Logistic Regression dan K-Nearest Neighbors (KNN). Selain memberikan akurasi yang lebih tinggi, Random Forest juga dikenal tangguh terhadap overfitting dan mampu menangani data dengan dimensi yang kompleks secara efektif.
