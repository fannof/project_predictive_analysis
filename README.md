# Laporan Proyek Machine Learning - Novan Nur Hidayat

PENERAPAN ANALISIS PREDIKSI UNTUK HARGA RUMAH (STUDI KASUS: SEATTLE, WASHING, USA)

## Domain Proyek

- Latar belakang

Rumah adalah bangunan fondasi yang dibutuhkan kehidupan seseorang untuk beristirahat. Mengingat hal ini, rumah tempat tinggal juga perlu memiliki nilai yang selain memiliki dimensi, keadaan, dan yang paling penting, aspek ekonomi yang membutuhkan investasi waktu yang signifikan, yaitu untuk memahami cara hidup individu.  Karena harga tanah, rumah, dan fasilitas lainnya umumnya sangat tinggi dan dapat menyebabkan harga naik setiap tahun, kualitas hidup di distrik kota umumnya cukup baik.  Setiap hari harga rumah naik sedikit demi sedikit, Kenaikan harga ini dapat dijelaskan oleh beberapa faktor atau bahkan harga perihal pendukung yang ditawarkan. Harga berfluktuasi dan tidak selalu dapat diprediksi, sehingga pembeli real estate membutuhkan sistem yang dapat memprediksi harga berdasarkan karakteristik unik properti. Karena fakta bahwa harga rumah berfluktuasi setiap tahun, peneliti melakukan penelitian menggunakan analisis regresi dan regresi linier berganda untuk memperkirakan harga rumah, kemudian menggunakan analisis data dan machine learning untuk membuat model yang dimaksudkan untuk memprediksi harga rumah. Analis data memiliki peluang menarik untuk memeriksa dan memperkirakan arah harga properti di pasar real estate seperti di Seattle, Washing, USA. Prediksi harga properti menjadi keterampilan yang lebih berguna dan signifikan. Nilai real estate adalah ukuran yang dapat diandalkan dari kesehatan ekonomi suatu negara serta keadaan pasar secara keseluruhan dan mengatur koleksi catatan penjualan real estate yang cukup besar yang disimpan berdasarkan informasi yang diberikan.

**Rubrik**

- Rumusan Masalah dan Solusi Permasalahan

Rumah bukan hanya sebagai tempat tinggal, tetapi juga investasi yang signifikan bagi banyak orang. Menyesuaikan nilai properti dapat memberikan manfaat finansial yang signifikan bagi pemilik rumah. Karena itu, memiliki pemahaman yang kuat tentang faktor-faktor yang mempengaruhi harga rumah dan kemampuan untuk memprediksi perubahan harga membuat mereka penting ketika membuat keputusan investasi. Faktor-faktor seperti lokasi, ukuran tanah, luas konstruksi, jumlah kamar tidur, dan fasilitas seperti kolam renang penting dalam menentukan harga rumah. Mengenali bagaimana masing-masing faktor ini mempengaruhi harga dapat membantu pembeli, penjual, dan investor membuat keputusan yang lebih tepat dan strategis.

Memanfaatkan kemajuan teknologi, Machine Learning dapat memberikan hasil yang lebih akurat dan konsisten ketika digunakan untuk menganalisis dan memprediksi harga rumah. Ini tidak hanya menguntungkan individu dalam menciptakan keputusan buy-in atau sell-out, tetapi juga dapat membantu lembaga pemerintah dan organisasi terkait dalam mengejar hak properti dan apropriasi.

- Hasil Riset Terkait

Dalam jurnal yang berjudul "Analisis Prediksi Harga Rumah Sesuai Spesifikasi Menggunakan Multiple Linear Regression" yang dipublikasikan oleh Muhammad Labib Mu'tashim, et all(2021) dijelaskan bahwa faktor yang mempengaruhi harga rumah yaitu lokasi, luas tanah, luas bangunan, jumlah kamar (ditujukan untuk pria atau wanita) dan tingkat garasinya.  Untuk menemukan prediksi harga ini, peneliti memerlukan metode yang dapat diterapkan pada data ini, dan metode tersebut dapat diturunkan dari regresi linier berganda. Prediksi dapat diturunkan dari berbagai jenis faktor yang terkait dengan variabel.  Faktor ini perlu dievaluasi sesuai dengan kriteria agar dapat diandalkan saat membuat prediksi. Dalam data uji sampel, digunakan 1001 baris data dan 7 kolom berisi data harga rumah yang tersedia di Jakarta Tenggara. Setelah data dikumpulkan, kemudian dibagi menjadi data pelatihan dan pengujian. Akhirnya, data digunakan untuk menghitung akurasi model Regresi Linier Berganda, yang menghasilkan akurasi 66%. Ini cukup baik untuk memperkirakan harga rumah berdasarkan spesifikasi spesifik yang diperlukan.

## Business Understanding

### Problem Statements

- Sebagai investasi, rumah membutuhkan pengetahuan tentang berbagai faktor yang mempengaruhi nilai properti. Harga yang fluktuatif dapat membuat seseorang enggan membuat keputusan investasi yang bijak. Akibatnya, perlu dipahami bagaimana mengidentifikasi dan menimbang faktor-faktor paling signifikan ketika menentukan nilai rumah.

- Implikasi ekonomi dan sosial dari hak milik harus dipertimbangkan. Berkurangnya akses ke sumber daya untuk masyarakat umum dengan pendapatan rendah atau tanpa pendapatan dapat menjadi masalah serius. Karena itu, solusi diperlukan untuk memahami bagaimana harga properti dapat berfluktuasi tanpa merusak stabilitas ekonomi dan kohesi sosial.

### Goals

- Tujuan utamanya adalah untuk mengembangkan model atau sistem prediktif yang dapat mengidentifikasi dan memberikan dukungan untuk faktor-faktor yang memiliki dampak terbesar pada harga rumah. Tujuan proses klarifikasi ini adalah untuk memberikan lebih banyak wawasan tentang bagaimana berbagai karakteristik properti, seperti lokasi, ukuran, dan fasilitas, mempengaruhi nilai rumah.

- Tujuan yang dimaksudkan adalah untuk mengembangkan kebijakan atau strategi yang dapat mengatasi ketidakseimbangan antara harga properti dan akses ke sumber daya yang tidak stabil. Solusi yang dihasilkan harus dapat memaksimalkan pertumbuhan ekonomi sekaligus memastikan keselamatan masyarakat dengan menyeimbangkan dampak sosial dan ekonomi dari fluktuasi harga properti.

**Rubrik**

### Solution statements

- Menggunakan pendekatan ensemble learning dengan menggabungkan beberapa model machine learning, seperti regresi linier berganda, decision tree, dan random forest. Hal ini bertujuan untuk mengoptimalkan akurasi prediksi dengan memanfaatkan kekuatan masing-masing model. Evaluasi model dapat menggunakan metrik seperti Mean Squared Error (MSE) untuk mengukur seberapa baik model memprediksi harga rumah.

## Data Understanding

Pasar real estate, seperti yang ada di Seattle, Washing, USA, menghadirkan peluang menarik bagi analis data untuk menganalisis dan memprediksi ke mana harga properti bergerak. Prediksi harga properti menjadi semakin penting dan menguntungkan. Harga properti merupakan indikator yang baik dari kondisi pasar secara keseluruhan dan kesehatan ekonomi suatu negara. Mempertimbangkan data yang diberikan, dalam memperdebatkan sejumlah besar catatan penjualan properti yang disimpan dalam format yang tidak diketahui dan dengan masalah kualitas data yang tidak diketahui.
Data yang digunakan dalam proyek ini bersumber dari Kaggle (kaggle datasets download -d samuelcortinhas/house-price-prediction-seattle) 

### Variabel-variabel pada House Price Prediction dataset adalah sebagai berikut:

- beds : jumlah kamar tidur di properti.
- baths : Jumlah kamar mandi di properti. Catatan 0,5 sesuai dengan setengah bak mandi yang memiliki wastafel dan toilet tetapi tidak ada bak mandi atau pancuran.
- size : Total luas lantai properti.
- size_units : Unit pengukuran sebelumnya.
- lot_size : Total luas tanah tempat properti berada. Tanah itu milik pemilik rumah.
- lot_size_units : Unit pengukuran sebelumnya.
- zip_code : Kode pos. Ini adalah kode pos yang digunakan di AS.
- price : Harga properti dijual seharga (dolar AS).

**Rubrik**

### Exploratory Data Analysis

- Menangani missing value
Dari hasil output, terlihat bahwa kolom "lot_size" dan "lot_size_units" memiliki nilai yang hilang (NaN) sebanyak 347 data. Dengan menggunakan teknik dropna, sekarang DataFrame baru (rumah_cleaned_rows) tidak mengandung baris dengan nilai yang hilang di kolom "lot_size" dan "lot_size_units".

- Menangani outliers
Pada kasus ini, akan dideteksi outliers dengan teknik visualisasi data (boxplot). Kemudian, outliers akan ditangani dengan teknik IQR method. setelah ditangani dengan metode IQR method, dataset yang tersisa menjadi 1682 data.

- Univariate analysis
Selanjutnya, akan dilakukan proses analisis data dengan teknik Univariate EDA. Pertama, lakukan analisis pada fitur numerik.
Peningkatan harga rumah sebanding dengan penurunan jumlah sampel. Hal ini dapat dilihat jelas dari histogram "price" yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu x).
Semakin tinggi size, jumlah beds, dan jumlah baths dalam rumah, maka semakin mahal pula harga rumah.

- Multivariate analysis
Selanjutnya, akan dilakukan analisis data pada fitur numerik menggunakan teknik Multivariate EDA menggunakan fungsi pairplot() dan juga akan mengobservasi korelasi antara fitur numerik dengan fitur target menggunakan fungsi corr().
Pada pola sebaran data grafik pairplot, terlihat fitur "size" memiliki korelasi positif dengan fitur "price". Sedangkan kedua fitur "lot_size" dan "price" tidak memliki korelasi karena tidak membetuk pola. 

Pada grafik korelasi terlihat bahwa fitur 'beds', 'baths', dan 'size' memiliki skor korelasi yang besar dengan fitur target 'price'. Artinya, fitur 'price' berkorelasi tinggi dengan ketiga fitur tersebut. Sementara itu, fitur 'lot_size' dan 'zip_code' memiliki korelasi yang sangat kecil sehingga fitur tersebut dapat di-drop.

## Data Preparation

- Train-Test-Split
Proses membagi himpunan data menjadi data pelatihan dan pengujian adalah langkah yang diperlukan sebelum membuat model. Penting untuk memperkuat semua data yang tersedia untuk menilai beberapa generalisasi model ke data baru. Perlu dicatat bahwa setiap transformasi data yang dilakukan juga berfungsi sebagai komponen model. Karena data test set (uji) mentah, semua transformasi harus dilakukan pada data latih.

- Standarisasi
Ketika algoritma pembelajaran mesin diterapkan pada data dengan distribusi yang serupa atau menyimpang, mereka berkinerja lebih baik dan menyatu lebih cepat. Proses penskalaan dan standardisasi membantu mengubah data menjadi format yang lebih mudah dipahami oleh algoritma. 

Standardisasi adalah teknik transformasi yang paling umum digunakan dalam proses pembangunan model. Ini tidak akan mengubah fitur numerik menggunakan encoding. Teknik yang digunakan adalah StandarScaler dari library Scikit-learn.

### Penjelasan tahapan dan kenapa harus dilakukan proses tersebut

- Proses data prepraration
Pertama adalah proses train-test-split. Data dibagi menjadi 80% data training dan 20% data testing, karena jumlah seluruh data termasuk kecil, maka diperlukan lebih banyak data latih.

Proses standarisasi mengubah nilai mean menjadi 0 dan std menjadi 1. StandardScaler melakukan proses standardisasi parameter fitur terlebih dahulu dengan mengurangkan nilai mean (nilai rata-rata) dan kemudian membandingkannya dengan standar deviasi untuk menentukan distribusi.  StandardScaler menghasilkan distribusi dengan rata-rata 0 dan standar deviasi 1.

Tahapan diatas penting dilakukan karena algoritma machine learning memiliki performa lebih baik ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal.

## Modeling

Model akan dikembangkan dengan 3 algoritma yang berbeda, dan mencari mana yang memiliki performa paling baik. Beberapa algoritma tersebut adalah sebagai berikut:

1. k-NN
Langkah yang pertama, model KNN diinisialisasi dengan menentukan jumlah tetangga terdekat (parameter n_neighbors). Contoh dalam kasus ini adalah n_neighbors diatur ke 10, artinya model akan menggunakan 10 tetangga tetangga yang paling dekat untuk membuat prediksi.
Setelah model diinisialisasi, langkah selanjutnya adalah melatih model menggunakan data latih. Untuk melatih model dengan fitur X_train dan target y_train, gunakan fungsi fit(X_train, y_train).
Setelah proses pelatihan selesai, model sudah dapat membuat prediksi pada data latih untuk mengevaluasi performa model. 
Parameter yang Digunakan pada Model KNN:
n_neighbors: Jumlah tetangga terdekat yang digunakan untuk membuat prediksi.

2. Random Forest
Model Random Forest diinisialisasi dengan menentukan beberapa hyperparameter. 
Model Random Forest dilatih menggunakan data latih (X_train dan y_train). Fungsi fit(X_train, y_train) digunakan untuk melatih model.
Setelah pelatihan selesai, model sekarang dapat digunakan untuk membuat prediksi pada data latih. RF.predict(X_train) menghasilkan prediksi target berdasarkan fitur pada data latih.
Parameter yang Digunakan pada Model Random Forest:
n_estimators: Jumlah pohon keputusan dalam ensemble.
max_depth: Kedalaman maksimum setiap pohon keputusan.
random_state: Digunakan untuk memastikan hasil yang reproduktif.
n_jobs: Jumlah pekerjaan paralel yang akan dijalankan.

3. Boosting Algorithm
Model Boosting (AdaBoostRegressor) diinisialisasi dengan menentukan hyperparameter tertentu. Parameter yang diatur adalah learning_rate dengan nilai 0.05. random_state digunakan untuk memastikan reproduktibilitas hasil.
Model diarahkan untuk mempelajari hubungan antara fitur (X_train) dan target (y_train). Fungsi fit(X_train, y_train) digunakan untuk melatih model dengan data latih.
Setelah pelatihan selesai, sekarang model dapat digunakan untuk membuat prediksi pada data latih. boosting.predict(X_train) menghasilkan prediksi target berdasarkan fitur pada data latih.
Parameter yang Digunakan pada Model Boosting (AdaBoostRegressor):
learning_rate: Menentukan sejauh mana model belajar dari kesalahan sebelumnya. Nilai yang lebih kecil akan memperbaiki konvergensi, tetapi memerlukan jumlah estimator (pohon keputusan) yang lebih besar.
n_estimators: Jumlah estimator (pohon keputusan) yang digunakan.
base_estimator: Tipe model dasar yang digunakan. Secara default, digunakan pohon keputusan (DecisionTreeRegressor).
random_state: Digunakan untuk memastikan hasil yang reproduktif.

**Rubrik**

- Kelebihan dan kekurangan dari setiap algoritma yang digunakan:
--KNN memiliki MSE yang tinggi pada data uji, mungkin karena model tidak dapat menangkap pola yang kompleks dalam data tersebut.
--Random Forest memiliki MSE yang rendah pada data latih, menunjukkan kemampuan baik dalam menyesuaikan dengan data latih. Namun, terdapat peningkatan yang signifikan pada MSE pada data uji, mungkin menunjukkan adanya overfitting.
--Boosting memberikan hasil yang cukup baik pada data uji, menunjukkan kemampuan model untuk mengatasi kompleksitas data.

- Alasan memlilih Model Boosting Algorithm
Dari ketiga model diatas, Boosting memiliki MSE yang relatif lebih rendah pada data uji, menunjukkan kinerja yang lebih baik dibandingkan dengan KNN dan Random Forest dalam dataset harga rumah ini.
Berdasarkan hasil tersebut, solusi terbaik pada kasus ini adalah menggunakan model Boosting (AdaBoostRegressor) karena memberikan performa yang lebih baik dalam memprediksi harga rumah pada data yang belum pernah dilihat sebelumnya.

## Evaluation

- Metrik yang digunakan adalah MSE
Mean Squared Error (MSE) adalah salah satu metrik evaluasi yang umum digunakan dalam regresi untuk mengukur sejauh mana perbedaan antara nilai prediksi model dengan nilai aktual (ground truth). MSE dihitung dengan menjumlahkan kuadrat selisih antara setiap nilai prediksi dan nilai aktual, kemudian diambil rata-rata dari seluruh data. Nilai MSE semakin kecil semakin baik. Nilai MSE sama dengan nol berarti model memberikan prediksi yang sempurna sesuai dengan nilai aktual.

- Hasil proyek berdasarkan metrik evaluasi
1. K-Nearest Neighbors (KNN):
Data Latih (Train): MSE sekitar 44,360,753.27.
Data Uji (Test): MSE sekitar 54,239,740.50.
Interpretasi: Model KNN memiliki hasil yang lebih baik pada data latih dibandingkan dengan data uji, mungkin menunjukkan adanya overfitting atau kurangnya kemampuan untuk menggeneralisasi pada data yang belum pernah dilihat sebelumnya.

2. Random Forest (RF):
Data Latih (Train): MSE sekitar 12,390,218.33.
Data Uji (Test): MSE sekitar 58,255,407.25.
Interpretasi: Model Random Forest menunjukkan performa yang baik pada data latih, tetapi terdapat peningkatan yang signifikan pada MSE pada data uji, mungkin mengindikasikan adanya overfitting.

3. Boosting:
Data Latih (Train): MSE sekitar 49,336,053.97.
Data Uji (Test): MSE sekitar 51,079,691.29.
Interpretasi: Model Boosting menunjukkan hasil yang relatif baik pada data uji dibandingkan dengan model KNN dan Random Forest. Meskipun MSE pada data latih lebih tinggi dari Random Forest, kemampuan generalisasi pada data uji tampaknya lebih baik.

**Rubrik**

- Formula metrik dan cara kerja
Formula = MSE = n/1 ∑ n i=1 (y ​−y^i)^2
Dengan n adalah jumlah sampel
yi adalah nilai aktual
y^i adalah nilai prediksi

Cara Kerja:
MSE mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual.
Prosesnya melibatkan langkah-langkah berikut:
Menghitung selisih antara setiap nilai prediksi dan nilai aktual.
Mengkuadratkan setiap selisih.
Menjumlahkan semua nilai kuadrat.
Membagi jumlah nilai kuadrat dengan jumlah observasi (n) untuk mendapatkan rata-rata.

Interpretasi:
Semakin kecil nilai MSE, semakin baik model memperkirakan nilai aktual.
Kesalahan yang lebih besar akan memberikan kontribusi yang lebih besar terhadap nilai MSE karena nilai diangkat ke kuadrat.

---Ini adalah bagian akhir laporan---
