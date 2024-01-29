# Laporan Proyek Machine Learning - Novan Nur Hidayat

# PENERAPAN ANALISIS PREDIKSI UNTUK HARGA RUMAH (STUDI KASUS: SEATTLE, WASHING, USA)

## Domain Proyek

- Latar belakang

  Rumah adalah bangunan fondasi yang dibutuhkan kehidupan seseorang untuk beristirahat. Mengingat hal ini, rumah tempat tinggal juga perlu memiliki nilai yang selain memiliki dimensi, keadaan, dan yang paling penting, aspek ekonomi yang membutuhkan investasi waktu yang signifikan, yaitu untuk memahami cara hidup individu [1].  Karena harga tanah, rumah, dan fasilitas lainnya umumnya sangat tinggi dan dapat menyebabkan harga naik setiap tahun, kualitas hidup di distrik kota umumnya cukup baik.  Setiap hari harga rumah naik sedikit demi sedikit, Kenaikan harga ini dapat dijelaskan oleh beberapa faktor atau bahkan harga perihal pendukung yang ditawarkan. Harga berfluktuasi dan tidak selalu dapat diprediksi, sehingga pembeli real estate membutuhkan sistem yang dapat memprediksi harga berdasarkan karakteristik unik properti [2]. Karena fakta bahwa harga rumah berfluktuasi setiap tahun, peneliti melakukan penelitian menggunakan analisis regresi dan regresi linier berganda untuk memperkirakan harga rumah, kemudian menggunakan analisis data dan machine learning untuk membuat model yang dimaksudkan untuk memprediksi harga rumah. Analis data memiliki peluang menarik untuk memeriksa dan memperkirakan arah harga properti di pasar real estate seperti di Seattle, Washing, USA. Prediksi harga properti menjadi keterampilan yang lebih berguna dan signifikan. Nilai real estate adalah ukuran yang dapat diandalkan dari kesehatan ekonomi suatu negara serta keadaan pasar secara keseluruhan dan mengatur koleksi catatan penjualan real estate yang cukup besar yang disimpan berdasarkan informasi yang diberikan.

- Rumusan Masalah dan Solusi Permasalahan

  Rumah bukan hanya sebagai tempat tinggal, tetapi juga investasi yang signifikan bagi banyak orang. Menyesuaikan nilai properti dapat memberikan manfaat finansial yang signifikan bagi pemilik rumah. Karena itu, memiliki pemahaman yang kuat tentang faktor-faktor yang mempengaruhi harga rumah dan kemampuan untuk memprediksi perubahan harga membuat mereka penting ketika membuat keputusan investasi. Faktor-faktor seperti lokasi, ukuran tanah, luas konstruksi, jumlah kamar tidur, dan fasilitas seperti kolam renang penting dalam menentukan harga rumah. Mengenali bagaimana masing-masing faktor ini mempengaruhi harga dapat membantu pembeli, penjual, dan investor membuat keputusan yang lebih tepat dan strategis.

  Memanfaatkan kemajuan teknologi, Machine Learning dapat memberikan hasil yang lebih akurat dan konsisten ketika digunakan untuk menganalisis dan memprediksi harga rumah. Ini tidak hanya menguntungkan individu dalam menciptakan keputusan buy-in atau sell-out, tetapi juga dapat membantu lembaga pemerintah dan organisasi terkait dalam mengejar hak properti dan apropriasi.

- Hasil Riset Terkait

  Dalam jurnal yang berjudul "Analisis Prediksi Harga Rumah Sesuai Spesifikasi Menggunakan Multiple Linear Regression" yang dipublikasikan oleh Muhammad Labib Mu'tashim, et all(2021) dijelaskan bahwa faktor yang mempengaruhi harga rumah yaitu lokasi, luas tanah, luas bangunan, jumlah kamar (ditujukan untuk pria atau wanita) dan tingkat garasinya.  Untuk menemukan prediksi harga ini, peneliti memerlukan metode yang dapat diterapkan pada data ini, dan metode tersebut dapat diturunkan dari regresi linier berganda. Prediksi dapat diturunkan dari berbagai jenis faktor yang terkait dengan variabel.  Faktor ini perlu dievaluasi sesuai dengan kriteria agar dapat diandalkan saat membuat prediksi. Dalam data uji sampel, digunakan 1001 baris data dan 7 kolom berisi data harga rumah yang tersedia di Jakarta Tenggara. Setelah data dikumpulkan, kemudian dibagi menjadi data pelatihan dan pengujian. Akhirnya, data digunakan untuk menghitung akurasi model Regresi Linier Berganda, yang menghasilkan akurasi 66%. Ini cukup baik untuk memperkirakan harga rumah berdasarkan spesifikasi spesifik yang diperlukan.

## Business Understanding

### Problem Statements

- Apa saja faktor ekonomi yang paling signifikan yang dapat mempengaruhi nilai properti, dan bagaimana kita dapat mengidentifikasinya untuk membuat keputusan investasi yang bijak?

- Bagaimana fluktuasi harga properti dapat memengaruhi keputusan investasi dan kebijakan kepemilikan rumah?

### Goals

- Tujuan utamanya adalah untuk mengembangkan model atau sistem prediktif yang dapat mengidentifikasi dan memberikan dukungan untuk faktor-faktor yang memiliki dampak terbesar pada harga rumah. Tujuan proses klarifikasi ini adalah untuk memberikan lebih banyak wawasan tentang bagaimana berbagai karakteristik properti, seperti lokasi, ukuran, dan fasilitas, mempengaruhi nilai rumah.

- Tujuan yang dimaksudkan adalah untuk mengembangkan kebijakan atau strategi yang dapat mengatasi ketidakseimbangan antara harga properti dan akses ke sumber daya yang tidak stabil. Solusi yang dihasilkan harus dapat memaksimalkan pertumbuhan ekonomi sekaligus memastikan keselamatan masyarakat dengan menyeimbangkan dampak sosial dan ekonomi dari fluktuasi harga properti.

### Solution statements

- Menggunakan pendekatan ensemble learning dengan menggabungkan beberapa model machine learning, seperti regresi linier berganda, decision tree, dan random forest. Hal ini bertujuan untuk mengoptimalkan akurasi prediksi dengan memanfaatkan kekuatan masing-masing model. Evaluasi model dapat menggunakan metrik seperti Mean Squared Error (MSE) untuk mengukur seberapa baik model memprediksi harga rumah.

## Data Understanding

Pasar real estate, seperti yang ada di Seattle, Washing, USA, menghadirkan peluang menarik bagi analis data untuk menganalisis dan memprediksi ke mana harga properti bergerak. Prediksi harga properti menjadi semakin penting dan menguntungkan. Harga properti merupakan indikator yang baik dari kondisi pasar secara keseluruhan dan kesehatan ekonomi suatu negara. Mempertimbangkan data yang diberikan, dalam memperdebatkan sejumlah besar catatan penjualan properti yang disimpan dalam format yang tidak diketahui dan dengan masalah kualitas data yang tidak diketahui.

Data yang digunakan dalam proyek ini bersumber dari Kaggle (kaggle datasets download -d samuelcortinhas/house-price-prediction-seattle)

Jumlah data sebanyak 2016 data, yang terbagi dalam 8 kolom. Kolom pertama yaitu 'beds' memiliki 2016 data bertipe integer, kolom 'baths' memiliki 2016 data bertipe float, kolom 'size' memiliki 2016 data bertipe float, kolom 'size_units' memiliki 2016 data bertipe object, kolom 'lot_size' memiliki 1669 data bertipe float, kolom 'lot_size_units' memiliki 1669 data bertipe object, kolom 'zip_code' memiliki 2016 data bertipe int, dan yang terakhir kolom 'price' sebagai target memiliki 2016 data bertipe float. Terdapat banyak missing value pada kolom lot_size dan lot_size_units sebanyak 347 data. Dataset dapat lebih lanjut dilihat pada Tabel 1.

Tabel 1. Dataset House Price Prediction - Seattle

|      | beds | baths |   size | size_units | lot_size | lot_size_units | zip_code | price     |
|-----:|-----:|------:|-------:|-----------:|---------:|---------------:|---------:|-----------|
|   0  |    3 |   2.5 | 2590.0 |       sqft |  6000.00 |           sqft |    98144 |  795000.0 |
|   1  |    4 |   2.0 | 2240.0 |       sqft |     0.31 |           acre |    98106 |  915000.0 |
|   2  |    4 |   3.0 | 2040.0 |       sqft |  3783.00 |           sqft |    98107 |  950000.0 |
|   3  |    4 |   3.0 | 3800.0 |       sqft |  5175.00 |           sqft |    98199 | 1950000.0 |
|   4  |    2 |   2.0 | 1042.0 |       sqft |      NaN |            NaN |    98102 |  950000.0 |
|  ... |  ... |   ... |    ... |        ... |      ... |            ... |      ... |       ... |
| 2011 |    3 |   2.0 | 1370.0 |       sqft |     0.50 |           acre |    98112 |  910000.0 |
| 2012 |    1 |   1.0 |  889.0 |       sqft |      NaN |            NaN |    98121 |  550000.0 |
| 2013 |    4 |   2.0 | 2140.0 |       sqft |  6250.00 |           sqft |    98199 | 1150000.0 |
| 2014 |    2 |   2.0 |  795.0 |       sqft |      NaN |            NaN |    98103 |  590000.0 |
| 2015 |    3 |   2.0 | 1710.0 |       sqft |  4267.00 |           sqft |    98133 |  659000.0 |

Tabel 2. Informasi lebih lanjut mengenai dataset

|       |        beds |       baths |         size |    lot_size |     zip_code |        price |
|------:|------------:|------------:|-------------:|------------:|-------------:|-------------:|
| count | 2016.000000 | 2016.000000 |  2016.000000 | 1669.000000 |  2016.000000 | 2.016000e+03 |
|  mean |    2.857639 |    2.159970 |  1735.740575 | 3871.059694 | 98123.638889 | 9.636252e+05 |
|  std  |    1.255092 |    1.002023 |   920.132591 | 2719.402066 |    22.650819 | 9.440954e+05 |
|  min  |    1.000000 |    0.500000 |   250.000000 |    0.230000 | 98101.000000 | 1.590000e+05 |
|  25%  |    2.000000 |    1.500000 |  1068.750000 | 1252.000000 | 98108.000000 | 6.017500e+05 |
|  50%  |    3.000000 |    2.000000 |  1560.000000 | 4000.000000 | 98117.000000 | 8.000000e+05 |
|  75%  |    4.000000 |    2.500000 |  2222.500000 | 6000.000000 | 98126.000000 | 1.105250e+06 |
|  max  |   15.000000 |    9.000000 | 11010.000000 | 9998.000000 | 98199.000000 | 2.500000e+07 |

Dapat dilihat pada tabel 2 untuk rata-rata harga rumah yaitu berada pada angka 963 ribu dollar US. Untuk harga rumah yang paling murah berada di angka 159 ribu dollar US dan yang termahal mencapai angka 2,5 juta dollar US.

### Variabel-variabel pada House Price Prediction dataset adalah sebagai berikut:

- beds : jumlah kamar tidur di properti.
- baths : Jumlah kamar mandi di properti. Catatan 0,5 sesuai dengan setengah bak mandi yang memiliki wastafel dan toilet tetapi tidak ada bak mandi atau pancuran.
- size : Total luas lantai properti.
- size_units : Unit pengukuran sebelumnya.
- lot_size : Total luas tanah tempat properti berada. Tanah itu milik pemilik rumah.
- lot_size_units : Unit pengukuran sebelumnya.
- zip_code : Kode pos. Ini adalah kode pos yang digunakan di AS.
- price : Harga properti dijual seharga (dolar AS).

### Exploratory Data Analysis

- Menangani missing value

  Dari hasil output, terlihat bahwa kolom "lot_size" dan "lot_size_units" memiliki nilai yang hilang (NaN) sebanyak 347 data. Dengan menggunakan teknik dropna, sekarang DataFrame rumah tidak mengandung baris dengan nilai yang hilang di kolom "lot_size" dan "lot_size_units".

  Tabel 3. Dataset sudah bersih dari missing value

  |      | beds | baths | size   | size_units | lot_size | lot_size_units | zip_code | price     |
  |-----:|-----:|-------|--------|------------|----------|----------------|----------|-----------|
  | 0    | 3    | 2.5   | 2590.0 | sqft       | 6000.00  | sqft           | 98144    | 795000.0  |
  | 1    | 4    | 2.0   | 2240.0 | sqft       | 0.31     | acre           | 98106    | 915000.0  |
  | 2    | 4    | 3.0   | 2040.0 | sqft       | 3783.00  | sqft           | 98107    | 950000.0  |
  | 3    | 4    | 3.0   | 3800.0 | sqft       | 5175.00  | sqft           | 98199    | 1950000.0 |
  | 5    | 2    | 2.0   | 1190.0 | sqft       | 1.00     | acre           | 98107    | 740000.0  |
  | ...  | ...  | ...   | ...    | ...        | ...      | ...            | ...      | ...       |
  | 2009 | 3    | 3.5   | 1680.0 | sqft       | 1486.00  | sqft           | 98126    | 675000.0  |
  | 2010 | 2    | 2.0   | 1400.0 | sqft       | 0.34     | acre           | 98199    | 699950.0  |
  | 2011 | 3    | 2.0   | 1370.0 | sqft       | 0.50     | acre           | 98112    | 910000.0  |
  | 2013 | 4    | 2.0   | 2140.0 | sqft       | 6250.00  | sqft           | 98199    | 1150000.0 |
  | 2015 | 3    | 2.0   | 1710.0 | sqft       | 4267.00  | sqft           | 98133    | 659000.0  |

- Menangani outliers

  Pada kasus ini, akan dideteksi outliers dengan teknik visualisasi data (boxplot). Kemudian, outliers akan ditangani dengan teknik IQR method. setelah ditangani dengan metode IQR method, dataset yang tersisa menjadi 1682 data.

  Gambar 1. Deteksi outliers pada kolom 'beds'

  ![beds](https://github.com/fannof/project_predictive_analysis/assets/99071605/30d6a272-05ce-40f2-b8f3-6b172301e2e0)


  Terlihat pada gambar 1, terdapat 4 outliers di kolom 'beds'.

  Gambar 2. Deteksi outliers pada kolom 'baths'

  ![image](https://github.com/fannof/project_predictive_analysis/assets/99071605/5e8ae48e-c7c4-4c26-8fd9-7829b2478f78)

  Terlihat pada gambar 2, terdapat 6 outliers di kolom 'baths'

  Gambar 3. Deteksi outliers pada kolom 'size'

  ![image](https://github.com/fannof/project_predictive_analysis/assets/99071605/f97ce15a-a3cc-49be-be23-00760a9470b2)

  Pada gambar 3, terdapat banyak sekali outliers, semua outliers ini akan ditangani dengan metode IQR Method.

- Univariate analysis

  Selanjutnya, akan dilakukan proses analisis data dengan teknik Univariate EDA. Pertama, lakukan analisis pada fitur numerik.
Peningkatan harga rumah sebanding dengan penurunan jumlah sampel. Hal ini dapat dilihat jelas dari histogram "price" pada gambar 4, dimana grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu x).
Semakin tinggi size, jumlah beds, dan jumlah baths dalam rumah, maka semakin mahal pula harga rumah.

  Gambar 4. Plot grafik masing-masing fitur

  ![image](https://github.com/fannof/project_predictive_analysis/assets/99071605/1d2c2658-3292-4522-a8a7-100deff8dd2b)

- Multivariate analysis

  Selanjutnya, akan dilakukan analisis data pada fitur numerik menggunakan teknik Multivariate EDA menggunakan fungsi pairplot() dan juga akan mengobservasi korelasi antara fitur numerik dengan fitur target menggunakan fungsi corr().
  Pada gambar 5 yaitu pola sebaran data grafik pairplot, terlihat fitur "size" memiliki korelasi positif dengan fitur "price". Sedangkan kedua fitur "lot_size" dan "price" tidak memliki korelasi karena tidak membetuk pola.

  Gambar 5. Grafik korelasi antar fitur numerik dan fitur target

  ![image](https://github.com/fannof/project_predictive_analysis/assets/99071605/c38c6c64-e41a-45eb-9022-7a6d66971ff9)

  Gambar 6. Matrik korelasi antar fitur numerik

  ![image](https://github.com/fannof/project_predictive_analysis/assets/99071605/570e873b-5804-431d-b0f2-10efdbed5690)

  Pada gambar 6, grafik korelasi terlihat bahwa fitur 'beds', 'baths', dan 'size' memiliki skor korelasi yang besar dengan fitur target 'price'. Artinya, fitur 'price' berkorelasi tinggi dengan ketiga fitur tersebut. Sementara itu, fitur 'lot_size' dan 'zip_code' memiliki korelasi yang sangat kecil sehingga fitur tersebut dapat di-drop.

  Tabel 4. Dataframe setelah fitur yang tidak dibutuhkan di-drop

  |   | beds | baths |   size |    price |
  |--:|-----:|------:|-------:|---------:|
  | 0 |    3 |   2.5 | 2590.0 | 795000.0 |
  | 1 |    4 |   2.0 | 2240.0 | 915000.0 |
  | 2 |    4 |   3.0 | 2040.0 | 950000.0 |
  | 5 |    2 |   2.0 | 1190.0 | 740000.0 |
  | 6 |    1 |   1.0 |  670.0 | 460000.0 |

  Gambar 7. Grafik plot antar fitur numerik setelah fitur yang tidak dibutuhkan di-drop

  ![image](https://github.com/fannof/project_predictive_analysis/assets/99071605/89e1a48b-9af0-426a-a357-16fcf9c87624)

## Data Preparation

- Train-Test-Split

  Proses membagi himpunan data menjadi data pelatihan dan pengujian adalah langkah yang diperlukan sebelum membuat model. Ini penting dilakukan untuk memperkuat semua data yang tersedia untuk menilai beberapa generalisasi model ke data baru. Perlu dicatat bahwa setiap transformasi data yang dilakukan juga berfungsi sebagai komponen model. Karena data test set (uji) mentah, semua transformasi harus dilakukan pada data latih. Data dibagi menjadi 80% data training dan 20% data testing, karena jumlah seluruh data termasuk kecil, maka diperlukan lebih banyak data latih.

  Total sampel seluruh dataset: 1380

  Total sampel data latih: 1104

  Total sampel data uji: 276

- Standarisasi

  Ketika algoritma pembelajaran mesin diterapkan pada data dengan distribusi yang serupa atau menyimpang, mereka berkinerja lebih baik dan menyatu lebih cepat. Proses penskalaan dan standardisasi membantu mengubah data menjadi format yang lebih mudah dipahami oleh algoritma.

  Tabel 5. 

  |      |      beds |     baths |      size |
  |-----:|----------:|----------:|----------:|
  | 1779 | -1.973241 | -1.412389 | -1.635772 |
  |  316 |  0.014403 | -0.809653 | -0.143301 |
  | 1385 | -0.979419 |  0.395818 | -1.011919 |
  | 1666 | -1.973241 | -1.412389 | -1.416378 |
  | 2004 | -0.979419 | -0.206917 | -0.744767 |

  Standardisasi adalah teknik transformasi yang paling umum digunakan dalam proses pembangunan model. Ini tidak akan mengubah fitur numerik menggunakan encoding. Teknik yang digunakan adalah StandarScaler dari library Scikit-learn.

  Tabel 6.

  |       |      beds | baths     | size      |
  |------:|----------:|-----------|-----------|
  | count | 1104.0000 | 1104.0000 | 1104.0000 |
  | mean  | 0.0000    | -0.0000   | 0.0000    |
  | std   | 1.0005    | 1.0005    | 1.0005    |
  | min   | -1.9732   | -2.0151   | -2.2029   |
  | 25%   | -0.9794   | -0.8097   | -0.7679   |
  | 50%   | 0.0144    | -0.2069   | -0.1209   |
  | 75%   | 1.0082    | 0.3958    | 0.6776    |
  | max   | 3.9897    | 3.4095    | 3.3342    |

### Penjelasan tahapan dan kenapa harus dilakukan proses tersebut

- Proses data prepraration

  Pertama adalah proses train-test-split. Data dibagi menjadi 80% data training dan 20% data testing, karena jumlah seluruh data termasuk kecil, maka diperlukan lebih banyak data latih.

  Proses standarisasi mengubah nilai mean menjadi 0 dan std menjadi 1. StandardScaler melakukan proses standardisasi parameter fitur terlebih dahulu dengan mengurangkan nilai mean (nilai rata-rata) dan kemudian membandingkannya dengan standar deviasi untuk menentukan distribusi.  StandardScaler menghasilkan distribusi dengan rata-rata 0 dan standar deviasi 1.

  Tahapan diatas penting dilakukan karena algoritma machine learning memiliki performa lebih baik ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal.

## Modeling

Model akan dikembangkan dengan 3 algoritma yang berbeda, dan mencari mana yang memiliki performa paling baik. Beberapa algoritma tersebut adalah sebagai berikut:

1. k-NN

   Merupakan algoritma supervised learning yang mengklasifikasikan hasil instance yang baru dibuat berdasarkan mayoritas kategori k-tetangga terdekat.
   Tujuan algoritma ini adalah untuk mengklasifikasikan objek baru berdasarkan atribut dan data sampel-sampel dari set pelatihan.
   Algoritma k-Nearest Neighbor menggunakan Neighborhood Classification sebagai nilai prediksi yang berasal dari instance baru. Seperti yang terlihat pada gambar 8.

   Gambar 8. Algoritma k-NN

   ![th](https://github.com/fannof/project_predictive_analysis/assets/99071605/2ac7ac6a-1790-4f6e-8a9b-82645c01735b)


    Langkah yang pertama, model k-NN diinisialisasi dengan menentukan jumlah tetangga terdekat (parameter n_neighbors). Contoh dalam kasus ini adalah n_neighbors diatur ke 10, artinya model akan menggunakan 10 tetangga tetangga yang paling dekat untuk membuat prediksi.
    Setelah model diinisialisasi, langkah selanjutnya adalah melatih model menggunakan data latih. Untuk melatih model dengan fitur X_train dan target y_train, gunakan fungsi fit(X_train, y_train).
    Setelah proses pelatihan selesai, model sudah dapat membuat prediksi pada data latih untuk mengevaluasi performa model. 
  
    Parameter yang Digunakan pada Model KNN:

    - n_neighbors: Jumlah tetangga terdekat yang digunakan untuk membuat prediksi.

3. Random Forest

    Model Random Forest diinisialisasi dengan menentukan beberapa hyperparameter.
    Model Random Forest dilatih menggunakan data latih (X_train dan y_train). Fungsi fit(X_train, y_train) digunakan untuk melatih model.
    Setelah pelatihan selesai, model sekarang dapat digunakan untuk membuat prediksi pada data latih. RF.predict(X_train) menghasilkan prediksi target berdasarkan fitur pada data latih.
  
    Parameter yang Digunakan pada Model Random Forest:
  
    - n_estimators: Jumlah pohon keputusan dalam ensemble.
  
    - max_depth: Kedalaman maksimum setiap pohon keputusan.

    - random_state: Digunakan untuk memastikan hasil yang reproduktif.
  
    - n_jobs: Jumlah pekerjaan paralel yang akan dijalankan.

4. Boosting Algorithm

    Model Boosting (AdaBoostRegressor) diinisialisasi dengan menentukan hyperparameter tertentu. Parameter yang diatur adalah learning_rate dengan nilai 0.05. random_state digunakan untuk memastikan reproduktibilitas hasil.
    Model diarahkan untuk mempelajari hubungan antara fitur (X_train) dan target (y_train). Fungsi fit(X_train, y_train) digunakan untuk melatih model dengan data latih.
    Setelah pelatihan selesai, sekarang model dapat digunakan untuk membuat prediksi pada data latih. boosting.predict(X_train) menghasilkan prediksi target berdasarkan fitur pada data latih.
   
    Parameter yang Digunakan pada Model Boosting (AdaBoostRegressor):
   
    - learning_rate: Menentukan sejauh mana model belajar dari kesalahan sebelumnya. Nilai yang lebih kecil akan memperbaiki konvergensi, tetapi memerlukan jumlah estimator (pohon       keputusan) yang lebih besar.
  
    - n_estimators: Jumlah estimator (pohon keputusan) yang digunakan.

    - base_estimator: Tipe model dasar yang digunakan. Secara default, digunakan pohon keputusan (DecisionTreeRegressor).

    - random_state: Digunakan untuk memastikan hasil yang reproduktif.

  - Kelebihan dan kekurangan dari setiap algoritma yang digunakan:

    -- KNN memiliki MSE yang tinggi pada data uji, mungkin karena model tidak dapat menangkap pola yang kompleks dalam data tersebut.

    -- Random Forest memiliki MSE yang rendah pada data latih, menunjukkan kemampuan baik dalam menyesuaikan dengan data latih. Namun, terdapat peningkatan yang signifikan pada MSE pada data uji, mungkin menunjukkan adanya overfitting.

    -- Boosting memberikan hasil yang cukup baik pada data uji, menunjukkan kemampuan model untuk mengatasi kompleksitas data.

  - Alasan memlilih Model Boosting Algorithm

    Dari ketiga model diatas, Boosting memiliki MSE yang relatif lebih rendah pada data uji, menunjukkan kinerja yang lebih baik dibandingkan dengan KNN dan Random Forest dalam dataset harga rumah ini.

    Berdasarkan hasil tersebut, solusi terbaik pada kasus ini adalah menggunakan model Boosting (AdaBoostRegressor) karena memberikan performa yang lebih baik dalam memprediksi harga rumah pada data yang belum pernah dilihat sebelumnya.

## Evaluation

- Metrik yang digunakan adalah MSE

  Mean Squared Error (MSE) adalah salah satu metrik evaluasi yang umum digunakan dalam regresi untuk mengukur sejauh mana perbedaan antara nilai prediksi model dengan nilai aktual (ground truth). MSE dihitung dengan menjumlahkan kuadrat selisih antara setiap nilai prediksi dan nilai aktual, kemudian diambil rata-rata dari seluruh data. Nilai MSE semakin kecil semakin baik. Nilai MSE sama dengan nol berarti model memberikan prediksi yang sempurna sesuai dengan nilai aktual.

  Tabel 7. 

  |          |           train | test            |
  |---------:|----------------:|-----------------|
  | KNN      | 44360753.266994 | 54239740.502329 |
  | RF       | 12390218.332403 | 58255407.24918  |
  | Boosting | 49336053.971147 | 51079691.292662 |

  Gambar

  ![image](https://github.com/fannof/project_predictive_analysis/assets/99071605/5985c172-5ba8-49a3-b9d9-530935352cae)

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

|      |    y_true | prediksi_KNN | prediksi_RF | prediksi_Boosting |
|-----:|----------:|-------------:|------------:|------------------:|
| 1680 | 1725000.0 |    1285250.0 |   1316599.0 |         1306669.6 |

**Rubrik**

The Mean Squared Error (MSE) is calculated using the formula:

$\[ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]$

where $\( n \)$ is the number of observations, $\( y_i \)$ is the actual value, and $\( \hat{y}_i \)$ is the predicted value.

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
