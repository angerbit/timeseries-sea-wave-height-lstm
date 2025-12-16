## Project Overview
Prediksi tinggi gelombang laut merupakan salah satu tantangan penting dalam bidang oseanografi, perkapalan dan energi terbarukan. Kemampuan memprediksi tinggi gelombang secara akurat dapat membantu dalam mitigasi bencana, perencanaan pelayaran dan optimalisasi pembangkit listrik tenaga gelombang.
Proyek ini bertujuan untuk mengembangkan model deep learning menggunakan asitektur Long Short-Term Memory (LSTM) untuk memprediksi tinggi gelombang berdasarkan data time series. Pemilihan arsitektur LSTM ini bertujuan untuk menangani ketergantungan jangka panjang dalam data. Pendekatan ini telah terbukti lebih unggul dalam menangkap pola temporal data gelombang (Wang et al., 2024)
## Business Understanding
### Problem Statements
- Tinggi gelombang merupakan variabel dinamis yang dipengaruhi oleh banyak faktor seperti angin, pasang surut, dan arus laut.
- Prediksi manual atau berbasis model statistik tradisional (seperti ARIMA) seringkali kurang akurat untuk data non-linear dan kompleks (Zhang, 2003).
- Dibutuhkan pendekatan yang lebih canggih untuk meningkatkan akurasi prediksi tinggi gelombang.
### Goals
- Mengembangkan model LSTM yang dapat memprediksi gelombang laut dengan tingkat akurasi yang tinggi.
- Melakukan evaluasi performa model menggunakan metrik akurasi dan MSE (mean squared error)
- Membandingkan hasil prediksi dengan data sebenarnya.
### Solution Statements
- Menggunakan varian RNN yang efektif untuk pemodelan time series yaitu LSTM
- Menerapkan beberapa teknik dalam preprocessing data.
- Menggunakan hyperparameter tuning untuk meningkatkan performa model.
## Data Understanding

### Sumber dan Deskripsi Data
Data yang digunakan dalam proyek ini bersumber dari **Climate Data Store (CDS) Copernicus**, khususnya dataset **ERA5 post-processed daily statistics on single level from 1940 to present**. Dataset ini menyediakan data statistik harian parameter oseanografi dan meteorologi dengan resolusi spasial 0.25° × 0.25° (sekitar 31 km).

**Detail Dataset:**
- **Dimensi Data**: 2,191 record (baris) × 6 fitur (kolom)
- **Periode Data**: 1 Januari 2020 - 31 Desember 2025 (6 tahun)
- **Format Data**: NetCDF
- **Lisensi**: Free and open access
- **Resolusi Temporal**: Harian

**Kondisi Data:**
- Tidak terdapat missing values (data lengkap)
- Tidak ditemukan data duplikat
- Nilai outlier telah ditangani melalui quality control oleh ECMWF
- Nilai -9999 menunjukkan data tidak tersedia (tidak ditemukan pada periode ini)

**Tautan Dataset:**
[ERA5 Daily Statistics Dataset](https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=overview)

### Variabel/Fitur yang Digunakan

Berikut adalah variabel utama yang digunakan dalam analisis beserta deskripsi teknisnya:

| No | Kode Variabel | Nama Variabel | Deskripsi | Satuan | Rentang Nilai |
|----|--------------|---------------|-----------|--------|--------------|
| 1 | hmax | Maximum Wave Height | Tinggi gelombang tunggal maksimum dalam interval waktu | meter | 0 - 20 m |
| 2 | mp2 | Mean Wave Period | Rata-rata periode gelombang berdasarkan zero-crossing | detik | 0 - 25 s |
| 3 | mwd | Mean Wave Direction | Arah propagasi gelombang relatif terhadap utara sejati | derajat | 0°-360° |
| 4 | pp1d | Peak Wave Period | Periode gelombang dengan energi spektral maksimum | detik | 3 - 30 s |
| 5 | sst | Sea Surface Temperature | Suhu permukaan laut pada kedalaman 0.5 m | Kelvin | 270 - 310 K |
| 6 | swh | Significant Wave Height | Tinggi gelombang signifikan (rata-rata 1/3 tertinggi) | meter | 0 - 15 m |

**Keterangan Tambahan:**
- Semua variabel merupakan hasil reanalisis model ECMWF
- Nilai merupakan rata-rata spasial untuk area studi tertentu
## Data Preparation
### Tahapan Persiapan Data
Proses persiapan data dilakukan secara berurutan dengan tahapan sebagai berikut:
1. **Noise Reduction**
   - Menggunakan filter Savitzky-Golay untuk mengurangi noise pada data time series
   - Parameter: window_length=15, polyorder=3
   - Dilakukan pada semua variabel numerik
   - Menghasilkan variabel dengan suffix '_sg' (contoh: swh_sg)

2. **Handling Missing Values**
   - Mengecek dan menghapus record dengan NaN values (jika ada)
   - Pada dataset ini tidak ditemukan missing values
   - Backup strategy: Interpolasi linear untuk data kontinu jika ditemukan NaN

3. **Normalisasi Data** 
   - Menggunakan MinMaxScaler dengan range [0,1]
   - Rumus: X' = (X - X_min)/(X_max - X_min)
   - Dilakukan pada semua fitur sebelum pemodelan
   - Menyimpan parameter scaling untuk inverse transform

4. **Pembagian Dataset**
   - Split sequential (karena data time series)
   - Training set: 1400 record pertama
   - Test set: 517 record terakhir

5. **Windowing Data (Khusus Model LSTM)**
   - Membuat sequence data dengan window size=32 timesteps
   - Format: [samples, timesteps, features]
   - Dilakukan terpisah untuk training dan test set


## Modeling
### Arsitektur Model
Model LSTM dirancang untuk prediksi multi-output (6 fitur gelombang) yang diimplementasikan menggunakan keras dan menghasilkan model summary sebagai berikut:
Model: "sequential_6"
| Layer (type) | Output Shape | Param # | 
|------|------| ------|
|lstm_18 (LSTM)|(None, 6, 32)|8320    |  
 lstm_19 (LSTM)|(None, 6, 16)|3136   |   
 dropout_6 (Dropout)|(None, 6, 16)|0|         
 lstm_20 (LSTM)|(None, 10)|1080|
 dense_6 (Dense)|(None, 6)| 66 |  
=======================================
Total params: 12,602
Trainable params: 12,602
Non-trainable params: 0

### Penjelasan Arsitektur
1. **LSTM Layer Pertama**:
   - Unit: 32 (default activation tanh)
   - Input shape: (n_timesteps, n_features)
   - Return_sequences=True (untuk meneruskan sequence ke layer berikutnya)
   - Berfungsi sebagai feature extractor pola temporal
2. **LSTM Layer Kedua**:
   - Unit: 16 (dimensi tersembunyi dikurangi)
   - Return_sequences=True (mempertahankan struktur sequence)
   - Bertujuan menangkap dependensi jangka panjang yang lebih kompleks
3. **Dropout Layer**:
   - Rate: 0.2 (default)
   - Berfungsi sebagai regularisasi dengan menonaktifkan 20% neuron secara acak
   - Mencegah overfitting selama training
4. **LSTM Layer Ketiga**:
   - Unit: 10
   - Return_sequences=False (mengkonversi sequence ke vektor)
   - Sebagai encoder akhir sebelum dense layer

5. **Dense Layer**:
   - Unit: 6 (sesuai jumlah output)
   - Activation linear (default untuk masalah regresi)
   - Bertugas memetakan fitur tersembunyi ke output target

### Cara Kerja LSTM
LSTM bekerja dengan menggunakan mekanisme *gate* yang mengatur aliran informasi dalam jaringan. Pertama, **forget gate** menentukan informasi mana yang akan dibuang dari *cell state* (memori jangka panjang) dengan menggunakan fungsi sigmoid, menghasilkan nilai antara 0 (lupakan sepenuhnya) dan 1 (pertahankan sepenuhnya). Selanjutnya, **input gate** memutuskan informasi baru apa yang akan disimpan ke dalam *cell state*. Proses ini melibatkan dua langkah: (1) sebuah sigmoid layer memutuskan nilai mana yang akan diperbarui, dan (2) sebuah layer *tanh* menghasilkan vektor kandidat nilai baru. Kedua hasil ini kemudian digabungkan untuk memperbarui *cell state*. Terakhir, **output gate** mengatur nilai output berdasarkan *cell state* yang telah diperbarui. 

Pada implementasi ini, tiga lapisan LSTM digunakan untuk:
1. Ekstraksi fitur temporal level rendah (layer 1)
2. Pemodelan dependensi level menengah (layer 2)
3. Agregasi temporal ke representasi fixed-length (layer 3)

### Parameter Training
- Optimizer: Adam (default learning_rate=0.001)
- Loss function: Mean Squared Error (sesuai masalah regresi)
- Metrics: MSE (Mean Squared Error)
- Batch size: 32 (default)


## Evaluation

### Hasil Pelatihan Model

Pelatihan model dilakukan selama 300 *epochs* dengan pembagian data sebesar 75% untuk *training* dan 25% untuk *validation*. Waktu pelatihan yang dibutuhkan relatif singkat, yaitu 1 menit 13.2 detik. Gambar di bawah menunjukkan grafik akurasi dan kerugian model selama proses pelatihan:
![Hasil Training](https://github.com/angerbit/SeaWaveHeight_LSTM_forecasting/blob/main/image/Training%20Result.png?raw=true)

Model menunjukkan performa yang cukup baik, dengan akurasi *training* mencapai 80% dan akurasi *validation* stabil di angka 70%, yang mengindikasikan bahwa model **tidak mengalami overfitting**. Nilai *training loss* menurun signifikan hingga 0.02 dan *validation loss* stabil di 0.12 setelah ±150 *epochs*, dengan **konvergensi tercapai sejak sekitar epoch ke-100**. Hal ini menunjukkan bahwa arsitektur dan parameter yang dipilih berhasil melatih model secara efektif, sesuai dengan salah satu solution statement, yaitu penerapan **hyperparameter tuning** untuk meningkatkan performa.

### Hasil Plot Prediksi

![Plot Hasil Prediksi](https://github.com/angerbit/SeaWaveHeight_LSTM_forecasting/blob/main/image/Prediction%20Plot.png?raw=true)

Untuk mengevaluasi performa model dalam konteks *business understanding*, berikut adalah nilai RMSE (*Root Mean Squared Error*) dari masing-masing parameter prediksi:

* **RMSE of swh**: 7.241 m
* **RMSE of hmax**: 0.146 m
* **RMSE of mp2**: 0.274 s
* **RMSE of pp1d**: 0.644 s
* **RMSE of mwd**: 0.397°
* **RMSE of sst**: 0.625 K

Model berhasil memprediksi beberapa parameter penting seperti *maximum wave height* (hmax) dengan sangat akurat, di mana nilai RMSE yang rendah menunjukkan **potensi aplikasi praktis dalam keselamatan pelayaran dan operasional maritim**, yang merupakan bagian penting dari pemahaman bisnis kelautan. Sementara itu, variabel *significant wave height* (swh) masih memiliki error cukup tinggi, terutama pada kondisi ekstrem (gelombang di atas 2.5 m), yang menandakan bahwa model perlu diperbaiki untuk menangani dinamika ekstrem—suatu hal penting dalam mitigasi risiko laut.

Prediksi periode gelombang (pp1d dan mp2) serta arah gelombang (mwd) dan suhu permukaan laut (sst) menunjukkan kemampuan model dalam menangkap tren musiman dan variasi harian, menjawab kebutuhan akan metode prediksi yang lebih adaptif terhadap data non-linear dan kompleks, sebagaimana dinyatakan dalam *problem statements*.

### Keterkaitan dengan Business Understanding
Model yang dikembangkan menggunakan pendekatan LSTM berhasil menjawab tantangan utama dari *problem statements*, yaitu:

* Ketidakmampuan model statistik tradisional dalam menangkap pola data laut yang kompleks dan non-linear, kini telah dijawab melalui penerapan LSTM yang memang didesain untuk menangani data sekuensial dan temporal.
* Pendekatan ini juga telah menunjukkan efektivitas dalam mengurangi error prediksi untuk sebagian besar parameter, mendukung pencapaian *goal* untuk menghasilkan model prediksi gelombang laut dengan **tingkat akurasi yang tinggi**.

Dari sisi *solution statements*, implementasi preprocessing, pemilihan arsitektur LSTM, dan tuning hyperparameter **terbukti berdampak langsung** terhadap performa model, baik dari segi kecepatan konvergensi maupun stabilitas hasil validasi.

Secara keseluruhan, model telah memenuhi **tujuan utama** proyek, yaitu mengembangkan sistem prediksi gelombang laut yang akurat. Meskipun masih terdapat ruang perbaikan pada prediksi kondisi ekstrem, hasil saat ini menunjukkan bahwa solusi yang diusulkan **berdampak nyata terhadap permasalahan awal**, dan model sudah cukup **layak untuk digunakan sebagai dasar dalam sistem monitoring atau peringatan dini**, dengan pengembangan lebih lanjut untuk peningkatan robustness.

## Kesimpulan
Berdasarkan hasil pemodelan, diperoleh kesimpulan bahwa Metode _Long-Short Term Memory_ (LSTM) terbukti akurat dalam memprediksi gelombang laut dan parameter pendukungnya dimana ditunjukkan pada akurasi training dan validation data yang mencapai 80%. Plot _time series_ juga menunjukkan bahwa model LTSM dapat memahami pola dinamika gelombang laut dan parameter lainnya.
## Referensi
- Copernicus Climate Change Service, Climate Data Store, (2024): ERA5 post-processed daily-statistics on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: 10.24381/cds.4991cf48 (Accessed on 20-04-2025)
- Wang, J., Bethel, B. J., Xie, W., & Dong, C. (2024). A hybrid model for significant wave height prediction based on an improved empirical wavelet transform decomposition and long-short term memory network. Ocean Modelling, 102367.
- Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. Neurocomputing, 50, 159-175.
