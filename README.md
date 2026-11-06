# Earthquake Daily Forecasting Using LSTM

Proyek Machine Learning untuk melakukan peramalan jumlah gempa bumi signifikan harian menggunakan metode **Long Short-Term Memory (LSTM)** dan mengimplementasikan hasilnya ke dalam aplikasi web berbasis Flask.

## Deskripsi

Proyek ini menggunakan data historis kejadian gempa bumi dari tahun 1965 sampai 2016. Data diproses menjadi jumlah kejadian gempa signifikan per hari, kemudian digunakan untuk melatih model LSTM.

Model menggunakan data 90 hari sebelumnya untuk memperkirakan jumlah kejadian pada hari berikutnya. Hasil peramalan dapat ditampilkan melalui aplikasi web dengan pilihan periode 1, 7, 14, dan 30 hari.

**Catatan:** Proyek ini memperkirakan jumlah catatan kejadian gempa signifikan per hari. Proyek ini bukan sistem prediksi waktu, lokasi, atau magnitudo gempa bumi dan bukan sistem peringatan dini.

## Fitur

- Validasi dan pengolahan dataset gempa bumi.
- Seleksi kejadian dengan magnitudo minimal 5,5.
- Agregasi jumlah kejadian gempa berdasarkan tanggal.
- Pembangunan model menggunakan LSTM.
- Perbandingan tiga konfigurasi LSTM.
- Evaluasi menggunakan RMSE, MAE, MAPE, WAPE, dan R².
- Perbandingan dengan metode naive forecast.
- Peramalan untuk periode 1, 7, 14, dan 30 hari.
- Dashboard berbasis Flask.
- Visualisasi data historis dan hasil peramalan.

## Dataset

Dataset yang digunakan adalah **USGS Significant Earthquakes**, dengan periode:

- Periode awal: 2 Januari 1965
- Periode akhir: 30 Desember 2016
- Jumlah data awal: 23.412 catatan
- Jumlah data setelah seleksi: 23.232 catatan
- Jumlah data setelah agregasi harian: 18.991 baris

Kriteria seleksi data:

- Jenis kejadian: `Earthquake`
- Magnitudo: minimal 5,5

Variabel target adalah `jumlah_gempa`, yaitu jumlah catatan kejadian gempa yang memenuhi kriteria pada setiap tanggal.

**Sumber dataset:** https://www.kaggle.com/datasets/usgs/earthquake-database

## Metode

Model utama yang digunakan adalah Long Short-Term Memory (LSTM).

Konfigurasi eksperimen:

| Parameter | Nilai |
|---|---|
| Metode | LSTM |
| Time window | 90 hari |
| Pembagian data | 80% training, 10% validation, 10% testing |
| Optimizer | Adam |
| Learning rate | 0,001 |
| Loss function | Mean Squared Error |
| Batch size | 64 |
| Maksimum epoch | 30 |
| Early stopping patience | 5 |

Tiga konfigurasi model yang dibandingkan:

1. `LSTM_32`
2. `LSTM_64`
3. `LSTM_Stacked`

Model dipilih berdasarkan nilai RMSE terkecil pada data validasi.

## Hasil Evaluasi

Model yang terpilih adalah `LSTM_32`.

### Hasil pengujian

| Model | RMSE | MAE | MAPE (%) | WAPE (%) | R² |
|---|---:|---:|---:|---:|---:|
| LSTM_32 | 1,518400 | 1,109084 | 45,661122 | 88,280640 | -0,015379 |
| Naive Forecast | 1,967098 | 1,282105 | 74,881971 | 102,052780 | -0,704150 |

Hasil menunjukkan bahwa LSTM_32 memperoleh nilai RMSE dan MAE yang lebih rendah dibandingkan metode naive forecast.

Namun, nilai R² masih negatif dan hasil peramalan jangka lebih panjang cenderung mendekati nilai konstan. Hal ini menunjukkan bahwa model masih belum mampu menangkap variasi jumlah kejadian gempa secara optimal.

## Teknologi yang Digunakan

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML dan CSS
- Bootstrap
- Chart.js
- Jupyter Notebook

## Struktur Proyek

```text
TA-04-Earthquake-Forecasting/
│
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── data/
│   ├── raw/
│   │   └── database.csv
│   └── processed/
│       ├── earthquake_daily_counts.csv
│       └── future_forecast.csv
│
├── models/
│   ├── best_lstm_model.keras
│   ├── scaler.joblib
│   └── model_metadata.json
│
├── notebooks/
│   └── 01_Validasi_Dataset.ipynb
│
├── reports/
│   ├── evaluation_results.csv
│   └── validation_comparison.csv
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Instalasi dan Menjalankan Proyek

### 1. Clone repository

```bash
git clone https://github.com/ReidOates/AI4ML.git
```

Masuk ke direktori proyek:

```bash
cd TA-04-Earthquake-Forecasting
```

### 2. Buat virtual environment

Untuk Windows:

```bash
python -m venv venv
```

Aktifkan virtual environment:

```bash
venv\Scripts\activate
```

### 3. Instal dependensi

```bash
pip install -r requirements.txt
```

### 4. Siapkan file model dan data

Pastikan file yang diperlukan tersedia pada direktori berikut:

- `models/best_lstm_model.keras`
- `models/scaler.joblib`
- `models/model_metadata.json`
- File data historis yang dibutuhkan aplikasi pada direktori `data/processed/`

Dataset mentah dapat ditempatkan pada `data/raw/` jika ingin menjalankan ulang proses pengolahan data.

### 5. Jalankan aplikasi

```bash
python app/app.py
```

Buka browser dan akses:

```text
http://127.0.0.1:5000
```

## Keterbatasan

Beberapa keterbatasan dalam proyek ini:

1. Data yang digunakan hanya berupa jumlah kejadian gempa harian.
2. Model belum menggunakan informasi tambahan seperti lokasi, kedalaman, dan magnitudo sebagai fitur masukan.
3. Distribusi data tidak merata dan memiliki banyak hari dengan nilai nol.
4. Hasil prediksi masih cenderung mendekati nilai rata-rata.
5. Model belum mampu menggambarkan lonjakan jumlah kejadian secara optimal.
6. Hasil peramalan tidak dapat digunakan sebagai dasar untuk menentukan waktu dan lokasi pasti terjadinya gempa.

## Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan:

- Menambahkan fitur magnitudo, kedalaman, dan lokasi kejadian.
- Menguji pendekatan yang sesuai untuk data dengan banyak nilai nol.
- Membandingkan LSTM dengan metode peramalan lainnya.
- Menguji panjang time window yang berbeda.
- Mengembangkan strategi peramalan untuk mengurangi kecenderungan prediksi konstan.

## Penulis

Muhammad Emil Mushthopa

Program Studi Teknik Informatika  
Universitas Bale Bandung

## Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dan tugas akademik. Penggunaan dataset tetap mengikuti ketentuan lisensi dan atribusi dari sumber dataset aslinya.
