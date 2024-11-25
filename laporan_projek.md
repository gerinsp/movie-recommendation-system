# Laporan Proyek Machine Learning - Gerin Sena Pratama

## Project Overview

Dalam era digital saat ini, sistem rekomendasi memainkan peran penting di berbagai platform, seperti e-commerce, layanan streaming, dan media sosial. Meskipun platform tersebut menawarkan banyak pilihan, pengguna sering kali kesulitan menemukan item yang relevan dengan preferensi mereka. Sistem rekomendasi tradisional sering kali memberikan hasil yang kurang akurat, yang dapat mengurangi pengalaman pengguna dan tingkat retensi mereka. Masalah ini timbul karena keterbatasan dalam menangkap preferensi pengguna yang lebih spesifik dan dinamis, serta kurangnya personalisasi dalam rekomendasi yang diberikan.

Proyek ini bertujuan untuk mengatasi masalah tersebut dengan mengembangkan sistem rekomendasi film menggunakan dua pendekatan utama: content-based filtering dan collaborative filtering. Collaborative filtering (CF) merekomendasikan item berdasarkan kemiripan pengguna dalam hal memilih atau memberi nilai kepada item, sedangkan content-based filtering (CBF) merekomendasikan item berdasarkan kemiripan item dalam hal isi atau konten yang disukai oleh pengguna [1]. Kedua pendekatan ini memiliki kekuatan dan kelemahan masing-masing. Content-based filtering terbatas pada item yang sudah dikenal oleh pengguna, sementara collaborative filtering dapat kurang efektif jika data interaksi pengguna belum cukup banyak. Oleh karena itu, meskipun kedua metode ini berdiri sendiri, masing-masing dapat memberikan rekomendasi yang lebih baik sesuai dengan konteks dan data yang tersedia.

Dengan penerapan sistem rekomendasi yang lebih baik, diharapkan dapat meningkatkan pengalaman pengguna dalam menemukan film yang sesuai dengan preferensi mereka, yang pada gilirannya dapat meningkatkan kepuasan dan retensi pengguna di platform.

**Referensi:**

[1] Hidayat Arfisko, Hilmi & Wibowo, Agung Toto. (2021). Sistem Rekomendasi Film Menggunakan Metode Hybrid Collaborative Filtering dan Content-Based Filtering. Fakultas Informatika, Universitas Telkom, Bandung, Indonesia.

## Business Understanding

### Problem Statements
1. Bagaimana cara merekomendasikan film kepada pengguna berdasarkan preferensi genre mereka?  
2. Bagaimana cara merekomendasikan film kepada pengguna berdasarkan perilaku pengguna lain yang memiliki pola preferensi serupa?

### Goals
1. Mengembangkan sistem rekomendasi berbasis konten untuk memberikan rekomendasi berdasarkan genre film.  
2. Mengembangkan sistem rekomendasi berbasis kolaborasi untuk memberikan rekomendasi berdasarkan pola rating pengguna lain.

### Solution Statements
1. *Content-based filtering* menggunakan representasi fitur dari genre film.  
2. *Collaborative filtering* menggunakan pendekatan neural network untuk mempelajari hubungan antara pengguna dan film.

## Data Understanding

Data yang digunakan dalam proyek ini diambil dari Kaggle, yang berisi 2 file csv, yaitu movie dan rating. Dataset dapat diunduh melalui tautan berikut: [Movies Dataset](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system).

### Informasi Data

#### 1. Movies Dataset
- **Jumlah Data:** Dataset ini memiliki 62423 baris dan 3 kolom.
- **Duplicate Values:** Dataset ini tidak mengandung data duplikat.
- **Missing Values:** Tidak terdapat missing values pada data.

#### 2. Ratings Dataset
- **Jumlah Data:** Dataset ini memiliki 25000095 baris dan 4 kolom.
- **Duplicate Values:** Dataset ini tidak mengandung data duplikat.
- **Missing Values:** Tidak terdapat missing values pada data.

### Variabel-variabel dalam dataset adalah sebagai berikut:

#### 1. Movies Dataset
Dataset ini berisi informasi tentang film yang ada dalam sistem rekomendasi, dengan kolom-kolom berikut:

- **movieId**: ID unik untuk setiap film dalam dataset.
- **title**: Judul film, yang juga mencakup tahun rilis dalam format tahun (misalnya "Toy Story (1995)").
- **genres**: Kategori atau genre dari film, yang dapat mencakup beberapa genre yang dipisahkan oleh tanda pemisah ("|"), seperti "Adventure", "Animation", "Comedy", dll.

#### 2. Ratings Dataset
Dataset ini berisi informasi mengenai rating yang diberikan oleh pengguna untuk setiap film. Kolom-kolom dalam dataset ini meliputi:

- **userId**: ID unik untuk setiap pengguna yang memberikan rating.
- **movieId**: ID film yang dirating oleh pengguna. Ini merujuk ke ID film di *Movies Dataset*.
- **rating**: Nilai rating yang diberikan oleh pengguna, biasanya dalam rentang 1 hingga 5, yang menggambarkan seberapa besar pengguna menyukai film tersebut.
- **timestamp**: Waktu atau timestamp ketika rating diberikan oleh pengguna, dalam format Unix timestamp.

### Exploratory Data Analysis
- Distribusi Rating: Sebagian besar pengguna memberikan rating di kisaran 3, 4, 5, serta rating yang lebih spesifik seperti 3.5 dan 4.5. Hal ini menunjukkan bahwa mayoritas rating yang diberikan pengguna memiliki nilai menengah hingga tinggi.

![Visualisasi Rating](image/rating.png)

- Genre yang Paling Banyak Muncul: Genre yang paling banyak muncul adalah "Comedy" dan "Drama", yang menunjukkan preferensi pengguna terhadap film dengan tema komedi dan drama.

![Visualisasi Genre](image/genre.png)

## Data Preparation

Langkah-langkah yang dilakukan:
1. Menghapus film yang tidak mempunyai genre atau *'no genres listed'*. 
```python
movies_data = movies_data[movies_data['genres'] != '(no genres listed)']
```
Alasan dilakukan tahapan ini:  
- Menghapus data yang tidak mempunyai genre dikarenakan data tidak berpengaruh signifikan pada model.

2. Melakukan undersampling, yaitu mengambil 10000 sample data secara acak.
```python
ratings_sampled = ratings_data.sample(n=10000, random_state=42)
```
Alasan dilakukan tahapan ini:  
- Dikarenakan jumlah data yang cukup besar dan resource yang terbatas maka harus dilakukan undersampling untuk mengurangi dimensi data dan menghemat memori.

3. Memisahkan genre pada data dengan fungsi `split('|')` dan memasukkannya ke dalam TF-IDF
```python
movies_sampled['genres'] = movies_sampled['genres'].str.split('|')

movies_sampled['genres_str'] = movies_sampled['genres'].apply(lambda x: ' '.join(x))

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies_sampled['genres_str'])
```
Alasan dilakukan tahapan ini:  
- Split digunakan untuk memisahkan genre yang digabungkan dalam satu kolom menjadi elemen-elemen terpisah, dan TF-IDF digunakan untuk mengubah genre tersebut menjadi representasi numerik yang dapat digunakan untuk mengukur kesamaan antar item.

4. Melakukan encoding pada data user dan movie
```python
# Melakukan encoding userId
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}

# Melakukan proses encoding movieId
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}
```
Alasan dilakukan tahapan ini:

- Encoding pada data user dan movie bertujuan untuk mengubah ID pengguna dan ID film yang awalnya berupa data kategorikal menjadi numerik, sehingga dapat digunakan sebagai input pada model RecommenderNet.

5. Melakukan pengacakan data
```python
df = df.sample(frac=1, random_state=42)
```
Alasan dilakukan tahapan ini:

- Menghindari bias urutan data dan memastikan model tidak terpengaruh pola urutan tertentu. selain itu juga untuk meningkatkan generalisasi model dengan distribusi data yang lebih acak.

6. Melakukan split data menjadi set training dan validation 
```python
x = df[['user', 'movie']].values

y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
```
Alasan dilakukan tahapan ini:
- Tahapan ini dilakukan untuk melatih model dengan 80% data dan menguji kinerjanya pada 20% data sisanya untuk menghindari overfitting dan memastikan kemampuan generalisasi model.


## Modeling and Results

Pada tahap ini, dua pendekatan utama digunakan untuk merekomendasikan film, yaitu **Content-Based Filtering** dan **Collaborative Filtering**. Berikut penjelasan mengenai masing-masing metode dan hasil rekomendasi yang dihasilkan:

### 1. Content-Based Filtering

#### Tahapan dan Cara Kerja Content-Based Filtering:

1. **Persiapan Data:** Menggunakan _TF-IDF Vectorizer_ untuk mengubah genre film menjadi representasi vektor. Teknik ini mengonversi data teks (genre film) menjadi bentuk numerik yang dapat diproses oleh algoritma.
2. **Penghitungan Kesamaan:** Setelah data genre film diubah menjadi vektor, kesamaan antar film dihitung menggunakan _Cosine Similarity_. Ini mengukur sejauh mana dua film memiliki kesamaan dalam hal genre yang disukai pengguna.
3. **Rekomendasi Film:** Berdasarkan kesamaan tersebut, sistem merekomendasikan film yang memiliki genre serupa dengan film yang sudah disukai pengguna.

#### Hasil Rekomendasi Content-Based Filtering:

Berikut adalah 10 rekomendasi film teratas untuk film **Toy Story (1995)** berdasarkan pendekatan Content-Based Filtering:

| title                            | genres                                              | similarity_score |
|----------------------------------|-----------------------------------------------------|------------------|
| Antz (1998)                      | \[Adventure, Animation, Children, Comedy, Fantasy\] | 1.0              |
| Toy Story 2 (1999)               | \[Adventure, Animation, Children, Comedy, Fantasy\] | 1.0              |
| Emperor's New Groove, The (2000) | \[Adventure, Animation, Children, Comedy, Fantasy\] | 1.0              |
| Monsters, Inc. (2001)            | \[Adventure, Animation, Children, Comedy, Fantasy\] | 1.0              |
| Shrek the Third (2007)           | \[Adventure, Animation, Children, Comedy, Fantasy\] | 1.0              |
| Tale of Despereaux, The (2008)   | \[Adventure, Animation, Children, Comedy, Fantasy\] | 1.0              |
| Asterix and the Vikings (2006)   | \[Adventure, Animation, Children, Comedy, Fantasy\] | 1.0              |
| Boxtrolls, The (2014)            | \[Adventure, Animation, Children, Comedy, Fantasy\] | 1.0              |
| The Good Dinosaur (2015)         | \[Adventure, Animation, Children, Comedy, Fantasy\] | 1.0              |
| Moana (2016)                     | \[Adventure, Animation, Children, Comedy, Fantasy\] | 1.0              |

### 2. Collaborative Filtering

#### Tahapan dan Cara Kerja Collaborative Filtering:

1. **Data Encoding:** Mengubah `userId` dan `movieId` menjadi representasi angka (encoded) agar dapat digunakan dalam model neural network.
2. **Inisialisasi model**: Memasukan parameter num_user (9041), num_movie (3658), dan embedding_size (50) ke dalam fungsi RecommenderNet untuk mendefinisikan ukuran embedding pengguna dan film.
3. **Kompilasi Model**: Model dikompilasi menggunakan optimizer Adam dengan learning_rate=0.001, yang dikenal efisien dalam memperbarui bobot pada neural network. Fungsi loss binary cross-entropy digunakan karena cocok untuk mengukur kesalahan dalam klasifikasi biner, seperti pada rekomendasi berbasis interaksi pengguna.
4. **Pelatihan Model:** Model dilatih dengan epochs=100 dan batch_size=8 untuk memastikan model mempelajari pola interaksi pengguna secara mendalam tanpa menyebabkan overfitting.
5. **Rekomendasi Film:** Setelah model dilatih, sistem merekomendasikan film berdasarkan rating yang diprediksi untuk film yang belum dinilai oleh pengguna.

#### Hasil Rekomendasi Collaborative Filtering:

Berikut adalah 10 rekomendasi film teratas untuk pengguna dengan ID **145319** berdasarkan pendekatan Collaborative Filtering:

| title                             | genres                           |
|-----------------------------------|----------------------------------|
| Postman, The (Postino, Il) (1994) | \[Comedy, Drama, Romance\]       |
| Mr. Holland's Opus (1995)         | \[Drama\]                        |
| 39 Steps, The (1935)              | \[Drama, Mystery, Thriller\]     |
| Stalker (1979)                    | \[Drama, Mystery, Sci-Fi\]       |
| Cool Hand Luke (1967)             | \[Drama\]                        |
| High Noon (1952)                  | \[Drama, Western\]               |
| Rocky (1976)                      | \[Drama\]                        |
| Stepmom (1998)                    | \[Drama\]                        |
| Risky Business (1983)             | \[Comedy\]                       |
| Last Unicorn, The (1982)          | \[Animation, Children, Fantasy\] |

---

### Catatan:

- Pada **Content-Based Filtering**, rekomendasi didasarkan pada genre film yang serupa dengan film yang sudah disukai pengguna, tanpa mempertimbangkan interaksi dengan pengguna lain.
- Pada **Collaborative Filtering**, rekomendasi didasarkan pada perilaku pengguna lain yang memiliki pola rating serupa, sehingga dapat memberikan rekomendasi yang lebih personal.

Dengan kedua pendekatan ini, sistem rekomendasi dapat memberikan pilihan film yang relevan berdasarkan kesamaan konten dan interaksi pengguna.


## Evaluation

Metrik evaluasi yang digunakan dalam proyek ini adalah RMSE (Root Mean Square Error) dan MAE (Mean Absolute Error) untuk metode Collaborative Filtering dan Precision untuk metode berbasis Content-based Filtering.

### Content-Based Filtering:
- **Precision:** 100.00% 

### Collaborative Filtering:
- **RMSE:** 0.2613  
- **MAE:** 0.2111  

Hasil evaluasi ini menunjukkan performa dari masing-masing metode menggunakan metrik yang relevan. **Precision** untuk **Content-Based Filtering** menunjukkan tingkat akurasi rekomendasi yang sangat tinggi, sedangkan **Collaborative Filtering** menghasilkan nilai **RMSE** dan **MAE** yang memberikan gambaran mengenai kesalahan rata-rata dalam prediksi.

---

### Penjelasan Formula Metrik

#### 1. RMSE (Root Mean Squared Error)

**Formula:**

$$
\text{RMSE}(y, \hat{y}) = \sqrt{\frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{N}}
$$

**Penjelasan:**
- **$y$**: Nilai aktual (observasi nyata).
- **$ŷ$**: Nilai prediksi dari model.
- **$N$**: Jumlah data.
- **$(y_i - ŷ_i)$**: Selisih antara nilai aktual dan prediksi pada titik ke-i.
- **$(y_i - ŷ_i)²$**: Kuadrat dari selisih tersebut, untuk memastikan bahwa error negatif dan positif dihitung sebagai kesalahan.
- **$Σ (y_i - ŷ_i)²$**: Penjumlahan dari semua kuadrat kesalahan.
- **$N$**: Pembagi untuk mendapatkan rata-rata kesalahan kuadrat per data.
- **$√$**: Akar kuadrat untuk mengembalikan kesalahan ke skala aslinya.

**Cara Kerja:**  
RMSE mengukur seberapa besar perbedaan antara nilai prediksi dan nilai sebenarnya. Metrik ini memberikan penekanan lebih besar pada kesalahan yang besar karena kuadrat dari selisih. Nilai **RMSE yang lebih kecil** menunjukkan prediksi yang lebih akurat.

---

#### 2. MAE (Mean Absolute Error)

**Formula:**

$$
\text{MAPE} = \frac{1}{N} \sum_{i=1}^{N} \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100
$$

**Penjelasan:**
- **$y$**: Nilai aktual (observasi nyata).
- **$ŷ$**: Nilai prediksi dari model.
- **$N$**: Jumlah data.
- **$|y_i - ŷ_i|$**: Nilai absolut dari perbedaan antara nilai aktual dan nilai prediksi.
- **$(y_i)$**: Nilai aktual pada titik ke-i.
- **$× 100$**: Menghitung persentase kesalahan.

**Cara Kerja:**  
MAE menghitung rata-rata selisih absolut antara nilai aktual dan prediksi. Nilai **MAE yang lebih kecil** menunjukkan model yang lebih akurat dalam memprediksi nilai.

---

#### 3. Recommender System Precision

**Formula:**

$$
\text{Precision} = \frac{\text{Number of relevant recommendations}}{\text{Number of recommendations made}}
$$

**Penjelasan:**
- **Number of relevant recommendations**: Jumlah rekomendasi yang relevan.
- **Number of recommendations made**: Jumlah total rekomendasi yang diberikan oleh sistem.

**Cara Kerja:**  
Precision mengukur sejauh mana rekomendasi yang diberikan oleh sistem relevan dengan preferensi pengguna. Precision yang lebih tinggi menunjukkan bahwa semakin banyak rekomendasi yang relevan dari total rekomendasi yang diberikan.

---

### Kesimpulan Metrik:
- **RMSE** memberikan gambaran seberapa besar kesalahan rata-rata dalam satuan yang sama dengan data asli. Semakin kecil RMSE, semakin baik model.  
- **MAE** menunjukkan nilai deviasi rata-rata secara absolut. Metrik ini mudah dipahami dan memberikan indikasi akurasi prediksi.
- **Precision** mengukur sejauh mana rekomendasi yang diberikan oleh sistem relevan dengan preferensi pengguna. Semakin tinggi Precision, semakin banyak rekomendasi yang sesuai dengan kebutuhan pengguna di antara semua rekomendasi yang diberikan.

Ketiga metrik ini saling melengkapi untuk memberikan penilaian lengkap terhadap kualitas prediksi model dan akurasi rekomendasi. RMSE dan MAE fokus pada kesalahan prediksi numerik, sementara Precision berfokus pada relevansi rekomendasi dalam konteks sistem rekomendasi.