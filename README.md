# TelematicsIoT

- lstm.py: Berisi definisi kelas arsitektur model (LSTMModel) menggunakan API MindSpore Cell. Berkas ini bertindak sebagai cetakan (blueprint) yang diimpor baik saat pelatihan maupun inferensi.

- notebook_training.ipynb: Notebook untuk training model prediksi parameter lingkungan. memuat data historis (data_bersih.csv), melakukan rekayasa fitur waktu, normalisasi data, pelatihan model secara batch, evaluasi metrik ($R^2$, MAE, RMSE), hingga penyimpanan bobot.

- model_lstm.ckpt: File checkpoint biner hasil ekspor dari MindSpore yang menyimpan seluruh parameter bobot (weights dan biases) yang telah dioptimalkan selama pelatihan.

- scaler.save: Berkas serialisasi MinMaxScaler dari scikit-learn yang diekspor menggunakan joblib. Berkas ini sangat krusial untuk menjaga konsistensi rentang angka transformasi antara fase training dan testing.

- SimulLstm.ipynb: Skrip otomatisasi yang berjalan terus menerus (infinite loop) untuk menyimulasikan pembacaan data sensor untuk prediksi 3 jam ke depan, dan mengunggah hasilnya kembali ke OBS.

- transformers.py: Berisi arsitektur utama model hybrid LSTMTransformer dan blok kustom MultiHeadAttention. Berkas ini mendefinisikan bagaimana data diekstraksi secara temporal dan struktural.

- training_transformers.ipynb: Notebook Training Model yang mencakup tahapan, melakukan pembersihan data kosong (data cleansing), normalisasi fitur dan target kontrol, pelatihan dengan Gradient Clipping, hingga evaluasi performa (MAE, RMSE, R^2).

- simulaktuator.ipynb: SSkrip otomatisasi yang berjalan terus menerus (infinite loop) untuk menyimulasikan penentuan state aktuator dan mengunggah hasilnya kembali ke OBS.
- lstm_transformer.ckpt / model.ckpt: Berkas biner checkpoint MindSpore yang menyimpan seluruh parameter bobot (weights & biases) hasil training yang siap digunakan di tahap produksi.

- scaler.save: Berkas serialisasi MinMaxScaler dari scikit-learn untuk mengunci parameter normalisasi agar konversi data sensor saat inferensi tidak fluktuatif.
