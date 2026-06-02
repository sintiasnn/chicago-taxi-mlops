# Chicago Taxi Tip Prediction — MLOps Pipeline

Nama: sintiawati

Username dicoding: sintiawati

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Chicago Taxi Trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) |
| Masalah | Memprediksi apakah seorang penumpang taksi akan memberikan tip berdasarkan karakteristik perjalanan, seperti jarak tempuh, durasi, tarif, metode pembayaran, dan perusahaan taksi. |
| Solusi machine learning | End-to-end MLOps pipeline menggunakan TensorFlow Extended (TFX) dengan Apache Beam orchestrator. Model dideploy menggunakan TensorFlow Serving (C++), dimonitoring dengan Prometheus + Grafana, dan hyperparameter di-tuning otomatis dengan Keras Tuner. |
| Metode pengolahan | Fitur numerik (trip_miles, fare, trip_seconds, trip_start_timestamp) dikonversi ke float32. Fitur kategorikal (payment_type, company, trip_start_hour, trip_start_day, trip_start_month) dikonversi ke int64 via tf.Example. |
| Arsitektur model | 4 fitur numerik + embedding untuk 5 fitur kategorikal (16 dimensi). Dense layers: 384 → 64 → 128 unit, masing-masing dengan BatchNormalization + Dropout 0.2. Output sigmoid. Hyperparameter dituning dengan Keras Tuner. |
| Metrik evaluasi | Binary Accuracy, AUC |
| Performa model | Model diblessing oleh TFX Evaluator dan dipush ke serving. Credit Card → prob tip ~0.98, Cash → prob tip ~0.02. |
| Opsi deployment | Railway + TensorFlow Serving (C++) via Docker image `tensorflow/serving`. Input format: serialized tf.Example (base64). REST API di port 8501. |
| Web app | [chicago-taxi-mlops](https://chicago-taxi-mlops-production.up.railway.app) |
| Monitoring | Prometheus scrape TF Serving metrics (`:tensorflow:serving:request_count`, `request_latency`), divisualisasikan dengan Grafana dashboard. Traffic generator script mengirim request setiap 5 detik. |
