# Tubes2-AI

## Deskripsi Singkat
Repositori ini berisi implementasi _from scratch_ dari beberapa model yang ada pada library scikit-learn dan analisis terhadap dataset UNSW-NB15, yaitu dataset yang berisi raw network packets yang dibuat menggunakan IXIA PerfectStorm oleh Cyber Range Lab UNSW Canberra. Dataset ini terdiri dari 10 jenis aktivitas (9 jenis attack dan 1 aktivitas normal). Sembilan jenis attack yang termasuk ke dalam dataset ini adalah Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode dan Worms.


## Cara penggunaan
1. Pastikan python telah terinstall
2. Pindah ke folder `src` dan install semua library yang diperlukan dengan menjalankan command berikut:
   
   ```
   pip install -r requirements.txt
   ```
3. Pastikan semua features yang ada pada dataset sudah dalam format numerikal (ex: int, float) agar mencegah terjadinya error (perhatikan contoh penggunaan pada notebook).
4. ``` SANGAT DISARANKAN ``` untuk menjalankan notebook pada google colab.

Berikut adalah model yang diimplementasikan pada repositori beriikut: 

**Supervised Learning:**
- [v] KNN
- [v] Gaussian Naive Bayes
- [v] ID3
