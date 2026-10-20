# RSSI-based proximity-zone classification of a Rogue Access Point: KNN vs SVM (simulation)

Kode, data, dan keluaran eksperimen untuk naskah Evaluasi Algoritma KNN dan SVM dalam Klasifikasi Zona Rogue Access Point Berbasis RSSI pada Data Simulasi Empat Access Point.

**English summary.** This repository contains the simulated RSSI dataset (10,201 grid points, 4 access points, 1 rogue AP), the data-generation script, the experiment script (eksperimen_revisi.py), and the logs, tables, and figures cited in the manuscript. The task is a proximity-zone classification (label = station within 25 m of the rogue AP), not detection of a rogue AP on unseen topologies.

## Struktur folder

```
.
├── 01_Simulation/
│   └── data_collector.py   # skrip asli yang menghasilkan dataset
├── 02_Dataset/
│   └── dataset_rssi.csv                          # dataset kanonik (10.201 baris)
├── 03_Machine_Learning/
│   └── eksperimen_revisi.py                      # seluruh eksperimen (Bagian 1-5)
├── 04_Results/
│   ├── logs/      hasil_bagian1.txt ... hasil_bagian5.txt
│   ├── tables/    hasil_bagian2_metrik.csv, hasil_bagian3_multiseed.csv
│   └── figures/   gambar_cm.png, gambar_roc.png, gambar_mcc_seed.png,
│                  gambar_galat_spasial.png.
├── requirements.txt
├── LICENSE
└── README.md
```

## Lingkungan yang dipakai

Ubuntu 22.04 LTS (VirtualBox), AMD Ryzen 3 3300U (2 core), RAM 7,8 GB, Python 3.10.12, scikit-learn 1.7.2, NumPy 1.26.4, pandas 2.3.3, SciPy 1.8.0, Mininet-WiFi 2.7 (hanya untuk skrip pembangkit data). Versi dicatat otomatis di bagian atas setiap berkas `hasil_bagian*.txt`.

## Menjalankan ulang

Dari folder `03_Machine_Learning` (keluaran CSV/PNG ditulis ke folder kerja saat ini):

```bash
pip install -r ../requirements.txt
python3 eksperimen_revisi.py --bagian 1 2>&1 | tee hasil_bagian1.txt                 # verifikasi data, pembanding, ablation fitur (detik)
python3 eksperimen_revisi.py --bagian 2 --k_default 3 2>&1 | tee hasil_bagian2.txt   # ablation S1-S4, McNemar, DeLong, waktu, galat spasial (~6-10 menit)
python3 eksperimen_revisi.py --bagian 3 2>&1 | tee hasil_bagian3.txt                 # 10 seed, split acak vs blok spasial (~15-30 menit)
python3 eksperimen_revisi.py --bagian 4 2>&1 | tee hasil_bagian4.txt                 # plateau k dan C x gamma, pembanding linear (~3-6 menit)
python3 eksperimen_revisi.py --bagian 5 2>&1 | tee hasil_bagian5.txt                 # waktu per konfigurasi dan gambar final (< 2 menit)
```

Opsi: `--data <path ke CSV>`, `--seeds <jumlah>`, `--csv3 <path hasil_bagian3_multiseed.csv>`. Bagian 5 memeriksa otomatis bahwa _confusion matrix_ sama dengan Bagian 2 (`COCOK`); bila `TIDAK COCOK`, jangan pakai gambarnya.

## Hubungan keluaran dengan naskah

| Elemen naskah                                    | Sumber                             |
| ------------------------------------------------ | ---------------------------------- |
| Validasi data (label, derau, grid)               | Bagian 1A                          |
| Tabel 1 (ablation S1-S4), uji McNemar dan DeLong | Bagian 2                           |
| Tabel 2 (rerata ± SD 10 seed), Gambar 5          | Bagian 3 (+ Bagian 5 untuk gambar) |
| Tabel 3 (pembanding sederhana)                   | Bagian 1C, 2, dan 4C               |
| Plateau hyperparameter                           | Bagian 4                           |
| Tabel 4 (waktu komputasi)                        | Bagian 5 (uji skala: Bagian 2)     |
| Gambar 3, 4                                      | Bagian 5                           |
| Gambar 6 (galat spasial)                         | Bagian 2                           |

## Catatan penting

- **Dataset kanonik.** Skrip pembangkit data tidak menetapkan _random seed_ untuk derau, sehingga menjalankannya ulang menghasilkan realisasi derau yang berbeda (statistiknya serupa, angkanya tidak identik). Angka pada naskah memakai `dataset_rssi.csv` di repositori ini.
- **RSSI dihitung analitis.** Nilai RSSI dihitung dengan model _log-distance_ (RSSI(1 m) = −30 dBm, n = 3) ditambah derau seragam ±3 dB. _Mininet-WiFi_ hanya membangun topologi dan menempatkan _station_.
- **Label berbasis zona.** `is_rogue = 1` bila jarak _station_ ke RAP (40, 10) kurang dari 25 m. Ini klasifikasi zona kedekatan pada satu topologi dan satu posisi RAP, bukan deteksi RAP pada topologi lain.
- **`ConvergenceWarning` pada SVM tanpa _scaling_ (S3) wajar**; iterasi dibatasi 500.000.
- **Waktu komputasi bergantung mesin.** Hanya selisih waktu latih KNN vs SVM yang konsisten antar-eksekusi.

## Sitasi dan lisensi

Sitasi: Naskah dalam proses penelaahan di Jurnal Informatika: Jurnal Pengembangan IT; rincian sitasi (volume, halaman, DOI) akan ditambahkan setelah terbit.
Lisensi: Kode dan dataset dirilis di bawah lisensi MIT (lihat berkas [LICENSE](LICENSE)).
