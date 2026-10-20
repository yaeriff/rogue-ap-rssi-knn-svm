#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore", message=".*NumPy version.*")

import argparse
import os
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, make_scorer,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                     GroupShuffleSplit, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Konstanta (sesuai script pembuat dataset)
AP = {"ap1": (10, 50), "ap2": (50, 50), "ap3": (90, 50), "ap4": (40, 10)}
RADIUS = 25.0
FEATS = ["rssi_ap1", "rssi_ap2", "rssi_ap3", "rssi_ap4"]
SEED = 42
MAXITER = 500000  # pengaman agar SVM linear/poly tanpa scaling tidak menggantung
mcc_scorer = make_scorer(matthews_corrcoef)


# Utilitas umum
def model_rssi(d):
    d = np.maximum(d, 1.0)
    return -30.0 - 30.0 * np.log10(d)  # RSSI(1 m) = -30 dBm, n = 3


def muat_data(path):
    df = pd.read_csv(path)
    for k, (ax, ay) in AP.items():
        df["d_" + k] = np.hypot(df["x"] - ax, df["y"] - ay)
    return df


def info_lingkungan(path):
    print("--- Lingkungan eksekusi ---")
    print("Waktu menjalankan :", datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("Sistem operasi    :", platform.platform())
    print("Python            :", platform.python_version())
    print("numpy / pandas / scipy / scikit-learn :",
          np.__version__, "/", pd.__version__, "/", scipy.__version__, "/",
          sklearn.__version__)
    cpu, ram = "?", "?"
    try:
        for baris in open("/proc/cpuinfo"):
            if baris.startswith("model name"):
                cpu = baris.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    try:
        for baris in open("/proc/meminfo"):
            if baris.startswith("MemTotal"):
                ram = "%.1f GB" % (int(baris.split()[1]) / 1024 / 1024)
                break
    except Exception:
        pass
    print("CPU / core / RAM  :", cpu, "/", os.cpu_count(), "core /", ram)
    try:
        hasil = subprocess.run(["mn", "--version"], capture_output=True,
                               text=True, timeout=10)
        print("Mininet           :", (hasil.stdout or hasil.stderr).strip())
    except Exception:
        print("Mininet           : (tidak terdeteksi, cek manual: mn --version)")
    try:
        from importlib.metadata import version
        print("Mininet-WiFi(pip) :", version("mininet-wifi"))
    except Exception:
        print("Mininet-WiFi      : (tidak terdeteksi via pip, cek manual)")
    tgl = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    print("Berkas dataset    :", os.path.abspath(path), "| terakhir diubah", tgl)
    print()


def skor(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def metrik(y, pred, s):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return dict(
        acc=accuracy_score(y, pred),
        prec=precision_score(y, pred, zero_division=0),
        rec=recall_score(y, pred),
        f1=f1_score(y, pred),
        f1_macro=f1_score(y, pred, average="macro"),
        mcc=matthews_corrcoef(y, pred),
        auc=roc_auc_score(y, s),
        prauc=average_precision_score(y, s),
        TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
        FPR=fp / (fp + tn), FNR=fn / (fn + tp),
    )


# Uji statistik
def mcnemar_exact(y, pa, pb):
    ca, cb = (pa == y), (pb == y)
    n11 = int((ca & cb).sum())
    n10 = int((ca & ~cb).sum())
    n01 = int((~ca & cb).sum())
    n00 = int((~ca & ~cb).sum())
    nd = n10 + n01
    p = 1.0 if nd == 0 else min(1.0, 2 * stats.binom.cdf(min(n10, n01), nd, 0.5))
    return n11, n10, n01, n00, p


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def delong_test(y, s1, s2):
    """Uji DeLong berpasangan untuk selisih AUC (Sun & Xu, 2014)."""
    y = np.asarray(y)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    n = len(y) - m
    preds = np.vstack((np.asarray(s1), np.asarray(s2)))[:, order]
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    l = np.array([1, -1])
    var = float(l @ cov @ l.T)
    if var <= 0:
        return aucs[0], aucs[1], float("nan"), float("nan")
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


# Model, grid, tuning
def buat_pipeline(algo, skala, **kw):
    langkah = []
    if skala:
        langkah.append(("scaler", StandardScaler()))
    clf = KNeighborsClassifier(**kw) if algo == "KNN" else SVC(max_iter=MAXITER, **kw)
    langkah.append(("clf", clf))
    return Pipeline(langkah)


def grid(algo, ringkas=False):
    if algo == "KNN":
        return {"clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 75, 101],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"]}
    cw = [None] if ringkas else [None, "balanced"]
    cs = [0.1, 1, 10, 100, 1000] if ringkas else [0.1, 1, 10, 100]
    g = [{"clf__kernel": ["rbf"], "clf__C": cs,
          "clf__gamma": [1, 0.1, 0.01, 0.001], "clf__class_weight": cw},
         {"clf__kernel": ["linear"], "clf__C": [0.1, 1, 10],
          "clf__class_weight": cw}]
    if not ringkas:
        g += [{"clf__kernel": ["sigmoid"], "clf__C": [0.1, 1, 10, 100],
               "clf__gamma": [1, 0.1, 0.01, 0.001], "clf__class_weight": cw},
              {"clf__kernel": ["poly"], "clf__degree": [2, 3],
               "clf__C": [1, 10], "clf__gamma": ["scale"],
               "clf__class_weight": cw}]
    return g


def tuning(algo, skala, Xtr, ytr, cv, ringkas=False, groups=None):
    gs = GridSearchCV(buat_pipeline(algo, skala), grid(algo, ringkas),
                      scoring=mcc_scorer, cv=cv, n_jobs=-1)
    if groups is None:
        gs.fit(Xtr, ytr)
    else:
        gs.fit(Xtr, ytr, groups=groups)
    return gs


def ukur_waktu(est, Xtr, ytr, Xte, ulang=5):
    ft, pt = [], []
    for _ in range(ulang):
        m = clone(est)
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        ft.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        m.predict(Xte)
        pt.append((time.perf_counter() - t0) / len(Xte))
    return float(np.median(ft) * 1000), float(np.median(pt) * 1000), m


def tampil(df_, desimal=4):
    print(df_.round(desimal).to_string(index=False))


# BAGIAN 1
def bagian1(df, args):
    print("=== BAGIAN 1: verifikasi dataset, replikasi angka lama, pembanding ===\n")
    print(f"A. Fakta dataset\n   Baris: {len(df)} | Rogue: {int(df.is_rogue.sum())} "
          f"({100 * df.is_rogue.mean():.2f}%)")
    cocok = ((df.d_ap4 < RADIUS).astype(int) == df.is_rogue).mean()
    print(f"   Label identik dengan aturan (jarak ke ap4 < {RADIUS:.0f} m) pada "
          f"{100 * cocok:.2f}% baris")
    print("   Langkah grid x:", np.unique(np.diff(np.sort(df.x.unique()))),
          "| langkah y:", np.unique(np.diff(np.sort(df.y.unique()))))
    print("\n   Karakter derau (RSSI di berkas - RSSI model log-distance n=3, -30 dBm @1 m):")
    baris = []
    for k in AP:
        r = df["rssi_" + k] - model_rssi(df["d_" + k])
        baris.append([k, r.mean(), r.std(), r.min(), r.max(), stats.kurtosis(r)])
    tampil(pd.DataFrame(baris, columns=["AP", "rerata", "SD", "min", "maks",
                                        "kurtosis_ekses"]), 3)
    print("   Pembanding: uniform(-3,3) -> SD 1,732; kurtosis ekses -1,2 | "
          "Gaussian -> kurtosis ekses 0")

    print("\nB. Replikasi protokol lama (TANPA stratify, seed 42, tanpa scaling)")
    Xtr, Xte, ytr, yte = train_test_split(df[FEATS], df.is_rogue, test_size=0.3,
                                          random_state=SEED)
    print(f"   Rogue di data uji: {int(yte.sum())} dari {len(yte)} "
          f"({100 * yte.mean():.2f}%)")
    for nama, m in [("KNN k=3", KNeighborsClassifier(3)),
                    ("KNN k=5", KNeighborsClassifier(5)),
                    ("SVM default", SVC())]:
        m.fit(Xtr, ytr)
        p, s = m.predict(Xte), skor(m, Xte)
        print(f"   {nama:12s} akurasi={100 * accuracy_score(yte, p):.2f}%  "
              f"MCC={matthews_corrcoef(yte, p):.4f}  AUC={roc_auc_score(yte, s):.4f}")
    print("   Target laporan lama: KNN 97,39% | SVM 97,88%")

    print("\nC. Pembanding sederhana & ablation fitur "
          "(protokol baru: stratified, seed 42)")
    Xtr, Xte, ytr, yte = train_test_split(df[FEATS], df.is_rogue, test_size=0.3,
                                          random_state=SEED, stratify=df.is_rogue)
    subset = {"rssi_ap4 saja": ["rssi_ap4"],
              "rssi_ap1-3 (tanpa ap4)": ["rssi_ap1", "rssi_ap2", "rssi_ap3"],
              "keempat fitur": FEATS}
    pembuat = {
        "Decision stump (depth 1)": lambda: DecisionTreeClassifier(max_depth=1, random_state=SEED),
        "Logistic Regression": lambda: Pipeline([("scaler", StandardScaler()),
                                                 ("clf", LogisticRegression(max_iter=1000))]),
        "Random Forest (200)": lambda: RandomForestClassifier(n_estimators=200,
                                                              random_state=SEED, n_jobs=-1),
    }
    hasil = []
    d = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    p = d.predict(Xte)
    hasil.append(["Dummy (selalu Normal)", "-", accuracy_score(yte, p),
                  f1_score(yte, p, zero_division=0), matthews_corrcoef(yte, p), 0.5])
    for sn, cols in subset.items():
        for nm, mk in pembuat.items():
            m = mk().fit(Xtr[cols], ytr)
            p, s = m.predict(Xte[cols]), skor(m, Xte[cols])
            hasil.append([nm, sn, accuracy_score(yte, p), f1_score(yte, p),
                          matthews_corrcoef(yte, p), roc_auc_score(yte, s)])
    tampil(pd.DataFrame(hasil, columns=["Model", "Fitur", "Akurasi", "F1_Rogue",
                                        "MCC", "AUC"]))
    st = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(Xtr[["rssi_ap4"]], ytr)
    print(f"\n   Ambang stump pada rssi_ap4 = {st.tree_.threshold[0]:.2f} dBm "
          f"(nilai teoretis tanpa derau: {model_rssi(RADIUS):.2f} dBm)")


# BAGIAN 2
def bagian2(df, args):
    print("=== BAGIAN 2: ablation 4 skenario, uji statistik, waktu, galat spasial ===\n")
    X, y = df[FEATS], df.is_rogue
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                          stratify=y)
    yv = yte.values
    print(f"Split stratified 70:30, seed {SEED}: latih={len(ytr)} (Rogue {int(ytr.sum())}), "
          f"uji={len(yte)} (Rogue {int(yte.sum())}, {100 * yte.mean():.2f}%)")
    print(f"Scoring GridSearchCV = MCC, 5-fold stratified, scaler berada di dalam Pipeline.")
    print(f"KNN default memakai k={args.k_default}.\n")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    skenario = [("S1 default, tanpa scaling", False, False),
                ("S2 default + scaling", False, True),
                ("S3 tuning, tanpa scaling", True, False),
                ("S4 tuning + scaling", True, True)]
    est, preds, scs, baris, param = {}, {}, {}, [], []
    for sk_nama, tune, skala in skenario:
        sk = sk_nama.split()[0]
        for algo in ("KNN", "SVM"):
            t0 = time.time()
            if tune:
                gs = tuning(algo, skala, Xtr, ytr, cv)
                m, bp, cvs = gs.best_estimator_, gs.best_params_, gs.best_score_
            else:
                kw = {"n_neighbors": args.k_default} if algo == "KNN" else {}
                m = buat_pipeline(algo, skala, **kw).fit(Xtr, ytr)
                bp, cvs = "(default)", float("nan")
            p, s = m.predict(Xte), skor(m, Xte)
            est[(algo, sk)], preds[(algo, sk)], scs[(algo, sk)] = m, p, s
            baris.append(dict(skenario=sk_nama, algo=algo, **metrik(yv, p, s)))
            param.append([sk, algo, cvs, bp])
            print(f"   selesai {sk} {algo} ({time.time() - t0:.0f} dtk)", flush=True)
    hasil = pd.DataFrame(baris)
    print("\n--- Tabel metrik (data uji) ---")
    tampil(hasil[["skenario", "algo", "acc", "prec", "rec", "f1", "f1_macro",
                  "mcc", "auc", "prauc"]])
    print("\n--- Confusion matrix & laju galat ---")
    tampil(hasil[["skenario", "algo", "TP", "TN", "FP", "FN", "FPR", "FNR"]])
    print("\n--- Parameter terbaik (MCC rerata 5-fold CV pada data latih) ---")
    for sk, algo, cvs, bp in param:
        print(f"   {sk} {algo}: CV-MCC={cvs:.4f} | {bp}")
    hasil.to_csv("hasil_bagian2_metrik.csv", index=False)

    print("\n--- McNemar eksak: KNN vs SVM per skenario (kesalahan prediksi berpasangan) ---")
    print("   n11=keduanya benar, n10=SVM benar & KNN salah, n01=SVM salah & KNN benar, n00=keduanya salah")
    for sk in ("S1", "S2", "S3", "S4"):
        n11, n10, n01, n00, p = mcnemar_exact(yv, preds[("SVM", sk)], preds[("KNN", sk)])
        print(f"   {sk}: n11={n11} n10={n10} n01={n01} n00={n00} | p={p:.4f}")

    print("\n--- McNemar eksak: dampak perlakuan pada algoritma yang sama (vs S1) ---")
    for algo in ("KNN", "SVM"):
        for sk in ("S2", "S3", "S4"):
            n11, n10, n01, n00, p = mcnemar_exact(yv, preds[(algo, "S1")], preds[(algo, sk)])
            print(f"   {algo} S1 vs {sk}: salah hanya di S1={n10}, salah hanya di {sk}={n01} | p={p:.4f}")

    print("\n--- Uji DeLong: selisih AUC KNN vs SVM ---")
    for sk in ("S1", "S4"):
        a1, a2, z, p = delong_test(yv, scs[("KNN", sk)], scs[("SVM", sk)])
        print(f"   {sk}: AUC KNN={a1:.4f} | AUC SVM={a2:.4f} | z={z:.3f} | p={p:.4f}")

    print("\n--- Waktu komputasi model S4 (median 5 kali; fit pada 7.140 data latih) ---")
    waktu = {}
    for algo in ("KNN", "SVM"):
        ft, pt, mfit = ukur_waktu(est[(algo, "S4")], Xtr, ytr, Xte)
        waktu[algo] = (ft, pt)
        extra = ""
        if algo == "KNN":
            extra = f" | algoritma pencarian tetangga: {getattr(mfit.named_steps['clf'], '_fit_method', 'auto')}"
        print(f"   {algo}: latih={ft:.2f} ms | inferensi={pt:.4f} ms/sampel "
              f"(batch {len(Xte)} sampel){extra}")

    print("\n--- Uji skala data latih (model S4, inferensi pada seluruh data uji) ---")
    baris = []
    for n in (441, 1000, 2500, 5000, len(Xtr)):
        if n < len(Xtr):
            Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=n, random_state=SEED,
                                            stratify=ytr)
        else:
            Xs, ys = Xtr, ytr
        r = [n]
        for algo in ("KNN", "SVM"):
            ft, pt, _ = ukur_waktu(est[(algo, "S4")], Xs, ys, Xte, ulang=3)
            r += [ft, pt]
        baris.append(r)
    tampil(pd.DataFrame(baris, columns=["n_latih", "KNN_latih_ms", "KNN_inf_ms/sampel",
                                        "SVM_latih_ms", "SVM_inf_ms/sampel"]))

    print("\n--- Analisis galat spasial (S4) ---")
    te = df.loc[Xte.index]
    galat = {}
    for algo in ("KNN", "SVM"):
        salah = preds[(algo, "S4")] != yv
        e = te[salah]
        jb = (e.d_ap4 - RADIUS).abs()
        galat[algo] = e
        print(f"   {algo}: FP={int((e.is_rogue == 0).sum())} FN={int((e.is_rogue == 1).sum())} | "
              f"jarak galat ke batas zona 25 m: median={jb.median():.2f} m, maks={jb.max():.2f} m, "
              f"<=3 m: {100 * (jb <= 3).mean():.1f}%, <=5 m: {100 * (jb <= 5).mean():.1f}%")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
        for ax, algo in zip(axs, ("KNN", "SVM")):
            ax.scatter(te.x, te.y, s=2, c="lightgrey", label="Data uji benar")
            e = galat[algo]
            fp, fn = e[e.is_rogue == 0], e[e.is_rogue == 1]
            ax.scatter(fp.x, fp.y, s=14, c="tab:red", label=f"FP ({len(fp)})")
            ax.scatter(fn.x, fn.y, s=14, c="tab:blue", label=f"FN ({len(fn)})")
            ax.add_patch(Circle(AP["ap4"], RADIUS, fill=False, ls="--", ec="black",
                                label="Batas zona 25 m"))
            for nm, (ax_, ay_) in AP.items():
                ax.plot(ax_, ay_, "k^" if nm != "ap4" else "r*", ms=9)
            ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
            ax.set_title(algo); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
            ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.savefig("gambar_galat_spasial.png", dpi=300)
        print("   Gambar disimpan: gambar_galat_spasial.png")
    except Exception as ex:
        print("   (gambar dilewati:", ex, ")")


# BAGIAN 3
def bagian3(df, args):
    print("=== BAGIAN 3: pengulangan multi-seed (grid ringkas: RBF+linear, tanpa class_weight) ===\n")
    X, y = df[FEATS], df.is_rogue
    blok = ((df.x // 10).astype(int) * 11 + (df.y // 10).astype(int))
    hasil = []
    for tipe in ("Acak-stratified", "Blok spasial 10x10 m"):
        for i in range(args.seeds):
            sd = SEED + i
            if tipe.startswith("Acak"):
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                                      random_state=sd, stratify=y)
                gtr, cv = None, StratifiedKFold(5, shuffle=True, random_state=sd)
            else:
                itr, ite = next(GroupShuffleSplit(1, test_size=0.3,
                                                  random_state=sd).split(X, y, groups=blok))
                Xtr, Xte, ytr, yte = X.iloc[itr], X.iloc[ite], y.iloc[itr], y.iloc[ite]
                gtr, cv = blok.iloc[itr].values, GroupKFold(5)
                if yte.sum() < 5 or ytr.sum() < 5:
                    print(f"   [{tipe}] seed {sd}: dilewati (Rogue terlalu sedikit)")
                    continue
            for algo in ("KNN", "SVM"):
                gs = tuning(algo, True, Xtr, ytr, cv, ringkas=True, groups=gtr)
                m = gs.best_estimator_
                p, s = m.predict(Xte), skor(m, Xte)
                hasil.append(dict(split=tipe, seed=sd, algo=algo, cv_mcc=gs.best_score_,
                                  rogue_uji=int(yte.sum()), params=str(gs.best_params_),
                                  **metrik(yte.values, p, s)))
            print(f"   selesai [{tipe}] seed {sd}", flush=True)
    H = pd.DataFrame(hasil)
    H.to_csv("hasil_bagian3_multiseed.csv", index=False)
    kolom = ["acc", "prec", "rec", "f1", "f1_macro", "mcc", "auc", "prauc"]
    print("\n--- Rerata +- SD antar seed (data uji) ---")
    for (tipe, algo), g in H.groupby(["split", "algo"]):
        print(f"\n   {tipe} | {algo} | n seed = {len(g)}")
        for k in kolom:
            print(f"      {k:9s}: {g[k].mean():.4f} +- {g[k].std(ddof=1):.4f}")
    print("\n--- Selisih berpasangan KNN - SVM pada MCC ---")
    for tipe, g in H.groupby("split"):
        a = g[g.algo == "KNN"].set_index("seed")["mcc"]
        b = g[g.algo == "SVM"].set_index("seed")["mcc"]
        dlt = (a - b).dropna()
        print(f"   {tipe}: rerata selisih={dlt.mean():+.4f} +- {dlt.std(ddof=1):.4f} | "
              f"KNN lebih tinggi pada {int((dlt > 0).sum())} dari {len(dlt)} seed")
    print("\n--- Parameter terpilih (frekuensi) ---")
    for (tipe, algo), g in H.groupby(["split", "algo"]):
        print(f"   {tipe} | {algo}:")
        for prm, c in Counter(g.params).most_common(3):
            print(f"      {c}x {prm}")

# Bagian 4
def bagian4(df, args):
    print("=== BAGIAN 4: plateau hyperparameter dan pembanding linear "
          "(seed 42, stratified, scaler di dalam Pipeline) ===\n")
    X, y = df[FEATS], df.is_rogue
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                          stratify=y)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

    print("A. KNN: sapuan k (CV 5-fold x 3 ulangan pada data latih; MCC uji hanya informasi)")
    ks = [5, 11, 21, 31, 51, 75, 101, 151, 201, 301]
    for w in ("uniform", "distance"):
        baris = []
        for k in ks:
            pipe = buat_pipeline("KNN", True, n_neighbors=k, weights=w)
            sc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring=mcc_scorer, n_jobs=-1)
            p = pipe.fit(Xtr, ytr).predict(Xte)
            baris.append([k, sc.mean(), sc.std(ddof=1), matthews_corrcoef(yte, p)])
        T = pd.DataFrame(baris, columns=["k", "CV_MCC", "SD_CV", "MCC_uji"])
        tb = T.loc[T.CV_MCC.idxmax()]
        se = tb.SD_CV / np.sqrt(len(sc))
        k1se = int(T[T.CV_MCC >= tb.CV_MCC - se].k.min())
        print(f"\n   weights = {w}")
        tampil(T)
        print(f"   k terbaik = {int(tb.k)} | k terkecil dalam 1 SE dari terbaik = {k1se}")

    print("\nB. SVM RBF: CV-MCC 5-fold pada kombinasi C x gamma (data latih)")
    cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
    Cs, gm = [1, 10, 100, 1000, 10000], [0.1, 0.03, 0.01, 0.003, 0.001]
    tab = pd.DataFrame(index=[f"C={c}" for c in Cs], columns=[f"g={g}" for g in gm],
                       dtype=float)
    for c in Cs:
        for g in gm:
            pipe = buat_pipeline("SVM", True, kernel="rbf", C=c, gamma=g)
            tab.loc[f"C={c}", f"g={g}"] = cross_val_score(
                pipe, Xtr, ytr, cv=cv5, scoring=mcc_scorer, n_jobs=-1).mean()
        print(f"   selesai C={c}", flush=True)
    print(tab.round(4).to_string())
    bi = tab.stack().idxmax()
    print(f"   Terbaik: {bi[0]}, {bi[1]} -> CV-MCC={tab.stack().max():.4f}")

    print("\nC. Pembanding linear (CV 5-fold x 3 ulangan; MCC uji)")
    def lr():
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000))])
    kandidat = {
        "Logistic Regression, rssi_ap4 saja": (["rssi_ap4"], lr()),
        "Logistic Regression, 4 fitur": (FEATS, lr()),
        "SVM linear (C=1), rssi_ap4 saja": (["rssi_ap4"], buat_pipeline("SVM", True, kernel="linear", C=1)),
        "SVM linear (C=1), 4 fitur": (FEATS, buat_pipeline("SVM", True, kernel="linear", C=1)),
    }
    for nama, (cols, pipe) in kandidat.items():
        sc = cross_val_score(pipe, Xtr[cols], ytr, cv=cv, scoring=mcc_scorer, n_jobs=-1)
        p = clone(pipe).fit(Xtr[cols], ytr).predict(Xte[cols])
        print(f"   {nama:38s} CV-MCC={sc.mean():.4f} +- {sc.std(ddof=1):.4f} | "
              f"MCC uji={matthews_corrcoef(yte, p):.4f}")


# Konfigurasi final model S4 (hasil Bagian 2, seed 42) dan harapan confusion matrix
PARAM_KNN = dict(n_neighbors=75, weights="uniform", metric="manhattan")
PARAM_SVM = dict(kernel="rbf", C=10, gamma=0.1)
HARAPAN = {"KNN": (405, 2596, 22, 38), "SVM": (412, 2593, 25, 31)}  # TP, TN, FP, FN


def bagian5(df, args):
    print("=== BAGIAN 5: waktu komputasi menurut konfigurasi + gambar final (300 dpi) ===\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from sklearn.metrics import roc_curve
    koma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

    X, y = df[FEATS], df.is_rogue
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED,
                                          stratify=y)
    yv = yte.values
    final = {"KNN": buat_pipeline("KNN", True, **PARAM_KNN),
             "SVM": buat_pipeline("SVM", True, **PARAM_SVM)}
    sc, pr = {}, {}
    print("A. Kesesuaian dengan Bagian 2 (S4, seed 42)")
    semua_cocok = True
    for nama, m in final.items():
        m.fit(Xtr, ytr)
        pr[nama], sc[nama] = m.predict(Xte), skor(m, Xte)
        tn, fp, fn, tp = confusion_matrix(yv, pr[nama]).ravel()
        ok = (int(tp), int(tn), int(fp), int(fn)) == HARAPAN[nama]
        semua_cocok &= ok
        print(f"   {nama}: TP={tp} TN={tn} FP={fp} FN={fn} | harapan {HARAPAN[nama]} -> "
              f"{'COCOK' if ok else 'TIDAK COCOK'}")
    if not semua_cocok:
        print("   PERINGATAN: gambar TIDAK boleh dipakai sebelum penyebab ketidakcocokan jelas.")

    print("\nB. Waktu komputasi menurut konfigurasi (median 7 kali; latih pada "
          f"{len(Xtr)} data; inferensi batch {len(Xte)} sampel; latensi = 1 sampel, median 200 panggilan)")
    konf = [("KNN k=21 euclidean", "KNN", dict(n_neighbors=21, weights="uniform", metric="euclidean")),
            ("KNN k=31 euclidean", "KNN", dict(n_neighbors=31, weights="uniform", metric="euclidean")),
            ("KNN k=75 manhattan (S4)", "KNN", PARAM_KNN),
            ("KNN k=101 euclidean", "KNN", dict(n_neighbors=101, weights="uniform", metric="euclidean")),
            ("SVM RBF C=10 gamma=0,1 (S4)", "SVM", PARAM_SVM)]
    baris = []
    for nama, algo, kw in konf:
        ft, pt, m = ukur_waktu(buat_pipeline(algo, True, **kw), Xtr, ytr, Xte, ulang=7)
        lat = []
        for i in range(200):
            xi = Xte.iloc[[i]]
            t0 = time.perf_counter()
            m.predict(xi)
            lat.append(time.perf_counter() - t0)
        clf = m.named_steps["clf"]
        info = int(clf.n_support_.sum()) if algo == "SVM" else getattr(clf, "_fit_method", "auto")
        baris.append([nama, ft, pt, float(np.median(lat) * 1000), info,
                      matthews_corrcoef(yv, m.predict(Xte))])
    tampil(pd.DataFrame(baris, columns=["Konfigurasi", "latih_ms", "inferensi_ms/sampel",
                                        "latensi_1sampel_ms", "n_SV/algoritma_KNN", "MCC_uji"]))

    # --- Gambar 1: confusion matrix ---
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.9))
    judul = {"KNN": "KNN (k=75, Manhattan)", "SVM": "SVM (RBF, C=10, \u03b3=0,1)"}
    for ax, nama in zip(axs, ("KNN", "SVM")):
        cm = confusion_matrix(yv, pr[nama])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                pct = 100 * cm[i, j] / cm[i].sum()
                ax.text(j, i, f"{cm[i, j]:,}".replace(",", ".") + f"\n({pct:.1f}%)".replace(".", ","),
                        ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Rogue"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Rogue"])
        ax.set_xlabel("Kelas prediksi"); ax.set_ylabel("Kelas sebenarnya")
        ax.set_title(judul[nama], fontsize=10)
    fig.tight_layout()
    fig.savefig("gambar_cm.png", dpi=300)
    plt.close(fig)

    # --- Gambar 2: kurva ROC (utuh + perbesaran) ---
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    warna = {"KNN": "tab:blue", "SVM": "tab:orange"}
    for ax, zoom in zip(axs, (False, True)):
        for nama in ("KNN", "SVM"):
            fpr, tpr, _ = roc_curve(yv, sc[nama])
            auc = roc_auc_score(yv, sc[nama])
            ax.plot(fpr, tpr, color=warna[nama], lw=1.8,
                    label=f"{nama} (AUC = {auc:.4f})".replace(".", ","))
        if not zoom:
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Acak (AUC = 0,5)")
            ax.set_title("Kurva ROC", fontsize=10)
        else:
            ax.set_xlim(0, 0.1); ax.set_ylim(0.85, 1.005)
            ax.set_title("Perbesaran (FPR 0\u20130,1)", fontsize=10)
        ax.set_xlabel("Laju positif palsu (FPR)"); ax.set_ylabel("Laju positif benar (TPR)")
        ax.xaxis.set_major_formatter(koma); ax.yaxis.set_major_formatter(koma)
        ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("gambar_roc.png", dpi=300)
    plt.close(fig)
    print("\nC. Gambar tersimpan: gambar_cm.png, gambar_roc.png")

    # --- Gambar 3: MCC antar-seed (dari hasil Bagian 3) ---
    if os.path.exists(args.csv3):
        H = pd.read_csv(args.csv3)
        urut = [t for t in ["Acak-stratified", "Blok spasial 10x10 m"] if t in set(H.split)]
        nama_x = {"Acak-stratified": "Acak-stratified", "Blok spasial 10x10 m": "Blok spasial (10\u00d710 m)"}
        fig, ax = plt.subplots(figsize=(6.2, 3.9))
        rng = np.random.RandomState(0)
        for gi, tipe in enumerate(urut, start=1):
            for algo, off in (("KNN", -0.2), ("SVM", 0.2)):
                v = H[(H.split == tipe) & (H.algo == algo)].mcc.values
                bp = ax.boxplot(v, positions=[gi + off], widths=0.3, showfliers=False,
                                patch_artist=True)
                for patch in bp["boxes"]:
                    patch.set_facecolor(warna[algo]); patch.set_alpha(0.45)
                for med in bp["medians"]:
                    med.set_color("black")
                ax.scatter(gi + off + rng.uniform(-0.06, 0.06, len(v)), v, s=14,
                           color=warna[algo], edgecolor="black", linewidth=0.4, zorder=3)
        ax.set_xticks(range(1, len(urut) + 1)); ax.set_xticklabels([nama_x[t] for t in urut])
        ax.set_xlabel("Skema pembagian data"); ax.set_ylabel("MCC (data uji)")
        ax.yaxis.set_major_formatter(koma); ax.grid(axis="y", alpha=0.3)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=warna[a], alpha=0.6, label=a) for a in ("KNN", "SVM")],
                  loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig("gambar_mcc_seed.png", dpi=300)
        plt.close(fig)
        print("   Gambar tersimpan: gambar_mcc_seed.png")
    else:
        print(f"   (gambar_mcc_seed.png dilewati: {args.csv3} tidak ditemukan)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bagian", choices=["1", "2", "3", "4", "5"], required=True)
    ap.add_argument("--data", default="../02_Dataset/dataset_rssi.csv")
    ap.add_argument("--k_default", type=int, default=5,
                    help="k KNN pada skenario default (ganti 3 bila baseline lama memakai k=3)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--csv3", default="hasil_bagian3_multiseed.csv",
                    help="CSV keluaran Bagian 3 (untuk gambar MCC antar-seed)")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        raise SystemExit(f"Berkas {args.data} tidak ditemukan. Jalankan dari folder "
                         f"03_Machine_Learning atau beri --data <path>.")
    df = muat_data(args.data)
    info_lingkungan(args.data)
    print("--- Konfigurasi grid pada eksekusi ini ---")
    print("KNN n_neighbors        :", grid("KNN")["clf__n_neighbors"])
    print("SVM RBF C (grid penuh) :", grid("SVM")[0]["clf__C"])
    print("SVM RBF C (ringkas, Bagian 3):", grid("SVM", ringkas=True)[0]["clf__C"])
    print()
    {"1": bagian1, "2": bagian2, "3": bagian3, "4": bagian4, "5": bagian5}[args.bagian](df, args)


if __name__ == "__main__":
    main()
