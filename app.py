import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

st.set_page_config(page_title="Customer Segmentation | RFM", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Helper kompatibel lintas versi Streamlit
# ---------------------------------------------------------------------------
def show_df(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def show_image(path, caption=None):
    try:
        st.image(str(path), caption=caption, width="stretch")
    except Exception:
        st.image(str(path), caption=caption, use_container_width=True)


# ---------------------------------------------------------------------------
# Helper pembaca file
# ---------------------------------------------------------------------------
@st.cache_data
def load_csv(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError):
        return None


def load_experiment_summary():
    path = OUTPUT_DIR / "experiment_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def find_col(df, *candidates, contains=None):
    """Cari nama kolom (tanpa peduli huruf besar/kecil) dari daftar kandidat."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if contains:
        for c in df.columns:
            if any(k in c.lower() for k in contains):
                return c
    return None


def best_row(df):
    sil = find_col(df, "silhouette", "silhouette_score")
    if sil is None or df[sil].dropna().empty:
        return None
    return df.loc[df[sil].idxmax()]


def row_label(prefix, row):
    parts = []
    for key in ["k", "n_clusters", "linkage", "eps", "min_samples", "min_cluster_size"]:
        for c in row.index:
            if c.lower() == key and pd.notna(row[c]):
                val = row[c]
                if isinstance(val, (float, np.floating)):
                    val = int(val) if float(val).is_integer() else round(float(val), 3)
                parts.append(f"{key}={val}")
    return f"{prefix} ({', '.join(parts)})" if parts else prefix


# ---------------------------------------------------------------------------
# HALAMAN 1: OVERVIEW DASHBOARD
# ---------------------------------------------------------------------------
def section_comparison():
    st.header("1. Perbandingan Model")
    sources = [
        ("K-Means", "results_kmeans.csv"),
        ("Agglomerative", "results_agglomerative.csv"),
        ("DBSCAN", "results_dbscan.csv"),
        ("HDBSCAN", "results_hdbscan.csv"),
    ]
    rows, frames = [], {}
    for name, fname in sources:
        df = load_csv(fname)
        if df is None:
            continue
        frames[name] = (fname, df)
        row = best_row(df)
        if row is None:
            continue

        def val(*cands, contains=None):
            c = find_col(df, *cands, contains=contains)
            return row[c] if c is not None else np.nan

        rows.append({
            "Model": row_label(name, row),
            "Silhouette": val("silhouette", "silhouette_score"),
            "Davies-Bouldin": val("davies_bouldin", "davies_bouldin_score"),
            "Calinski-Harabasz": val("calinski_harabasz", "calinski_harabasz_score"),
            "Noise (%)": val("noise_percentage", "noise_pct", "noise_percent", contains=["noise"]),
        })

    if not frames:
        st.warning("File results_*.csv tidak ditemukan di folder outputs/.")
        return

    st.markdown(
        "Konfigurasi terbaik tiap algoritma dipilih berdasarkan Silhouette tertinggi. "
        "Silhouette dan Calinski-Harabasz yang lebih tinggi umumnya lebih baik, "
        "sedangkan Davies-Bouldin yang lebih rendah umumnya lebih baik."
    )

    if rows:
        comparison = pd.DataFrame(rows)
        show_df(comparison)
        metric_options = [
            m for m in ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Noise (%)"]
            if comparison[m].notna().any()
        ]
        if metric_options:
            metric = st.selectbox("Pilih metrik untuk grafik perbandingan", metric_options)
            fig = px.bar(
                comparison.dropna(subset=[metric]),
                x="Model", y=metric, color="Model",
                title=f"Perbandingan model berdasarkan {metric}",
                text_auto=".4f",
            )
            fig.update_layout(showlegend=False)
            show_chart(fig)
    else:
        st.info("Kolom silhouette tidak ditemukan, sehingga ringkasan model terbaik tidak dapat dibuat.")

    with st.expander("Lihat semua hasil eksperimen per algoritma"):
        for name, (fname, df) in frames.items():
            st.markdown(f"**{name}** — `{fname}`")
            show_df(df)
            st.download_button(
                f"Unduh {fname}",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=fname,
                mime="text/csv",
                key=f"dl_{fname}",
            )


def section_profile():
    st.header("2. Profil Segmen Pelanggan")
    profile = load_csv("profile_final.csv")
    if profile is None:
        st.warning("profile_final.csv tidak ditemukan atau tidak dapat dibaca.")
        return

    show_df(profile)

    cluster_col = find_col(profile, "Cluster", "Segment", "Segmen", "Label",
                           contains=["cluster", "segment", "segmen", "label"])
    count_col = find_col(profile, "Customer_Count", "Count", "Jumlah",
                         contains=["count", "jumlah", "size"])

    plot_df = profile.copy()
    if cluster_col is not None:
        plot_df[cluster_col] = plot_df[cluster_col].astype(str)

    if cluster_col is not None and count_col is not None and pd.api.types.is_numeric_dtype(plot_df[count_col]):
        show_chart(px.pie(plot_df, names=cluster_col, values=count_col,
                          title="Distribusi pelanggan per segmen", hole=0.4))

    numeric_cols = plot_df.select_dtypes(include="number").columns.tolist()
    rfm_cols = [c for c in numeric_cols
                if any(k in c.lower() for k in ["recency", "frequency", "monetary"])
                and "median" not in c.lower()]
    default = rfm_cols if rfm_cols else numeric_cols[:3]
    if cluster_col is not None and numeric_cols:
        chosen = st.multiselect("Kolom numerik untuk grafik perbandingan", numeric_cols, default=default)
        if chosen:
            show_chart(px.bar(plot_df, x=cluster_col, y=chosen, barmode="group",
                              title="Perbandingan nilai per segmen",
                              labels={"value": "Nilai", "variable": "Variabel"}))

    holdout = load_csv("profile_validation_holdout.csv")
    if holdout is not None:
        with st.expander("Validasi holdout (profile_validation_holdout.csv)"):
            show_df(holdout)


def section_supporting():
    st.header("3. Data Pendukung")
    items = [
        ("Benchmark eta² (benchmark_eta2.csv)", "benchmark_eta2.csv"),
        ("Alur pembersihan data (data_cleaning_funnel.csv)", "data_cleaning_funnel.csv"),
        ("Perbandingan linkage (results_linkage_comparison.csv)", "results_linkage_comparison.csv"),
        ("Hasil pooled skenario B (results_pooled_valid_scenarioB.csv)", "results_pooled_valid_scenarioB.csv"),
    ]
    found = False
    for title, fname in items:
        df = load_csv(fname)
        if df is not None:
            found = True
            with st.expander(title):
                show_df(df)

    seg = load_csv("rfm_segments.csv")
    if seg is not None:
        found = True
        with st.expander(f"Data segmentasi pelanggan (rfm_segments.csv) — {len(seg):,} baris"):
            show_df(seg.head(200))
            st.caption("Menampilkan 200 baris pertama.")
            st.download_button(
                "Unduh rfm_segments.csv",
                data=seg.to_csv(index=False).encode("utf-8"),
                file_name="rfm_segments.csv",
                mime="text/csv",
                key="dl_rfm_segments",
            )

    summary = load_experiment_summary()
    if summary is not None:
        found = True
        with st.expander("Ringkasan eksperimen (experiment_summary.json)"):
            st.json(summary, expanded=False)

    if not found:
        st.info("Belum ada file pendukung di folder outputs/.")


def section_figures():
    st.header("4. Galeri Visualisasi")
    if not FIGURES_DIR.exists():
        st.info("Folder outputs/figures/ tidak ditemukan.")
        return
    images = sorted(p for p in FIGURES_DIR.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        st.info("Tidak ada gambar di outputs/figures/.")
        return
    names = [p.name for p in images]
    chosen = st.multiselect("Pilih gambar", names, default=names[:2])
    for i in range(0, len(chosen), 2):
        cols = st.columns(2)
        for col, name in zip(cols, chosen[i:i + 2]):
            with col:
                show_image(FIGURES_DIR / name, caption=name)


def render_overview():
    st.title("📊 Customer Segmentation Dashboard")
    st.caption(
        "Perbandingan K-Means, Agglomerative Clustering, dan DBSCAN "
        "untuk segmentasi pelanggan berbasis RFM"
    )

    if not OUTPUT_DIR.exists():
        st.error(f"Folder outputs/ tidak ditemukan di {BASE_DIR}. Pastikan app.py sejajar dengan folder outputs/.")
        return

    seg = load_csv("rfm_segments.csv")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baris data segmentasi", f"{len(seg):,}" if seg is not None else "—")
    cl_col = find_col(seg, "Cluster", "Segment", contains=["cluster", "segment", "label"]) if seg is not None else None
    c2.metric("Jumlah label segmen", seg[cl_col].nunique() if cl_col else "—")
    n_files = len([p for p in OUTPUT_DIR.glob("*.csv")])
    c3.metric("File hasil CSV", n_files)

    st.divider()
    for section in (section_comparison, section_profile, section_supporting, section_figures):
        try:
            section()
        except Exception as exc:  # satu bagian gagal tidak mematikan halaman
            st.error(f"Bagian '{section.__name__}' gagal ditampilkan: {exc}")
        st.divider()

    st.header("5. Interpretasi Bisnis")
    st.subheader("Active / High-Value Customers")
    st.markdown(
        "- Pertahankan pelanggan melalui program loyalitas.\n"
        "- Tawarkan promosi yang relevan berdasarkan riwayat pembelian.\n"
        "- Berikan penghargaan kepada pelanggan dengan frekuensi dan nilai transaksi tinggi."
    )
    st.subheader("Inactive / Low-Value Customers")
    st.markdown(
        "- Jalankan kampanye reaktivasi pelanggan.\n"
        "- Berikan voucher atau penawaran khusus secara terukur.\n"
        "- Gunakan pengingat dan rekomendasi produk untuk mendorong pembelian ulang."
    )
    st.caption(
        "Catatan: nama segmen merupakan interpretasi berdasarkan karakteristik RFM. "
        "Label cluster adalah label teknis dan tidak menunjukkan urutan nilai."
    )


# ---------------------------------------------------------------------------
# HALAMAN 2: EXPLORE DATASET
# ---------------------------------------------------------------------------
def read_uploaded(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def build_rfm_from_transactions(df, cust_col, date_col, inv_col, qty_col, price_col):
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[qty_col] = pd.to_numeric(work[qty_col], errors="coerce")
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work = work.dropna(subset=[cust_col, date_col, qty_col, price_col])
    work = work[(work[qty_col] > 0) & (work[price_col] > 0)]
    if work.empty:
        return None
    work["_amount"] = work[qty_col] * work[price_col]
    snapshot = work[date_col].max() + pd.Timedelta(days=1)
    g = work.groupby(cust_col)
    rfm = pd.DataFrame({
        "Recency": (snapshot - g[date_col].max()).dt.days,
        "Frequency": g[inv_col].nunique() if inv_col else g[date_col].nunique(),
        "Monetary": g["_amount"].sum(),
    }).reset_index().rename(columns={cust_col: "CustomerID"})
    return rfm


def build_profile(df, label_col):
    prof = df.groupby(label_col).agg(
        Customer_Count=("Recency", "size"),
        Recency_Mean=("Recency", "mean"),
        Recency_Median=("Recency", "median"),
        Frequency_Mean=("Frequency", "mean"),
        Frequency_Median=("Frequency", "median"),
        Monetary_Mean=("Monetary", "mean"),
        Monetary_Median=("Monetary", "median"),
    ).reset_index()
    prof["Customer_Percentage"] = prof["Customer_Count"] / prof["Customer_Count"].sum() * 100
    return prof


def run_clustering(rfm, algo, params, use_log):
    values = rfm[["Recency", "Frequency", "Monetary"]].astype(float).clip(lower=0)
    if use_log:
        values = np.log1p(values)
    X = StandardScaler().fit_transform(values)

    if algo == "K-Means":
        model = KMeans(n_clusters=params["k"], n_init=10, random_state=42)
    elif algo == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=params["k"], linkage=params["linkage"])
    else:
        model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    labels = model.fit_predict(X)

    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    metrics = {"Jumlah cluster": n_clusters, "Noise (%)": float((~mask).mean() * 100),
               "Silhouette": None, "Davies-Bouldin": None, "Calinski-Harabasz": None}
    if n_clusters >= 2 and mask.sum() > n_clusters:
        Xm, lm = X[mask], labels[mask]
        sample = 10000 if len(Xm) > 10000 else None
        metrics["Silhouette"] = float(silhouette_score(Xm, lm, sample_size=sample, random_state=42))
        metrics["Davies-Bouldin"] = float(davies_bouldin_score(Xm, lm))
        metrics["Calinski-Harabasz"] = float(calinski_harabasz_score(Xm, lm))
    return labels, metrics


def fmt(v, f):
    return "—" if v is None else format(v, f)


def render_explore_dataset():
    st.title("🔎 Explore Dataset")
    st.caption("Unggah dataset, pilih algoritma clustering, jalankan eksperimen, lalu unduh hasilnya.")

    st.header("1. Unggah Data")
    uploaded = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])
    if uploaded is None:
        st.info("Unggah dataset untuk memulai.")
        return
    try:
        raw = read_uploaded(uploaded)
    except Exception as exc:
        st.error(f"File tidak dapat dibaca: {exc}")
        return

    st.write(f"{len(raw):,} baris, {raw.shape[1]} kolom")
    show_df(raw.head(20))
    cols = raw.columns.tolist()
    mode = st.radio("Jenis data", ["Data RFM siap pakai", "Data transaksi mentah"], horizontal=True)

    st.header("2. Pilih Kolom")
    if mode == "Data RFM siap pakai":
        c1, c2, c3, c4 = st.columns(4)
        id_col = c1.selectbox("ID pelanggan (opsional)", ["(tidak ada)"] + cols)
        r_col = c2.selectbox("Recency", cols)
        f_col = c3.selectbox("Frequency", cols, index=min(1, len(cols) - 1))
        m_col = c4.selectbox("Monetary", cols, index=min(2, len(cols) - 1))
        if len({r_col, f_col, m_col}) < 3:
            st.warning("Kolom Recency, Frequency, dan Monetary harus berbeda.")
            return
        rfm = pd.DataFrame({
            "Recency": pd.to_numeric(raw[r_col], errors="coerce"),
            "Frequency": pd.to_numeric(raw[f_col], errors="coerce"),
            "Monetary": pd.to_numeric(raw[m_col], errors="coerce"),
        })
        if id_col != "(tidak ada)":
            rfm.insert(0, "CustomerID", raw[id_col].values)
        rfm = rfm.dropna(subset=["Recency", "Frequency", "Monetary"]).reset_index(drop=True)
    else:
        c1, c2, c3 = st.columns(3)
        cust_col = c1.selectbox("ID pelanggan", cols)
        date_col = c2.selectbox("Tanggal transaksi", cols)
        inv_col = c3.selectbox("Nomor invoice (opsional)", ["(tidak ada)"] + cols)
        c4, c5 = st.columns(2)
        qty_col = c4.selectbox("Quantity", cols)
        price_col = c5.selectbox("Harga satuan", cols)
        if len({cust_col, date_col, qty_col, price_col}) < 4:
            st.warning("Pilih kolom yang berbeda untuk setiap peran.")
            return
        rfm = build_rfm_from_transactions(
            raw, cust_col, date_col, None if inv_col == "(tidak ada)" else inv_col, qty_col, price_col)

    if rfm is None or rfm.empty:
        st.error("Tidak ada data valid setelah pembersihan. Periksa pemilihan kolom.")
        return
    st.success(f"Data RFM siap: {len(rfm):,} pelanggan/baris.")
    with st.expander("Lihat tabel RFM"):
        show_df(rfm.head(100))

    st.header("3. Pengaturan Clustering")
    algo = st.selectbox("Algoritma", ["K-Means", "Agglomerative", "DBSCAN"])
    use_log = st.checkbox("Transformasi log1p sebelum standardisasi", value=True)
    params = {}
    if algo in ("K-Means", "Agglomerative"):
        params["k"] = st.slider("Jumlah cluster (k)", 2, 10, 2)
    if algo == "Agglomerative":
        params["linkage"] = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
        if len(rfm) > 15000:
            st.warning("Agglomerative pada data besar lambat dan boros memori. Pertimbangkan K-Means.")
    if algo == "DBSCAN":
        c1, c2 = st.columns(2)
        params["eps"] = c1.slider("eps", 0.05, 3.0, 0.45, 0.05)
        params["min_samples"] = c2.slider("min_samples", 2, 50, 5)

    if st.button("Jalankan clustering", type="primary"):
        with st.spinner("Menjalankan clustering..."):
            try:
                labels, metrics = run_clustering(rfm, algo, params, use_log)
            except Exception as exc:
                st.error(f"Clustering gagal: {exc}")
                return
        st.session_state["explore_result"] = {
            "rfm": rfm.assign(Cluster=labels), "metrics": metrics, "algo": algo, "params": params}

    result = st.session_state.get("explore_result")
    if result is None:
        return

    st.header("4. Hasil")
    seg, metrics = result["rfm"], result["metrics"]
    st.caption(f"Algoritma: {result['algo']} — parameter: {result['params']}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Jumlah cluster", metrics["Jumlah cluster"])
    m2.metric("Noise (%)", f"{metrics['Noise (%)']:.2f}")
    m3.metric("Silhouette", fmt(metrics["Silhouette"], ".4f"))
    m4.metric("Davies-Bouldin", fmt(metrics["Davies-Bouldin"], ".4f"))
    m5.metric("Calinski-Harabasz", fmt(metrics["Calinski-Harabasz"], ",.2f"))
    if metrics["Silhouette"] is None:
        st.warning("Metrik tidak dapat dihitung karena cluster (di luar noise) kurang dari 2. Ubah parameter.")

    prof = build_profile(seg, "Cluster")
    st.subheader("Profil cluster")
    show_df(prof)
    if (seg["Cluster"] == -1).any():
        st.caption("Cluster -1 pada DBSCAN adalah noise (bukan segmen).")

    plot_prof = prof.copy()
    plot_prof["Cluster"] = plot_prof["Cluster"].astype(str)
    g1, g2 = st.columns(2)
    with g1:
        show_chart(px.pie(plot_prof, names="Cluster", values="Customer_Count",
                          title="Distribusi pelanggan", hole=0.4))
    with g2:
        show_chart(px.bar(plot_prof, x="Cluster", y=["Recency_Mean", "Frequency_Mean", "Monetary_Mean"],
                          barmode="group", title="Rata-rata RFM per cluster",
                          labels={"value": "Nilai rata-rata", "variable": "Variabel RFM"}))

    plot_df = seg.sample(min(5000, len(seg)), random_state=42).copy()
    plot_df["Cluster"] = plot_df["Cluster"].astype(str)
    plot_df = plot_df[(plot_df["Frequency"] > 0) & (plot_df["Monetary"] > 0)]
    show_chart(px.scatter(plot_df, x="Frequency", y="Monetary", color="Cluster", log_x=True, log_y=True,
                          opacity=0.6, title="Frequency vs Monetary (skala log, maks. 5.000 titik)"))

    st.subheader("Unduh hasil")
    d1, d2 = st.columns(2)
    d1.download_button("Unduh data segmentasi (CSV)", data=seg.to_csv(index=False).encode("utf-8"),
                       file_name="rfm_segments_new.csv", mime="text/csv")
    d2.download_button("Unduh profil cluster (CSV)", data=prof.to_csv(index=False).encode("utf-8"),
                       file_name="profile_cluster_new.csv", mime="text/csv")


# ---------------------------------------------------------------------------
page = st.sidebar.radio("Navigasi", ["Overview Dashboard", "Explore Dataset"])
if page == "Overview Dashboard":
    render_overview()
else:
    render_explore_dataset()
