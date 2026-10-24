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

# ---------------------------------------------------------------------------
# Konfigurasi path (struktur lama dipertahankan: results/ dan profiles/)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PROFILES_DIR = BASE_DIR / "profiles"
OUTPUT_DIR = BASE_DIR / "outputs"          # opsional: summary JSON + gambar
FIGURES_DIR = OUTPUT_DIR / "figures"

st.set_page_config(
    page_title="Customer Segmentation | RFM",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helper pembaca file (aman jika file tidak ada)
# ---------------------------------------------------------------------------
@st.cache_data
def load_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None


def load_experiment_summary():
    path = OUTPUT_DIR / "experiment_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------------
# HALAMAN 1: OVERVIEW DASHBOARD (hasil eksperimen tersimpan)
# ---------------------------------------------------------------------------
def render_overview():
    st.title("📊 Customer Segmentation Dashboard")
    st.caption(
        "Perbandingan K-Means, Agglomerative Clustering, dan DBSCAN "
        "untuk segmentasi pelanggan berbasis RFM"
    )

    files = {
        "kmeans_results": RESULTS_DIR / "results_kmeans.csv",
        "agg_results": RESULTS_DIR / "results_agglomerative.csv",
        "dbscan_results": RESULTS_DIR / "results_dbscan.csv",
        "kmeans_profile": PROFILES_DIR / "profile_kmeans_2.csv",
        "kmeans4_profile": PROFILES_DIR / "profile_kmeans_4.csv",
        "agg_profile": PROFILES_DIR / "profile_agglomerative_2.csv",
        "dbscan_profile": PROFILES_DIR / "profile_dbscan.csv",
    }
    data = {name: load_csv(path) for name, path in files.items()}
    missing = [str(files[n].relative_to(BASE_DIR)) for n, d in data.items() if d is None]
    if missing:
        st.error(
            "File hasil eksperimen berikut tidak ditemukan atau tidak dapat dibaca:\n\n"
            + "\n".join(f"- `{m}`" for m in missing)
        )
        st.stop()

    kmeans_results = data["kmeans_results"]
    agg_results = data["agg_results"]
    dbscan_results = data["dbscan_results"]
    kmeans_profile = data["kmeans_profile"]
    kmeans4_profile = data["kmeans4_profile"]
    agg_profile = data["agg_profile"]
    dbscan_profile = data["dbscan_profile"]

    # Konfigurasi terbaik dari hasil eksperimen tersimpan
    best_kmeans = kmeans_results.loc[kmeans_results["silhouette"].idxmax()]
    best_agg = agg_results.loc[agg_results["silhouette"].idxmax()]
    best_dbscan = dbscan_results.loc[dbscan_results["silhouette"].idxmax()]

    comparison = pd.DataFrame([
        {
            "Model": f"K-Means (k={int(best_kmeans['k'])})",
            "Silhouette": best_kmeans["silhouette"],
            "Davies-Bouldin": best_kmeans["davies_bouldin"],
            "Calinski-Harabasz": best_kmeans["calinski_harabasz"],
            "Noise (%)": 0.0,
        },
        {
            "Model": f"Agglomerative (k={int(best_agg['k'])})",
            "Silhouette": best_agg["silhouette"],
            "Davies-Bouldin": best_agg["davies_bouldin"],
            "Calinski-Harabasz": best_agg["calinski_harabasz"],
            "Noise (%)": 0.0,
        },
        {
            "Model": (
                f"DBSCAN (eps={best_dbscan['eps']:.2f}, "
                f"min_samples={int(best_dbscan['min_samples'])})"
            ),
            "Silhouette": best_dbscan["silhouette"],
            "Davies-Bouldin": best_dbscan["davies_bouldin"],
            "Calinski-Harabasz": best_dbscan["calinski_harabasz"],
            "Noise (%)": best_dbscan["noise_percentage"],
        },
    ])

    c1, c2, c3 = st.columns(3)
    c1.metric("Pelanggan dianalisis", "5.878")
    c2.metric("Transaksi valid", "779.425")
    c3.metric("Algoritma dibandingkan", "3")

    st.divider()

    # --- 1. Perbandingan model ---------------------------------------------
    st.header("1. Perbandingan Model")
    st.markdown(
        "Silhouette dan Calinski-Harabasz yang lebih tinggi umumnya lebih baik, "
        "sedangkan Davies-Bouldin yang lebih rendah umumnya lebih baik."
    )

    st.dataframe(
        comparison.style.format({
            "Silhouette": "{:.4f}",
            "Davies-Bouldin": "{:.4f}",
            "Calinski-Harabasz": "{:,.2f}",
            "Noise (%)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    metric_choice = st.selectbox(
        "Pilih metrik untuk grafik perbandingan",
        ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"],
    )
    fig_comparison = px.bar(
        comparison,
        x="Model",
        y=metric_choice,
        color="Model",
        title=f"Perbandingan model berdasarkan {metric_choice}",
        text_auto=".4f",
    )
    fig_comparison.update_layout(showlegend=False)
    st.plotly_chart(fig_comparison, use_container_width=True)

    st.info(
        "Pada eksperimen ini, K-Means k=2 memberikan hasil terbaik pada "
        "ketiga metrik evaluasi internal dibandingkan konfigurasi terpilih "
        "dari dua algoritma lainnya."
    )

    # --- Hasil eksperimen lengkap (tabel + unduh) ---------------------------
    with st.expander("Lihat semua hasil eksperimen per algoritma"):
        for name, df in [
            ("K-Means", kmeans_results),
            ("Agglomerative", agg_results),
            ("DBSCAN", dbscan_results),
        ]:
            st.markdown(f"**{name}**")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                f"Unduh hasil {name}",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"results_{name.lower()}.csv",
                mime="text/csv",
                key=f"download_overview_{name}",
            )

    # Ringkasan JSON opsional (folder outputs/)
    summary = load_experiment_summary()
    if summary is not None:
        with st.expander("Ringkasan eksperimen (experiment_summary.json)"):
            st.json(summary, expanded=False)

    st.divider()

    # --- 2. Profil segmen ---------------------------------------------------
    st.header("2. Eksplorasi Profil Segmen Pelanggan")

    model_choice = st.selectbox(
        "Pilih hasil segmentasi",
        [
            "K-Means (k=2)",
            "K-Means (k=4)",
            "Agglomerative (k=2)",
            "DBSCAN (eps=0.45)",
        ],
    )

    profiles = {
        "K-Means (k=2)": (kmeans_profile, "Cluster_KMeans_2"),
        "K-Means (k=4)": (kmeans4_profile, "Cluster_KMeans_4"),
        "Agglomerative (k=2)": (agg_profile, "Cluster_Agglomerative_2"),
        "DBSCAN (eps=0.45)": (dbscan_profile, "Cluster_DBSCAN"),
    }
    interpretations = {
        "K-Means (k=2)": {0: "Inactive / Low-Value", 1: "Active / High-Value"},
        "Agglomerative (k=2)": {0: "Active / High-Value", 1: "Inactive / Low-Value"},
        "DBSCAN (eps=0.45)": {
            0: "Relatively Active / Higher-Value",
            1: "Inactive / Low-Value",
        },
        "K-Means (k=4)": {
            0: "Very Inactive / Low-Value",
            1: "Recently Active / Lower Frequency",
            2: "Active / High-Value",
            3: "Moderate Activity / Value",
        },
    }

    profile, cluster_col = profiles[model_choice]
    display_profile = profile.copy()
    display_profile["Interpretasi"] = display_profile[cluster_col].map(
        interpretations[model_choice]
    )

    st.dataframe(
        display_profile.style.format({
            "Recency_Mean": "{:.2f}",
            "Recency_Median": "{:.2f}",
            "Frequency_Mean": "{:.2f}",
            "Frequency_Median": "{:.2f}",
            "Monetary_Mean": "{:,.2f}",
            "Monetary_Median": "{:,.2f}",
            "Customer_Percentage": "{:.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    pie_df = profile.copy()
    pie_df[cluster_col] = pie_df[cluster_col].astype(str)
    fig_size = px.pie(
        pie_df,
        names=cluster_col,
        values="Customer_Count",
        title=f"Distribusi pelanggan — {model_choice}",
        hole=0.4,
    )
    st.plotly_chart(fig_size, use_container_width=True)

    bar_df = display_profile.copy()
    bar_df[cluster_col] = bar_df[cluster_col].astype(str)
    fig_rfm = px.bar(
        bar_df,
        x=cluster_col,
        y=["Recency_Mean", "Frequency_Mean", "Monetary_Mean"],
        barmode="group",
        title="Perbandingan rata-rata RFM per cluster",
        labels={"value": "Nilai rata-rata", "variable": "Variabel RFM"},
    )
    st.plotly_chart(fig_rfm, use_container_width=True)

    if model_choice == "DBSCAN (eps=0.45)":
        st.warning(
            "Profil DBSCAN mencakup 5.807 pelanggan dalam dua cluster. "
            "Sebanyak 71 pelanggan (sekitar 1,21%) dikategorikan sebagai noise "
            "dan tidak tercantum dalam file profil cluster."
        )

    # Gambar opsional dari outputs/figures
    if FIGURES_DIR.exists():
        images = sorted(
            p for p in FIGURES_DIR.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if images:
            with st.expander(f"Galeri gambar hasil eksperimen ({len(images)})"):
                chosen = st.multiselect(
                    "Pilih gambar",
                    [p.name for p in images],
                    default=[images[0].name],
                )
                for p in images:
                    if p.name in chosen:
                        st.image(str(p), caption=p.name, use_container_width=True)

    st.divider()

    # --- 3. Interpretasi bisnis --------------------------------------------
    st.header("3. Interpretasi Bisnis")

    st.subheader("Active / High-Value Customers")
    st.markdown(
        "- Pertahankan pelanggan melalui program loyalitas.\n"
        "- Tawarkan promosi yang relevan berdasarkan riwayat pembelian.\n"
        "- Berikan penghargaan kepada pelanggan dengan frekuensi dan nilai "
        "transaksi tinggi."
    )

    st.subheader("Inactive / Low-Value Customers")
    st.markdown(
        "- Jalankan kampanye reaktivasi pelanggan.\n"
        "- Berikan voucher atau penawaran khusus secara terukur.\n"
        "- Gunakan pengingat dan rekomendasi produk untuk mendorong pembelian ulang."
    )

    st.caption(
        "Catatan: nama segmen merupakan interpretasi berdasarkan karakteristik "
        "RFM. Nominal Monetary mengikuti satuan mata uang pada dataset sumber. "
        "Label cluster adalah label teknis dan tidak menunjukkan urutan nilai."
    )


# ---------------------------------------------------------------------------
# HALAMAN 2: EXPLORE DATASET (upload + clustering baru)
# ---------------------------------------------------------------------------
def read_uploaded(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)  # butuh openpyxl


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

    grouped = work.groupby(cust_col)
    rfm = pd.DataFrame({
        "Recency": (snapshot - grouped[date_col].max()).dt.days,
        "Frequency": (
            grouped[inv_col].nunique() if inv_col else grouped[date_col].nunique()
        ),
        "Monetary": grouped["_amount"].sum(),
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
        model = AgglomerativeClustering(
            n_clusters=params["k"], linkage=params["linkage"]
        )
    else:
        model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])

    labels = model.fit_predict(X)

    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    metrics = {
        "Jumlah cluster": n_clusters,
        "Noise (%)": float((~mask).mean() * 100),
        "Silhouette": None,
        "Davies-Bouldin": None,
        "Calinski-Harabasz": None,
    }
    if n_clusters >= 2 and mask.sum() > n_clusters:
        Xm, lm = X[mask], labels[mask]
        sample = min(10000, len(Xm)) if len(Xm) > 10000 else None
        metrics["Silhouette"] = float(
            silhouette_score(Xm, lm, sample_size=sample, random_state=42)
        )
        metrics["Davies-Bouldin"] = float(davies_bouldin_score(Xm, lm))
        metrics["Calinski-Harabasz"] = float(calinski_harabasz_score(Xm, lm))
    return labels, metrics


def render_explore_dataset():
    st.title("🔎 Explore Dataset")
    st.caption(
        "Unggah dataset Anda sendiri, pilih algoritma clustering, jalankan "
        "eksperimen, lalu unduh hasilnya."
    )

    # --- 1. Upload ----------------------------------------------------------
    st.header("1. Unggah Data")
    uploaded = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])
    if uploaded is None:
        st.info("Unggah dataset untuk memulai.")
        return

    try:
        raw = read_uploaded(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"File tidak dapat dibaca: {exc}")
        return

    st.write(f"{len(raw):,} baris, {raw.shape[1]} kolom")
    st.dataframe(raw.head(20), use_container_width=True)

    cols = raw.columns.tolist()
    mode = st.radio(
        "Jenis data",
        ["Data RFM siap pakai", "Data transaksi mentah"],
        horizontal=True,
    )

    # --- 2. Mapping kolom ---------------------------------------------------
    st.header("2. Pilih Kolom")
    rfm = None

    if mode == "Data RFM siap pakai":
        c1, c2, c3, c4 = st.columns(4)
        id_col = c1.selectbox("Kolom ID pelanggan (opsional)", ["(tidak ada)"] + cols)
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
        cust_col = c1.selectbox("Kolom ID pelanggan", cols)
        date_col = c2.selectbox("Kolom tanggal transaksi", cols)
        inv_col = c3.selectbox("Kolom nomor invoice (opsional)", ["(tidak ada)"] + cols)
        c4, c5 = st.columns(2)
        qty_col = c4.selectbox("Kolom jumlah (Quantity)", cols)
        price_col = c5.selectbox("Kolom harga satuan (Price)", cols)

        if len({cust_col, date_col, qty_col, price_col}) < 4:
            st.warning("Pilih kolom yang berbeda untuk setiap peran.")
            return

        rfm = build_rfm_from_transactions(
            raw, cust_col, date_col,
            None if inv_col == "(tidak ada)" else inv_col,
            qty_col, price_col,
        )

    if rfm is None or rfm.empty:
        st.error("Tidak ada data valid setelah pembersihan. Periksa pemilihan kolom.")
        return

    st.success(f"Data RFM siap: {len(rfm):,} pelanggan/baris.")
    with st.expander("Lihat tabel RFM"):
        st.dataframe(rfm.head(100), use_container_width=True, hide_index=True)

    # --- 3. Pengaturan algoritma -------------------------------------------
    st.header("3. Pengaturan Clustering")
    algo = st.selectbox("Algoritma", ["K-Means", "Agglomerative", "DBSCAN"])
    use_log = st.checkbox(
        "Transformasi log1p sebelum standardisasi (disarankan untuk data RFM yang miring)",
        value=True,
    )

    params = {}
    if algo in ("K-Means", "Agglomerative"):
        params["k"] = st.slider("Jumlah cluster (k)", 2, 10, 2)
    if algo == "Agglomerative":
        params["linkage"] = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
        if len(rfm) > 15000:
            st.warning(
                "Agglomerative pada data besar membutuhkan banyak memori dan waktu. "
                "Pertimbangkan K-Means atau sampling data."
            )
    if algo == "DBSCAN":
        c1, c2 = st.columns(2)
        params["eps"] = c1.slider("eps", 0.05, 3.0, 0.45, 0.05)
        params["min_samples"] = c2.slider("min_samples", 2, 50, 5)

    if st.button("Jalankan clustering", type="primary"):
        with st.spinner("Menjalankan clustering..."):
            try:
                labels, metrics = run_clustering(rfm, algo, params, use_log)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Clustering gagal: {exc}")
                return
        st.session_state["explore_result"] = {
            "rfm": rfm.assign(Cluster=labels),
            "metrics": metrics,
            "algo": algo,
            "params": params,
        }

    # --- 4. Hasil -----------------------------------------------------------
    result = st.session_state.get("explore_result")
    if result is None:
        return

    st.header("4. Hasil")
    seg = result["rfm"]
    metrics = result["metrics"]
    st.caption(f"Algoritma: {result['algo']} — parameter: {result['params']}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Jumlah cluster", metrics["Jumlah cluster"])
    m2.metric("Noise (%)", f"{metrics['Noise (%)']:.2f}")
    m3.metric("Silhouette", "—" if metrics["Silhouette"] is None else f"{metrics['Silhouette']:.4f}")
    m4.metric("Davies-Bouldin", "—" if metrics["Davies-Bouldin"] is None else f"{metrics['Davies-Bouldin']:.4f}")
    m5.metric("Calinski-Harabasz", "—" if metrics["Calinski-Harabasz"] is None else f"{metrics['Calinski-Harabasz']:,.2f}")

    if metrics["Silhouette"] is None:
        st.warning(
            "Metrik evaluasi tidak dapat dihitung karena hasil memiliki kurang "
            "dari 2 cluster (di luar noise). Coba ubah parameter."
        )

    prof = build_profile(seg, "Cluster")
    st.subheader("Profil cluster")
    st.dataframe(
        prof.style.format({
            "Recency_Mean": "{:.2f}",
            "Recency_Median": "{:.2f}",
            "Frequency_Mean": "{:.2f}",
            "Frequency_Median": "{:.2f}",
            "Monetary_Mean": "{:,.2f}",
            "Monetary_Median": "{:,.2f}",
            "Customer_Percentage": "{:.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )
    if (seg["Cluster"] == -1).any():
        st.caption("Cluster -1 pada DBSCAN adalah noise (bukan segmen).")

    plot_prof = prof.copy()
    plot_prof["Cluster"] = plot_prof["Cluster"].astype(str)
    g1, g2 = st.columns(2)
    g1.plotly_chart(
        px.pie(plot_prof, names="Cluster", values="Customer_Count",
               title="Distribusi pelanggan", hole=0.4),
        use_container_width=True,
    )
    g2.plotly_chart(
        px.bar(plot_prof, x="Cluster",
               y=["Recency_Mean", "Frequency_Mean", "Monetary_Mean"],
               barmode="group", title="Rata-rata RFM per cluster",
               labels={"value": "Nilai rata-rata", "variable": "Variabel RFM"}),
        use_container_width=True,
    )

    plot_df = seg.sample(min(5000, len(seg)), random_state=42).copy()
    plot_df["Cluster"] = plot_df["Cluster"].astype(str)
    st.plotly_chart(
        px.scatter(plot_df, x="Frequency", y="Monetary", color="Cluster",
                   log_x=True, log_y=True, opacity=0.6,
                   title="Sebaran Frequency vs Monetary (skala log, maks. 5.000 titik)"),
        use_container_width=True,
    )

    st.subheader("Unduh hasil")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Unduh data segmentasi (CSV)",
        data=seg.to_csv(index=False).encode("utf-8"),
        file_name="rfm_segments.csv",
        mime="text/csv",
    )
    d2.download_button(
        "Unduh profil cluster (CSV)",
        data=prof.to_csv(index=False).encode("utf-8"),
        file_name="profile_cluster.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Navigasi
# ---------------------------------------------------------------------------
page = st.sidebar.radio("Navigasi", ["Overview Dashboard", "Explore Dataset"])

if page == "Overview Dashboard":
    render_overview()
else:
    render_explore_dataset()
