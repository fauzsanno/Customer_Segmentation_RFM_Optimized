import io
import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import RobustScaler

try:
    from sklearn.cluster import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Customer Segmentation | RFM Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Segmentation Dashboard")
st.markdown(
    """
    Interactive customer segmentation using **Recency, Frequency,
    and Monetary (RFM)** analysis with multiple clustering algorithms.
    """
)

st.caption(
    "Upload transaction data or an existing RFM table to run your own experiment."
)


# =========================================================
# DATA LOADING AND COLUMN NORMALIZATION
# =========================================================
def normalize_column_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"[_\-]+", " ", name)
    return re.sub(r"\s+", " ", name)


def normalize_columns(df):
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]

    aliases = {
        "invoice": ["invoice", "invoiceno", "invoice no", "invoice number"],
        "customer id": [
            "customer id", "customerid", "customer", "customer number"
        ],
        "invoice date": ["invoice date", "invoicedate", "date", "transaction date"],
        "quantity": ["quantity", "qty"],
        "price": ["price", "unit price", "unitprice", "unit cost"],
        "recency": ["recency", "r"],
        "frequency": ["frequency", "freq", "f"],
        "monetary": ["monetary", "monetary value", "spend", "total spend", "m"],
    }

    rename = {}
    for standard_name, possible_names in aliases.items():
        for name in possible_names:
            if name in df.columns:
                rename[name] = standard_name
                break

    return df.rename(columns=rename)


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes, filename):
    buffer = io.BytesIO(file_bytes)

    if filename.lower().endswith(".csv"):
        try:
            return pd.read_csv(buffer, low_memory=False)
        except UnicodeDecodeError:
            buffer.seek(0)
            return pd.read_csv(buffer, encoding="latin-1", low_memory=False)

    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)

    raise ValueError("Unsupported file format.")


def build_rfm(df):
    """Accept raw transactions or a precomputed RFM table."""
    df = normalize_columns(df)

    required_rfm = {"recency", "frequency", "monetary"}

    # Case 1: the uploaded file already contains RFM metrics.
    if required_rfm.issubset(df.columns):
        rfm = df.copy()

        if "customer id" not in rfm.columns:
            rfm.insert(0, "customer id", np.arange(1, len(rfm) + 1))

        rfm = rfm[["customer id", "recency", "frequency", "monetary"]]

        for col in ["recency", "frequency", "monetary"]:
            rfm[col] = pd.to_numeric(rfm[col], errors="coerce")

        rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["recency", "frequency", "monetary"]
        )

        rfm = rfm[
            (rfm["recency"] >= 0)
            & (rfm["frequency"] > 0)
            & (rfm["monetary"] >= 0)
        ]

        rfm = rfm.drop_duplicates(subset=["customer id"], keep="first")
        return rfm.reset_index(drop=True), "RFM table"

    # Case 2: raw transaction data.
    required_transactions = {
        "invoice", "customer id", "invoice date", "quantity", "price"
    }

    missing = required_transactions - set(df.columns)
    if missing:
        raise ValueError(
            "Kolom tidak dikenali. Upload tabel RFM dengan kolom "
            "Recency, Frequency, Monetary, atau data transaksi yang memiliki "
            "Invoice, Customer ID, InvoiceDate, Quantity, dan Price. "
            f"Kolom yang belum ditemukan: {', '.join(sorted(missing))}"
        )

    df["customer id"] = df["customer id"].astype("string").str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["invoice date"] = pd.to_datetime(
        df["invoice date"], errors="coerce", dayfirst=True
    )

    df = df.dropna(
        subset=["customer id", "invoice", "invoice date", "quantity", "price"]
    )

    df = df[
        (df["customer id"] != "")
        & (df["quantity"] > 0)
        & (df["price"] > 0)
    ].copy()

    # Exclude cancelled invoices, such as invoices beginning with C.
    df = df[
        ~df["invoice"].astype(str).str.strip().str.upper().str.startswith("C")
    ].copy()

    if df.empty:
        raise ValueError(
            "Tidak ada transaksi valid setelah proses pembersihan data."
        )

    df["sales"] = df["quantity"] * df["price"]
    reference_date = df["invoice date"].max().normalize() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("customer id", as_index=False)
        .agg(
            last_purchase=("invoice date", "max"),
            frequency=("invoice", "nunique"),
            monetary=("sales", "sum"),
        )
    )

    rfm["recency"] = (
        reference_date - rfm["last_purchase"].dt.normalize()
    ).dt.days

    rfm = rfm[
        ["customer id", "recency", "frequency", "monetary"]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    rfm = rfm[
        (rfm["recency"] >= 0)
        & (rfm["frequency"] > 0)
        & (rfm["monetary"] >= 0)
    ]

    return rfm.reset_index(drop=True), "Raw transactions"


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Experiment Settings")

    algorithm = st.selectbox(
        "Clustering algorithm",
        ["K-Means", "Agglomerative", "DBSCAN"] +
        (["HDBSCAN"] if HDBSCAN_AVAILABLE else []),
    )

    if algorithm in ["K-Means", "Agglomerative"]:
        n_clusters = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=10,
            value=4,
        )
    elif algorithm == "DBSCAN":
        eps = st.slider(
            "DBSCAN eps",
            min_value=0.1,
            max_value=3.0,
            value=0.8,
            step=0.1,
        )
        min_samples = st.slider(
            "DBSCAN min_samples",
            min_value=2,
            max_value=20,
            value=5,
        )
    else:
        min_cluster_size = st.slider(
            "HDBSCAN min_cluster_size",
            min_value=5,
            max_value=50,
            value=10,
        )
        min_samples_hdb = st.slider(
            "HDBSCAN min_samples",
            min_value=2,
            max_value=20,
            value=5,
        )

    st.info(
        "RFM features are log-transformed and scaled before clustering. "
        "The displayed cluster labels are algorithm-generated labels."
    )


# =========================================================
# UPLOAD DATA
# =========================================================
uploaded_file = st.file_uploader(
    "Upload a transaction dataset or RFM CSV/Excel file",
    type=["csv", "xlsx", "xls"],
    help="For raw transactions, use Online Retail II-style columns. "
         "For RFM data, include Recency, Frequency, and Monetary.",
    max_upload_size=200,
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to start the analysis.")

    st.markdown("### Expected input formats")

    with st.expander("Raw transaction data"):
        st.code(
            "Invoice | Customer ID | InvoiceDate | Quantity | Price\n"
            "536365  | 17850       | 2010-12-01 | 6        | 2.55",
            language="text",
        )

    with st.expander("Precomputed RFM data"):
        st.code(
            "CustomerID | Recency | Frequency | Monetary\n"
            "17850      | 12      | 8         | 1500.50",
            language="text",
        )

    st.stop()


# =========================================================
# PREPARE RFM
# =========================================================
try:
    raw_df = read_uploaded_file(
        uploaded_file.getvalue(),
        uploaded_file.name,
    )
    rfm, input_type = build_rfm(raw_df)
except Exception as exc:
    st.error(f"Gagal membaca atau memproses dataset: {exc}")
    st.stop()

if len(rfm) < 3:
    st.error("Diperlukan minimal 3 pelanggan valid untuk clustering.")
    st.stop()

st.success(
    f"Data berhasil diproses sebagai {input_type}. "
    f"Jumlah pelanggan valid: {len(rfm):,}."
)

with st.expander("Preview RFM data"):
    st.dataframe(rfm.head(20), use_container_width=True)


# =========================================================
# RFM SUMMARY
# =========================================================
st.subheader("📌 RFM Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(rfm):,}")
c2.metric("Average Recency", f"{rfm['recency'].mean():,.1f} days")
c3.metric("Average Frequency", f"{rfm['frequency'].mean():,.1f}")
c4.metric("Total Monetary", f"{rfm['monetary'].sum():,.2f}")

with st.expander("RFM descriptive statistics"):
    st.dataframe(
        rfm[["recency", "frequency", "monetary"]].describe().T,
        use_container_width=True,
    )


# =========================================================
# CLUSTERING
# =========================================================
st.subheader(f"🧪 Run {algorithm} clustering")

if st.button("Run Clustering", type="primary", use_container_width=True):
    with st.spinner("Preparing features and running clustering..."):
        feature_cols = ["recency", "frequency", "monetary"]
        X = rfm[feature_cols].to_numpy(dtype=float)

        # RFM values are usually right-skewed.
        X_log = np.log1p(np.clip(X, a_min=0, a_max=None))
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_log)

        try:
            if algorithm == "K-Means":
                if len(rfm) <= n_clusters:
                    st.error("Jumlah pelanggan harus lebih besar dari jumlah cluster.")
                    st.stop()

                model = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10,
                )
                labels = model.fit_predict(X_scaled)

            elif algorithm == "Agglomerative":
                if len(rfm) <= n_clusters:
                    st.error("Jumlah pelanggan harus lebih besar dari jumlah cluster.")
                    st.stop()

                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X_scaled)

            elif algorithm == "DBSCAN":
                model = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                )
                labels = model.fit_predict(X_scaled)

            else:
                model = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples_hdb,
                )
                labels = model.fit_predict(X_scaled)

        except Exception as exc:
            st.error(f"Clustering gagal: {exc}")
            st.stop()

        result = rfm.copy()
        result["cluster"] = labels

        # Evaluation excludes DBSCAN/HDBSCAN noise (-1).
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]
        valid_X = X_scaled[valid_mask]
        unique_valid = np.unique(valid_labels)

        silhouette = np.nan
        davies_bouldin = np.nan
        calinski_harabasz = np.nan

        if 2 <= len(unique_valid) < len(valid_labels):
            try:
                silhouette = silhouette_score(valid_X, valid_labels)
                davies_bouldin = davies_bouldin_score(valid_X, valid_labels)
                calinski_harabasz = calinski_harabasz_score(
                    valid_X, valid_labels
                )
            except Exception:
                pass

        # PCA projection for visualization.
        pca = PCA(n_components=2, random_state=42)
        coordinates = pca.fit_transform(X_scaled)
        result["PCA 1"] = coordinates[:, 0]
        result["PCA 2"] = coordinates[:, 1]
        result["cluster_name"] = result["cluster"].map(
            lambda x: "Noise / Outlier" if x == -1 else f"Cluster {x}"
        )

        st.session_state["rfm_cluster_result"] = result
        st.session_state["rfm_cluster_algorithm"] = algorithm
        st.session_state["rfm_cluster_metrics"] = {
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "calinski_harabasz": calinski_harabasz,
            "noise_count": int(np.sum(labels == -1)),
            "cluster_count": int(len(np.unique(labels[labels != -1]))),
            "pca_variance": float(pca.explained_variance_ratio_.sum()),
        }


# =========================================================
# RESULTS
# =========================================================
if "rfm_cluster_result" in st.session_state:
    result = st.session_state["rfm_cluster_result"]
    metrics = st.session_state["rfm_cluster_metrics"]
    result_algorithm = st.session_state["rfm_cluster_algorithm"]

    st.divider()
    st.header(f"📈 Results — {result_algorithm}")

    m1, m2, m3, m4 = st.columns(4)

    silhouette_value = metrics["silhouette"]
    db_value = metrics["davies_bouldin"]
    ch_value = metrics["calinski_harabasz"]

    m1.metric(
        "Silhouette Score",
        f"{silhouette_value:.4f}" if np.isfinite(silhouette_value) else "N/A",
    )
    m2.metric(
        "Davies–Bouldin Index",
        f"{db_value:.4f}" if np.isfinite(db_value) else "N/A",
    )
    m3.metric(
        "Calinski–Harabasz",
        f"{ch_value:,.2f}" if np.isfinite(ch_value) else "N/A",
    )
    m4.metric("Detected clusters", metrics["cluster_count"])

    if result_algorithm in ["DBSCAN", "HDBSCAN"]:
        st.caption(
            f"Noise/outlier points: {metrics['noise_count']:,}. "
            "Noise is excluded from the clustering quality metrics."
        )

    left, right = st.columns(2)

    with left:
        cluster_counts = (
            result["cluster_name"]
            .value_counts()
            .rename_axis("Cluster")
            .reset_index(name="Customers")
        )
        fig_counts = px.bar(
            cluster_counts,
            x="Cluster",
            y="Customers",
            title="Customer Distribution by Cluster",
            text="Customers",
        )
        st.plotly_chart(fig_counts, use_container_width=True)

    with right:
        fig_pca = px.scatter(
            result,
            x="PCA 1",
            y="PCA 2",
            color="cluster_name",
            hover_data=["customer id", "recency", "frequency", "monetary"],
            title="Cluster Visualization — PCA",
            opacity=0.75,
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    st.subheader("Customer Segment Profiles")

    profile = (
        result.groupby("cluster_name", as_index=False)
        .agg(
            customers=("customer id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_monetary=("monetary", "sum"),
        )
        .sort_values("customers", ascending=False)
    )

    profile["customer_percentage"] = (
        profile["customers"] / len(result) * 100
    ).round(2)

    for col in ["avg_recency", "avg_frequency", "avg_monetary", "total_monetary"]:
        profile[col] = profile[col].round(2)

    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.subheader("Segment Interpretation")
    st.caption(
        "Interpretation is based on observed RFM averages. Cluster numbers "
        "are arbitrary labels, not an inherent ranking of customer value."
    )

    for _, row in profile.iterrows():
        with st.expander(f"{row['cluster_name']} — {row['customers']:,} customers"):
            st.write(
                f"- Customer share: {row['customer_percentage']:.2f}%\n"
                f"- Average recency: {row['avg_recency']:.2f} days\n"
                f"- Average frequency: {row['avg_frequency']:.2f}\n"
                f"- Average monetary value: {row['avg_monetary']:.2f}\n"
                f"- Total monetary value: {row['total_monetary']:.2f}"
            )

    st.subheader("Download Results")

    download_left, download_right = st.columns(2)

    with download_left:
        csv_data = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Customer Segments (CSV)",
            data=csv_data,
            file_name="rfm_customer_segments.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with download_right:
        profile_csv = profile.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Segment Profiles (CSV)",
            data=profile_csv,
            file_name="rfm_segment_profiles.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(
        f"PCA variance explained by two components: "
        f"{metrics['pca_variance']:.2%}."
    )
