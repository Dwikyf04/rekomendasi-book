import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

# =====================================
# 🔹 SETUP PAGE
# =====================================
st.set_page_config(
    page_title="Book Recommender Dashboard",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Book Recommendation Dashboard")
st.write("Sistem rekomendasi buku berbasis **TF-IDF**, **Cosine Similarity**, dan **K-Means Clustering**")

@st.cache_resource
def load_model():
    """
    Memuat model TF-IDF, SVM, dan K-Means dari file .pkl.
    """
    try:
        tfidf = joblib.load("Model/tfidf_vectorizer.pkl")
        kmeans_model = joblib.load("Model/kmeans_model.pkl")
        knn_model = joblib.load("Model/knn_model.pkl")
        return tfidf, kmeans_model, knn_model
    except FileNotFoundError:
        st.error("File model (.pkl) tidak ditemukan di folder /Models.")
        return None, None, None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

# =====================================
# 🔹 LOAD DATA
# =====================================
@st.cache_data
def load_library_data(file_path="books.csv"):
    """
    Memuat data perpustakaan yang SUDAH DIOLAH (RINGKASAN) dari file CSV.
    File ini digunakan untuk Tab Rekomendasi.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File data '{file_path}' (RINGKASAN) tidak ditemukan.")
        return pd.DataFrame()
    except KeyError as e:
        st.error(f"Kolom {e} tidak ditemukan di 'data_perpustakaan.csv'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data CSV (Ringkasan): {e}")
        return pd.DataFrame()


# =====================================
# 🔹 TF-IDF + KMEANS
# =====================================
tfidf, svm_model, kmeans_model, knn_model = load_model()
library_data = load_library_data()


# =====================================
# 🔹 FUNGSI REKOMENDASI
# =====================================
if selected_page == "Beranda":
    st.markdown("""
        <div style='text-align:center; padding: 20px;'>
            <h1>Sistem Rekomendasi Perpustakaan Indonesia</h1>
            <p style='font-size:18px;'>Cari perpustakaan terbaik berbasis analisis ribuan ulasan Google Maps dengan NLP & Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()
def get_tfidf_recommendations(title_input, df, tfidf_matrix, top_n=5):
    try:
        idx = df[df["title"].str.lower().str.contains(title_input.lower(), na=False)].index[0]
    except IndexError:
        return None
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = cosine_sim.argsort()[::-1][1:top_n + 1]
    return df.iloc[similar_idx][["title", "authors", "categories"]].assign(similarity=cosine_sim[similar_idx])

def get_knn_recommendations(title_input, df, tfidf_matrix, top_n=5):
    model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
    model_knn.fit(tfidf_matrix)
    try:
        idx = df[df["title"].str.lower().str.contains(title_input.lower(), na=False)].index[0]
    except IndexError:
        return None
    distances, indices = model_knn.kneighbors(tfidf_matrix[idx], n_neighbors=top_n + 1)
    hasil = df.iloc[indices.flatten()[1:]][["title", "authors", "categories"]]
    hasil["distance"] = distances.flatten()[1:]
    return hasil

# =====================================
# 🔹 SIDEBAR MENU
# =====================================
menu = st.sidebar.radio("🔍 Pilih Analisis:", ["TF-IDF + Cosine Similarity", "KNN + TF-IDF", "K-Means Clustering"])

judul_input = st.sidebar.text_input("Masukkan judul buku:", "Harry Potter")

# =====================================
# 🔹 TAB 1: TF-IDF
# =====================================
if menu == "TF-IDF + Cosine Similarity":
    st.subheader("📘 Rekomendasi Berdasarkan TF-IDF + Cosine Similarity")
    if st.sidebar.button("Cari Rekomendasi"):
        hasil = get_tfidf_recommendations(judul_input, df, tfidf_matrix)
        if hasil is not None:
            st.dataframe(hasil)
        else:
            st.warning("Judul tidak ditemukan!")

# =====================================
# 🔹 TAB 2: KNN
# =====================================
elif menu == "KNN + TF-IDF":
    st.subheader("🤖 Rekomendasi Berdasarkan KNN + TF-IDF")
    if st.sidebar.button("Cari Rekomendasi (KNN)"):
        hasil = get_knn_recommendations(judul_input, df, tfidf_matrix)
        if hasil is not None:
            st.dataframe(hasil)
        else:
            st.warning("Judul tidak ditemukan!")

# =====================================
# 🔹 TAB 3: K-MEANS
# =====================================
elif menu == "K-Means Clustering":
    st.subheader("🎯 Visualisasi Klaster Buku (K-Means)")

    # Hitung jumlah buku per cluster
    cluster_counts = df["cluster"].value_counts().reset_index()
    cluster_counts.columns = ["cluster", "jumlah_buku"]

    fig = px.bar(cluster_counts, x="cluster", y="jumlah_buku", color="cluster",
                 title="Distribusi Buku Berdasarkan Cluster")
    st.plotly_chart(fig, use_container_width=True)

    # Menampilkan contoh buku dalam cluster tertentu
    selected_cluster = st.selectbox("Pilih cluster:", sorted(df["cluster"].unique()))
    st.dataframe(df[df["cluster"] == selected_cluster][["title", "authors", "categories"]].head(10))

    st.success("Total cluster: {}".format(k))









