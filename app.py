import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
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

# =====================================
# 🔹 LOAD DATA
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv("books.csv")
    df = df.dropna(subset=["title"])
    df = df.reset_index(drop=True)
    return df

df = load_data()

# =====================================
# 🔹 TF-IDF + KMEANS
# =====================================
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["title"])

k = 10
kmeans = KMeans(n_clusters=k, random_state=42)
df["cluster"] = kmeans.fit_predict(tfidf_matrix)

# =====================================
# 🔹 FUNGSI REKOMENDASI
# =====================================
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



