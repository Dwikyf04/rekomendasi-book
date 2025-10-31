import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

# ======================================================
# 1️⃣ KONFIGURASI DASAR
# ======================================================
st.set_page_config(
    page_title="Book Recommendation | Nanda",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# 2️⃣ STYLE (Seperti Web UNAIR)
# ======================================================
st.markdown("""
    <style>
        body {
            background-color: #F8FAFC;
        }
        .main {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #003366;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #00509E;
        }
        h1, h2, h3 {
            color: #003366;
        }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# 3️⃣ LOGIN SEDERHANA
# ======================================================
def login():
    st.sidebar.image("https://www.unair.ac.id/wp-content/uploads/2022/03/LOGO-UNAIR-2022.png", width=150)
    st.sidebar.title("📘 Login Pengguna")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_btn = st.sidebar.button("Masuk")

    if login_btn:
        if username == "admin" and password == "12345":
            st.session_state["login"] = True
            st.session_state["username"] = username
            st.success("✅ Login berhasil! Selamat datang, " + username)
        else:
            st.error("❌ Username atau password salah!")

if "login" not in st.session_state:
    login()

if "login" in st.session_state and st.session_state["login"] == True:
    # ======================================================
    # 4️⃣ LOAD DATA & MODEL
    # ======================================================
    @st.cache_resource
    def load_models():
        with open("model/tfidf_vectorizer.pkl", "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        with open("model/kmeans_model.pkl", "rb") as f:
            kmeans_model = pickle.load(f)
        with open("model/knn_model.pkl", "rb") as f:
            knn_model = pickle.load(f)
        with open("model/embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        return tfidf_vectorizer, kmeans_model, knn_model, embeddings

    tfidf_vectorizer, kmeans_model, knn_model, embeddings = load_models()

    @st.cache_data
    def load_books():
        return pd.read_csv("books.csv")

    df = load_books()

    # Pastikan kolom utama tersedia
    if not all(col in df.columns for col in ["title", "authors", "categories"]):
        st.warning("⚠️ Pastikan dataset memiliki kolom: title, authors, categories")
    else:
        # ======================================================
        # 5️⃣ FITUR REKOMENDASI
        # ======================================================
        st.title("📚 Sistem Rekomendasi Buku Nanda")
        st.markdown("Temukan buku menarik berdasarkan kesamaan isi dan tema!")

        judul_input = st.text_input("Masukkan Judul Buku", "Harry Potter")
        metode = st.selectbox("Pilih Metode Rekomendasi", [
            "TF-IDF + Cosine Similarity",
            "Embedding",
            "KNN + Embedding",
            "K-Means Cluster"
        ])

        if st.button("🔍 Tampilkan Rekomendasi"):
            # TF-IDF
            if metode == "TF-IDF + Cosine Similarity":
                tfidf_matrix = tfidf_vectorizer.transform(df["title"].astype(str))
                query_vec = tfidf_vectorizer.transform([judul_input])
                similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
                top_idx = similarity.argsort()[-5:][::-1]
                hasil = df.iloc[top_idx][["title", "authors", "categories"]]
                st.write("📖 Hasil Rekomendasi (TF-IDF):")
                st.dataframe(hasil)

            # Embedding
            elif metode == "Embedding":
                from sklearn.metrics.pairwise import cosine_similarity
                sim = cosine_similarity([embeddings.get(judul_input, np.zeros(384))], list(embeddings.values()))
                top_idx = np.argsort(sim[0])[-5:][::-1]
                hasil = pd.DataFrame({
                    "title": list(embeddings.keys())[i] for i in top_idx
                })
                st.write("📖 Hasil Rekomendasi (Embedding):")
                st.dataframe(hasil)

            # KNN
            elif metode == "KNN + Embedding":
                X = np.array(list(embeddings.values()))
                knn = NearestNeighbors(n_neighbors=5, metric="cosine")
                knn.fit(X)
                if judul_input in embeddings:
                    distances, indices = knn.kneighbors([embeddings[judul_input]])
                    hasil = pd.DataFrame({
                        "title": [list(embeddings.keys())[i] for i in indices[0]],
                        "distance": distances[0]
                    })
                    st.dataframe(hasil)
                else:
                    st.warning("Judul tidak ditemukan di embeddings!")

            # K-Means
            elif metode == "K-Means Cluster":
                tfidf_matrix = tfidf_vectorizer.transform(df["title"].astype(str))
                clusters = kmeans_model.predict(tfidf_matrix)
                df["cluster"] = clusters
                if judul_input in df["title"].values:
                    cluster_id = df[df["title"] == judul_input]["cluster"].values[0]
                    hasil = df[df["cluster"] == cluster_id].head(5)
                    st.write(f"📖 Buku dalam Cluster yang Sama (Cluster {cluster_id}):")
                    st.dataframe(hasil)
                else:
                    st.warning("Judul tidak ditemukan di dataset!")

        # ======================================================
        # 6️⃣ VISUALISASI CLUSTER
        # ======================================================
        with st.expander("📊 Visualisasi Cluster (K-Means)"):
            tfidf_matrix = tfidf_vectorizer.transform(df["title"].astype(str))
            df["cluster"] = kmeans_model.predict(tfidf_matrix)
            cluster_counts = df["cluster"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Jumlah Buku"]
            fig = px.bar(cluster_counts, x="Cluster", y="Jumlah Buku",
                         color="Cluster", title="Distribusi Buku per Cluster")
            st.plotly_chart(fig, use_container_width=True)

        # ======================================================
        # 7️⃣ LOGOUT
        # ======================================================
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()

