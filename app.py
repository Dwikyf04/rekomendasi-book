import streamlit as st
import pandas as pd
# import pickle  <- HAPUS
import joblib  # <-- FIX 2: Gunakan joblib
import numpy as np
import sqlite3
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

# ======================================================
# 1️⃣ CONFIG STREAMLIT
# ======================================================
st.set_page_config(
    page_title="Book Recommendation | Nanda",
    page_icon="📚",
    layout="wide"
)

# ======================================================
# 2️⃣ STYLE
# ======================================================
st.markdown("""
    <style>
        body { background-color: #F8FAFC; }
        .main {
            background-color: white; padding: 20px;
            border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #003366; color: white;
            border-radius: 8px; border: none; padding: 10px 20px;
        }
        .stButton>button:hover { background-color: #00509E; }
        h1, h2, h3 { color: #003366; }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# 3️⃣ DATABASE USER (SQLite)
# ======================================================
# FIX 4: Buat koneksi global
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS users(username TEXT, password TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS history(username TEXT, query TEXT, method TEXT)') # Pindahkan ini ke sini
    conn.commit()

def add_userdata(username, password):
    c.execute('INSERT INTO users(username, password) VALUES (?, ?)', (username, password))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    return c.fetchall()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

create_usertable() # Jalankan sekali saat start

# ======================================================
# 4️⃣ LOGIN & REGISTER PAGE
# ======================================================
menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register":
    st.sidebar.subheader("🔐 Buat Akun Baru")
    new_user = st.sidebar.text_input("Username")
    new_password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Daftar"):
        add_userdata(new_user, make_hashes(new_password))
        st.success("✅ Akun berhasil dibuat!")
        st.info("Silakan login menggunakan akun yang baru dibuat.")

elif choice == "Login":
    st.sidebar.subheader("👤 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Masuk"):
        hashed_pswd = make_hashes(password)
        result = login_user(username, hashed_pswd)
        if result:
            st.session_state["login"] = True
            st.session_state["username"] = username
            st.success(f"Selamat datang, {username}! 🎉")
        else:
            st.error("Username atau password salah!")

# ======================================================
# 5️⃣ LOAD MODEL & DATA (HANYA SETELAH LOGIN)
# ======================================================
if "login" in st.session_state and st.session_state["login"]:
    
    @st.cache_resource
    def load_models():
        # FIX 1: Pastikan path "models/" (dengan 's') sudah benar
        # FIX 2: Gunakan joblib.load()
        try:
            tfidf_vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
            kmeans_model = joblib.load("model/kmeans_model.pkl")
            knn_model = joblib.load("model/knn_model.pkl")
            embeddings = joblib.load("model/embeddings.pkl")
            return tfidf_vectorizer, kmeans_model, knn_model, embeddings
        except FileNotFoundError:
            st.error("ERROR: File model tidak ditemukan. Pastikan folder 'models' ada di repositori Anda.")
            return None, None, None, None
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None, None, None, None

    tfidf_vectorizer, kmeans_model, knn_model, embeddings = load_models()

    @st.cache_data
    def load_books():
        try:
            return pd.read_csv("data - books.csv")
        except FileNotFoundError:
            st.error("File 'books.csv' tidak ditemukan.")
            return pd.DataFrame()

    df = load_books()

    # FIX 3: Hitung TF-IDF Matrix SEKALI SAJA di sini
    if not df.empty and tfidf_vectorizer:
        try:
            tfidf_matrix = tfidf_vectorizer.transform(df["title"].astype(str))
        except Exception as e:
            st.error(f"Gagal membuat TF-IDF Matrix: {e}")
            tfidf_matrix = None
    else:
        tfidf_matrix = None

    # Hanya jalankan sisa aplikasi jika semua berhasil dimuat
    # KODE PERBAIKAN
    if not df.empty and tfidf_vectorizer and kmeans_model and knn_model and embeddings:
    
        # ======================================================
        # 6️⃣ FITUR REKOMENDASI
        # ======================================================
        st.title("📚 Sistem Rekomendasi Buku Nanda")
        st.markdown("Temukan buku terbaik yang cocok dengan selera Anda!")

        judul_input = st.text_input("Masukkan Judul Buku", "Harry Potter")
        metode = st.selectbox("Pilih Metode Rekomendasi", [
            "TF-IDF + Cosine Similarity",
            "Embedding",
            "KNN + Embedding",
            "K-Means Cluster"
        ])

        if st.button("🔍 Tampilkan Rekomendasi"):
            if metode == "TF-IDF + Cosine Similarity":
                # FIX 3: Gunakan tfidf_matrix yang sudah di-load
                query_vec = tfidf_vectorizer.transform([judul_input])
                similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
                top_idx = similarity.argsort()[-5:][::-1]
                hasil = df.iloc[top_idx][["title", "authors", "categories"]]
                st.dataframe(hasil)

            elif metode == "Embedding":
                if judul_input in embeddings:
                    from sklearn.metrics.pairwise import cosine_similarity
                    # Asumsi 'embeddings' adalah DICTIONARY {judul: vektor}
                    sim = cosine_similarity([embeddings[judul_input]], list(embeddings.values()))
                    top_idx = np.argsort(sim[0])[-5:][::-1]
                    hasil = pd.DataFrame({
                        "title": [list(embeddings.keys())[i] for i in top_idx]
                    })
                    st.dataframe(hasil)
                else:
                    st.warning("Judul tidak ditemukan di embeddings!")

            elif metode == "KNN + Embedding":
                # FIX 3: Gunakan knn_model yang sudah dilatih (bukan dilatih ulang)
                if judul_input in embeddings:
                    distances, indices = knn_model.kneighbors([embeddings[judul_input]])
                    hasil = pd.DataFrame({
                        "title": [list(embeddings.keys())[i] for i in indices[0]],
                        "distance": distances[0]
                    })
                    st.dataframe(hasil)
                else:
                    st.warning("Judul tidak ditemukan di embeddings!")

            elif metode == "K-Means Cluster":
                # FIX 3: Gunakan tfidf_matrix yang sudah di-load
                clusters = kmeans_model.predict(tfidf_matrix)
                df["cluster"] = clusters
                if judul_input in df["title"].values:
                    cluster_id = df[df["title"] == judul_input]["cluster"].values[0]
                    hasil = df[df["cluster"] == cluster_id].head(5)
                    st.dataframe(hasil)
                else:
                    st.warning("Judul tidak ditemukan di dataset!")

            # FIX 4: Simpan ke riwayat user menggunakan koneksi global
            try:
                c.execute('INSERT INTO history(username, query, method) VALUES (?, ?, ?)',
                          (st.session_state["username"], judul_input, metode))
                conn.commit()
            except Exception as e:
                st.warning(f"Gagal menyimpan riwayat: {e}")

        # ======================================================
        # 7️⃣ HISTORI PENCARIAN
        # ======================================================
        with st.expander("🕓 Lihat Riwayat Pencarian"):
            # FIX 4: Gunakan koneksi global
            c.execute("SELECT query, method FROM history WHERE username=?", (st.session_state["username"],))
            data = c.fetchall()
            if data:
                st.table(pd.DataFrame(data, columns=["Judul", "Metode"]))
            else:
                st.info("Belum ada riwayat pencarian.")

        # ======================================================
        # 8️⃣ VISUALISASI CLUSTER
        # ======================================================
        with st.expander("📊 Visualisasi Cluster (K-Means)"):
            # FIX 3: Gunakan tfidf_matrix yang sudah di-load
            df["cluster"] = kmeans_model.predict(tfidf_matrix)
            cluster_counts = df["cluster"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Jumlah Buku"]
            fig = px.bar(cluster_counts, x="Cluster", y="Jumlah Buku",
                         color="Cluster", title="Distribusi Buku per Cluster")
            st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # 9️⃣ LOGOUT
    # ======================================================
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun() # FIX 5: Ganti ke st.rerun()


