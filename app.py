# app.py
# Book Recommender Portfolio — Nanda
# GABUNGAN FINAL: UI (UNAIR) + DB (SQLite) + Model (Joblib)
# BARU: Model dan Data hanya dimuat SETELAH login berhasil.

import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Menggunakan joblib, bukan pickle
import os
import hashlib
import sqlite3  # Menggunakan SQLite, bukan JSON
from textwrap import shorten
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from streamlit_option_menu import option_menu
from sklearn.cluster import KMeans

# Optional semantic model
HAS_SBERT = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except Exception:
    HAS_SBERT = False

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")
ROOT = os.getcwd()
MODELS_DIR = os.path.join(ROOT, "model")
DB_FILE = os.path.join(ROOT, "users.db")  # Menggunakan file .db
BOOKS_CSV = os.path.join(ROOT, "data - books.csv")

# ---------------------------
# Style (dari File 1)
# ---------------------------
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

# ---------------------------
# Utilities: users.db (dari File 2, diadaptasi)
# ---------------------------
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS history(username TEXT, timestamp INTEGER, query TEXT, method TEXT)')
    conn.commit()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def register_user(username, password):
    try:
        c.execute('INSERT INTO users(username, password) VALUES (?, ?)', (username, make_hashes(password)))
        conn.commit()
        return True, "Registrasi berhasil."
    except sqlite3.IntegrityError:
        return False, "Username sudah terdaftar."
    except Exception as e:
        return False, f"Error: {e}"

def authenticate_user(username, password):
    hashed_pswd = make_hashes(password)
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_pswd))
    return c.fetchall()

def add_history(username, query, method):
    try:
        timestamp = int(pd.Timestamp.now().timestamp())
        c.execute('INSERT INTO history(username, timestamp, query, method) VALUES (?, ?, ?, ?)',
                  (username, timestamp, query, method))
        conn.commit()
    except Exception as e:
        st.error(f"Gagal menyimpan riwayat: {e}")

def get_history(username):
    c.execute("SELECT timestamp, query, method FROM history WHERE username=? ORDER BY timestamp DESC LIMIT 20", (username,))
    return c.fetchall()

# Inisialisasi tabel saat startup
create_usertable()

# ---------------------------
# Helpers for models & data (Definisi Global)
# ---------------------------
@st.cache_resource
def load_models():
    tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    kmeans_model = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    knn_model = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
    embeddings = joblib.load(os.path.join(MODELS_DIR, "embeddings.pkl"))
    tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"))
    return tfidf_vectorizer, kmeans_model, knn_model, embeddings, tfidf_matrix

def ensure_text_column(df):
    if 'text' not in df.columns:
        cols = [c for c in ['title','subtitle','authors','categories','description'] if c in df.columns]
        if cols:
            df['text'] = df[cols].fillna('').agg(' '.join, axis=1)
        else:
            df['text'] = df['title'].astype(str)
    return df

@st.cache_data
def load_books(file_path):
    if not os.path.exists(file_path):
        st.sidebar.warning(f"{file_path} tidak ditemukan. Gunakan tab 'Upload Data'.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        df = ensure_text_column(df)
        df['title_norm'] = df['title'].astype(str).str.strip()
        return df
    except Exception as e:
        st.sidebar.error(f"Gagal membaca {file_path}: {e}")
        return pd.DataFrame()


def fuzzy_match(query, choices):
    try:
        from thefuzz import process
        match = process.extractOne(query, choices)
        if match:
            return match[0], match[1]
    except Exception:
        import difflib
        matches = difflib.get_close_matches(query, choices, n=1)
        if matches:
            return matches[0], 80
    return None, 0

# ---------------------------
# UI Header (UNAIR-like, dari File 1)
# ---------------------------
st.markdown("""
<style>
.header { background-color:#002855; color:white; padding:16px; border-radius:6px; text-align:center; }
.navbar { background-color:#ffd100; padding:8px; border-radius:6px; display:flex; gap:12px; justify-content:center; font-weight:600;}
.card { background:white; padding:12px; border-radius:10px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h2>Book Recommender</h2></div>', unsafe_allow_html=True)


# ---------------------------
st.sidebar.title("Account")
menu = st.sidebar.selectbox("Menu", ["Login", "Register", "Profile"])

if menu == "Register":
    st.sidebar.subheader("🔐 Buat Akun")
    new_user = st.sidebar.text_input("Username", key="reg_user")
    new_pass = st.sidebar.text_input("Password", type="password", key="reg_pass")
    if st.sidebar.button("Daftar"):
        ok, msg = register_user(new_user, new_pass)
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)

elif menu == "Login":
    st.sidebar.subheader("🔑 Login")
    in_user = st.sidebar.text_input("Username", key="login_user")
    in_pass = st.sidebar.text_input("Password", type="password", key="login_pass")
    if st.sidebar.button("Login"):
        if authenticate_user(in_user, in_pass):
            st.session_state['username'] = in_user
            st.session_state['logged_in'] = True
            st.sidebar.success(f"Selamat datang, {in_user}!")
            st.rerun() 
        else:
            st.sidebar.error("Username/password salah")

elif menu == "Profile":
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        u = st.session_state['username']
        st.sidebar.write(f"Logged in as **{u}**")
        
        history = get_history(u)
        if history:
            st.sidebar.write("Riwayat pencarian:")
            for h in history[:8]:
                t = pd.to_datetime(h[0], unit='s').strftime("%Y-%m-%d %H:%M")
                st.sidebar.write(f"- {t} — {h[1]} ({h[2]})")
        
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()
    else:
        st.sidebar.info("Anda belum login. Silakan login.")


if 'logged_in' not in st.session_state or not st.session_state.get('logged_in'):
    st.info("ℹ️ Silakan login atau register melalui menu di sidebar untuk menggunakan aplikasi.")
else:
    try:
        tfidf, kmeans_model, knn_model, embeddings,tfidf_matrix = load_models()
    except FileNotFoundError:
        st.error("Gagal memuat file model. Pastikan folder 'model' dan file .pkl ada.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.stop()

    books_df = load_books(BOOKS_CSV)
  
    tab = option_menu(
        menu_title=None, 
        options=["Home", "Recommender", "Clusters" ,"About"], # "Clusters",
        icons=["house-door-fill", "star-fill", "search", "cloud-upload-fill", "info-circle-fill"], 
        orientation="horizontal",
        styles={
            "container": {
                "padding": "8px", 
                "background-color": "#ffd100", 
                "border-radius": "6px",
                "gap": "12px",
                "justify-content": "center"
            },
            "nav-link": {
                "font-weight": "600",
                "color": "#002855",
                "text-align": "center",
                "--hover-color": "#eee"
            },
            "nav-link-selected": {
                "background-color": "#002855", 
                "color": "white"
            },
        }
    )

  
    
    # ------- Home -------
    if tab == "Home":
        if not books_df.empty:
            st.write(f"Dataset: {len(books_df)} buku")
            st.dataframe(books_df[['title','authors','categories']].head(8))
        else:
            st.info("Dataset belum dimuat. Pergi ke tab 'Upload Data' untuk mengunggah 'data - books.csv'.")

    # ------- Recommender -------
    elif tab == "Recommender":
        st.header("🔎 Book Recommender (Metode KNN)")
        st.markdown("Cari judul buku yang ada di database. Sistem akan menemukan buku-buku lain yang paling mirip berdasarkan *item-based collaborative filtering* (sesuai jurnal Devika, dkk. [cite: 140]).")
    
   
        query = st.text_input("Cari judul buku:", value="Harry Potter and the Sorcerer's Stone (Book 1)")

   
        top_k = st.slider("Jumlah Hasil", 3, 12, 5) 


        cand = None
        if query and not books_df.empty:
            match, score = fuzzy_match(query, books_df['title_norm'].tolist())
        
            if score > 70:
                cand = match
                st.caption(f"Buku ditemukan di database: **{cand}** (skor {score})")
            else:
                st.caption("Ketik judul buku untuk memulai...")

    # --- Tombol Aksi ---
        if st.button("Dapatkan Rekomendasi"):
            with st.spinner("Mencari buku-buku yang mirip... ⏳"):
            
                if books_df.empty:
                    st.error("Dataset tidak tersedia.")
            
        
                elif knn_model is None or embeddings is None:
                    st.error("Model (KNN / Embeddings) tidak tersedia.")
            
            # Periksa apakah buku ditemukan di database
                elif not cand:
                    st.warning("Buku tidak ditemukan di database. Coba ketik judul yang lebih spesifik.")
                else:
                # --- Logika KNN (Sesuai Jurnal) ---
                    try:
                    
                        idx_query = books_df[books_df['title_norm'] == cand].index[0]
                    
                 
                        emb_array = np.array(embeddings)
                        q_emb = emb_array[idx_query].reshape(1, -1)
                    
                    # 
                        dists, idxs = knn_model.kneighbors(q_emb, n_neighbors=top_k + 1)
                    
                        top_idx = idxs.flatten()
                        top_dists = dists.flatten()

                   
                        st.subheader(f"Buku yang Mirip dengan '{cand}':")
                    
                    # Mulai dari 1 untuk melewati buku itu sendiri
                        if len(top_idx) > 1:
                            for i in range(1, len(top_idx)):
                                idx = top_idx[i]
                                dist = top_dists[i]
                                buku = books_df.iloc[idx]
                            
                                with st.container(border=True):
                                    st.markdown(f"**{buku['title']}**")
                                    st.caption(f"Penulis: {buku.get('authors', 'N/A')}")
                                
                               
                                    st.markdown(f"**Distance: {dist:.4f}**")
                    
                        # Simpan riwayat
                            if 'username' in st.session_state:
                                add_history(st.session_state['username'], query, "KNN (Item-Based)")
                    
                        else:
                            st.info("Tidak ada buku lain yang ditemukan.")

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat pemrosesan KNN: {e}")
    # ------- Clusters -------
    elif tab == "Clusters":
        st.header("Jelajahi Cluster Buku (K-Means)")
    
        if books_df.empty:
            st.warning("Dataset tidak tersedia.")
    
    # Periksa apakah model & matriks (dari load_models) sudah siap
        elif tfidf_matrix is None or kmeans_model is None:
            st.error("Model K-Means atau TF-IDF Matrix tidak dimuat. Fitur ini tidak tersedia.")
    
        else:
            with st.spinner("Menganalisis dan memprediksi cluster... ⏳"):
                try:
                # 1. Gunakan model yang SUDAH DIMUAT untuk memprediksi
                # Ini jauh lebih cepat daripada melatih ulang
                    cluster_labels = kmeans_model.predict(tfidf_matrix)
                    books_df['cluster'] = cluster_labels
                except Exception as e:
                    st.error(f"Gagal memprediksi cluster: {e}")
                # Hentikan eksekusi jika prediksi gagal
                    st.stop() 

        # --- Tampilkan Visualisasi (Grafik Batang) ---
            st.subheader("Distribusi Buku per Cluster")
            cluster_counts = books_df["cluster"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Jumlah Buku"]
        
            try:
            # Gunakan Plotly (jika sudah diimpor) untuk grafik interaktif
                import plotly.express as px
                fig = px.bar(cluster_counts.sort_values('Cluster'), 
                             x="Cluster", 
                             y="Jumlah Buku",
                             color="Cluster", 
                             title="Distribusi Buku per Cluster")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
            # Fallback ke bar_chart bawaan Streamlit jika plotly tidak ada
                st.bar_chart(cluster_counts.set_index('Cluster'))

        # --- Tampilkan Penjelajah Cluster (DataFrame) ---
            st.subheader("Jelajahi Isi Cluster")
        
        # Buat nama cluster (opsional tapi sangat disarankan)
        # SESUAIKAN NAMA INI dengan hasil analisis Anda
            cluster_names = {
                1: "Fiction",
                2: "Juvenile Fiction",
                3: "Other",
                # ... tambahkan sesuai jumlah 'k' Anda
            }
        
        # Ambil cluster unik dari data yang sudah diprediksi
            unique_clusters = sorted(books_df['cluster'].unique())
        
        # Buat label yang lebih deskriptif untuk selectbox
            display_options = [f"Cluster {i}: {cluster_names.get(i, 'Umum')}" for i in unique_clusters]
        
        # Tampilkan selectbox
            selected_display_name = st.selectbox("Pilih cluster untuk dijelajahi:", display_options)
        
        # Dapatkan angka cluster dari nama yang dipilih
            selected_cluster_index = display_options.index(selected_display_name)
            selected_cluster = unique_clusters[selected_cluster_index]

        # Tampilkan DataFrame untuk cluster yang dipilih
            st.dataframe(books_df[books_df['cluster'] == selected_cluster][['title', 'authors', 'categories']].head(50), 
                         use_container_width=True)
    # ------- About -------
    elif tab == "About":
        st.header("Tentang Aplikasi Ini")
        st.write("Aplikasi portofoli saya buat sendiri. terinspirasi dari pengalaman saya menggunakan website / OPAC di perpustakaan. saya menggunakanTF-IDF, embeddings, KNN, dan KMeans untuk rekomendasi buku.")
        st.write("referensi: Devika, P. V., Jyothisree, K., Rahul, P. V., Arjun, S., & Narayanan, J. (2021, July). Book recommendation system. In 2021 12th International Conference on Computing Communication and Networking Technologies (ICCCNT) (pp. 1-5). IEEE.")
        st.write("Model file dimuat dari folder `/model`. Database pengguna disimpan di `users.db`.")
        st.write("Untuk produksi: amankan kredensial, gunakan DB remote, dan jangan commit dataset privat ke GitHub.")

# Footer (diletakkan di luar 'else' agar selalu tampil)
st.markdown("---")
st.caption("© Nanda — Book Recommender Portfolio. Gunakan secara bertanggung jawab.")


