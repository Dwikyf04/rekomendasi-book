# app.py
# Book Recommender Portfolio — Nanda
# VERSI FINAL MAKSIMAL:
# UI (UNAIR) + DB (Supabase) + MODEL (Joblib 5 file)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os, hashlib # Hapus 'json' dan 'time'
from textwrap import shorten

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu

# Opsional: semantic embedding
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
# USERS_FILE dihapus, kita pakai Supabase
BOOKS_CSV = os.path.join(ROOT, "data - books.csv")

# ---------------------------
# KONEKSI DATABASE (Supabase)
# Menggantikan seluruh bagian 'USERS.JSON MANAGEMENT' dan 'sqlite3'
# ---------------------------
try:
    conn = st.connection("supabase_db", type="sql")
    DB_CONNECTED = True
except Exception:
    DB_CONNECTED = False
    st.sidebar.error("Koneksi database (Supabase) gagal. Periksa secrets.toml Anda.")

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def register_user(username, password):
    if not DB_CONNECTED: return False, "Database tidak terhubung."
    hashed_pw = make_hashes(password)
    try:
        with conn.session as s:
            s.execute(
                'INSERT INTO users(username, password) VALUES ($1, $2)',
                params=(username, hashed_pw)
            )
            s.commit()
        return True, "✅ Registrasi berhasil."
    except Exception as e:
        if "UniqueViolation" in str(e):
             return False, "❌ Username sudah terdaftar."
        return False, f"Error: {e}"

def authenticate_user(username, password):
    if not DB_CONNECTED: return False
    hashed_pw = make_hashes(password)
    result_df = conn.query(
        'SELECT * FROM users WHERE username = $1 AND password = $2',
        params=(username, hashed_pw),
        ttl=0 # Nonaktifkan cache untuk login
    )
    return not result_df.empty

def add_history(username, query, method):
    if not DB_CONNECTED: return
    timestamp = int(pd.Timestamp.now().timestamp())
    try:
        with conn.session as s:
            s.execute(
                'INSERT INTO history(username, timestamp, query, method) VALUES ($1, $2, $3, $4)',
                params=(username, timestamp, query, method)
            )
            s.commit()
    except Exception as e:
        st.error(f"Gagal menyimpan riwayat: {e}")

def get_history(username):
    if not DB_CONNECTED: return pd.DataFrame()
    # Langsung kembalikan DataFrame
    df = conn.query(
        "SELECT timestamp, query, method FROM history WHERE username=$1 ORDER BY timestamp DESC LIMIT 20",
        params=(username,)
    )
    return df

# ---------------------------
# LOAD MODELS
# ---------------------------
@st.cache_resource
def load_models():
    # Fungsi ini memuat SEMUA 5 file .pkl yang sudah sinkron
    def safe_load(name):
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            return joblib.load(path)
        else:
            # Ini akan menghentikan aplikasi jika file model utama tidak ada
            st.error(f"FATAL: Model tidak ditemukan di {path}. Harap jalankan 'train.py' dan upload file .pkl.")
            return None

    tfidf = safe_load("tfidf_vectorizer.pkl")
    kmeans = safe_load("kmeans_model.pkl")
    knn = safe_load("knn_model.pkl")
    embeddings = safe_load("embeddings.pkl")
    tfidf_matrix = safe_load("tfidf_matrix.pkl")
    
    # Memuat SBERT di sini
    sbert_model = None
    if HAS_SBERT:
        sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    else:
        st.warning("Gagal memuat SentenceTransformer. Fitur Recommender mungkin gagal.")

    return tfidf, kmeans, knn, embeddings, tfidf_matrix, sbert_model

def ensure_text_column(df):
    if 'text' not in df.columns:
        cols = [c for c in ['title','subtitle','authors','categories','description'] if c in df.columns]
        df['text'] = df[cols].fillna('').agg(' '.join, axis=1) if cols else df['title'].astype(str)
    return df

@st.cache_data
def load_books(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    df = ensure_text_column(df)
    df['title_norm'] = df['title'].astype(str).str.strip()
    return df

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
# STYLE UNAIR
# ---------------------------
st.markdown("""
<style>
.header { background-color:#002855; color:white; padding:16px; border-radius:6px; text-align:center; }
.navbar { background-color:#ffd100; padding:8px; border-radius:6px; display:flex; gap:12px; justify-content:center; font-weight:600;}
.card { background:white; padding:12px; border-radius:10px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="header"><h2>Book Recommender (Nanda)</h2></div>', unsafe_allow_html=True)

# ---------------------------
# SIDEBAR: LOGIN / REGISTER (Menggunakan backend Supabase)
# ---------------------------
st.sidebar.title("👤 Account")

menu = st.sidebar.selectbox("Menu", ["Login","Register","Profile"])

if menu == "Register":
    st.sidebar.subheader("🔐 Register Akun Baru")
    user = st.sidebar.text_input("Username", key="r_user")
    pwd = st.sidebar.text_input("Password", type="password", key="r_pass")
    if st.sidebar.button("Daftar"):
        ok, msg = register_user(user, pwd)
        (st.sidebar.success if ok else st.sidebar.error)(msg)

elif menu == "Login":
    st.sidebar.subheader("🔑 Login")
    user = st.sidebar.text_input("Username", key="l_user")
    pwd = st.sidebar.text_input("Password", type="password", key="l_pass")
    if st.sidebar.button("Login"):
        if authenticate_user(user, pwd):
            st.session_state['logged_in'] = True
            st.session_state['username'] = user
            st.sidebar.success(f"Selamat datang, {user}!")
            st.rerun()
        else:
            st.sidebar.error("Username / Password salah.")

elif menu == "Profile":
    if st.session_state.get('logged_in'):
        u = st.session_state['username']
        st.sidebar.write(f"👋 Hi, {u}")
        
        # PERBAIKAN: Membaca DataFrame dari Supabase
        history_df = get_history(u)
        if not history_df.empty:
            st.sidebar.markdown("📖 Riwayat Pencarian Terakhir:")
            history_df['Waktu'] = pd.to_datetime(history_df['timestamp'], unit='s').dt.strftime("%Y-%m-%d %H:%M")
            for _, row in history_df.head(8).iterrows():
                st.sidebar.caption(f"{row['Waktu']} — {row['query']} ({row['method']})")
        
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()
    else:
        st.sidebar.info("Silakan login dulu.")

# ---------------------------
# MAIN CONTENT (LOGIN-GATED)
# ---------------------------
if not st.session_state.get('logged_in'):
    st.info("Silakan login atau register terlebih dahulu untuk mengakses rekomendasi buku.")
    st.stop()

# --- Model & Data Dimuat HANYA SETELAH LOGIN ---
try:
    tfidf, kmeans_model, knn_model, embeddings, tfidf_matrix, sbert_model = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

books_df = load_books(BOOKS_CSV)

# Cek kritis jika model dan data tidak sinkron
if not books_df.empty and len(books_df) != embeddings.shape[0]:
    st.error(f"CRITICAL ERROR: Data dan Model tidak sinkron. Data CSV memiliki {len(books_df)} baris, tetapi model Embeddings memiliki {embeddings.shape[0]} baris.")
    st.info("Jalankan 'train.py' secara lokal dan upload ulang semua file .pkl di folder 'model/' Anda.")
    st.stop()
    
# --- Tampilan Tab (Navbar) ---
tab = option_menu(
    menu_title=None,
    options=["Home", "Recommender", "Clusters", "About"],
    icons=["house", "search", "grid", "info-circle"],
    orientation="horizontal",
    styles={
        "container": {"background-color": "#ffd100", "padding": "8px", "border-radius": "8px"},
        "nav-link": {"font-weight": "600", "color": "#002855"},
        "nav-link-selected": {"background-color": "#002855", "color": "white"},
    }
)

# ------- HOME -------
if tab == "Home":
    if not books_df.empty:
        st.markdown('📖 Dataset Buku', unsafe_allow_html=True)
        st.write(f"Total buku: {len(books_df)}")
        st.dataframe(books_df[['title','authors','categories']].head(8))
    else:
        st.warning("Dataset 'data - books.csv' belum dimuat.")

# ------- RECOMMENDER (KNN Sesuai Jurnal) -------
elif tab == "Recommender":
    st.header("🔎 Book Recommender (Metode KNN)")
    st.markdown("Cari judul buku yang ada di database. Sistem akan menemukan buku-buku lain yang paling mirip.")

    query = st.text_input("Masukkan judul buku:", value="")
    top_k = st.slider("Jumlah rekomendasi:", 3, 10, 5)

    cand = None
    if query:
        cand, score = fuzzy_match(query, books_df['title_norm'].tolist())
        if cand and score > 70: # Threshold
            st.caption(f"Mungkin maksud Anda: {cand} (skor {score})")
        else:
            cand = None # Reset jika skor terlalu rendah

    if st.button("Rekomendasikan"):
        if not cand:
            st.warning("Judul tidak ditemukan di database. Coba ketik lebih spesifik.")
        else:
            with st.spinner("Mencari..."):
                try:
                    # 1. Dapatkan index dari buku yang di-match
                    idx = books_df[books_df['title_norm'] == cand].index[0]
                    
                    # 2. Ambil embedding dari index tersebut (sudah sinkron)
                    q_emb = embeddings[idx].reshape(1, -1)
                    
                    # 3. Gunakan model KNN
                    dists, idxs = knn_model.kneighbors(q_emb, n_neighbors=top_k+1)

                    st.subheader(f"Hasil rekomendasi untuk: {cand}")
                    # Mulai dari [1:] untuk skip buku itu sendiri
                    for dist, i in zip(dists.flatten()[1:], idxs.flatten()[1:]):
                        b = books_df.iloc[i]
                        with st.container(border=True): # Tampilan Kartu
                            st.markdown(f"**{b['title']}** — *{b.get('authors','')}*")
                            st.caption(f"Kategori: {b.get('categories', 'N/A')}")
                            st.write(shorten(str(b.get('description', '')), width=200, placeholder="..."))
                            st.caption(f"Distance: {dist:.4f} (Semakin kecil semakin mirip)")

                    add_history(st.session_state['username'], query, "KNN")
                except Exception as e:
                    st.error(f"Error saat rekomendasi: {e}")

# ------- CLUSTERS (K-Means) -------
elif tab == "Clusters":
    st.header("📂 K-Means Clustering")
    if books_df.empty or kmeans_model is None or tfidf_matrix is None:
        st.warning("Data atau model (KMeans/TFIDF Matrix) belum dimuat.")
    else:
        try:
            # Prediksi (sekarang sudah sinkron)
            cluster_labels = kmeans_model.predict(tfidf_matrix)
            books_df['cluster'] = cluster_labels
            
            st.subheader("Distribusi Buku per Cluster")
            st.write(books_df.groupby('cluster').size().reset_index(name='count'))
            
            st.subheader("Jelajahi Isi Cluster")
            sel = st.selectbox("Pilih cluster:", sorted(books_df['cluster'].unique()))
            st.dataframe(books_df[books_df['cluster']==sel][['title','authors','categories']].head(50))
        except Exception as e:
            st.error(f"Error saat prediksi cluster: {e}. Pastikan model sinkron.")

# ------- ABOUT -------
elif tab == "About":
    st.header("Tentang Aplikasi")
    st.write("""
    Aplikasi ini dikembangkan oleh Nanda (Universitas Airlangga)
    Menggunakan pendekatan TF-IDF, Embeddings, KNN, dan K-Means untuk merekomendasikan buku.
    Terinspirasi oleh sistem OPAC perpustakaan dan penelitian:
    Devika et al., (2021). Book Recommendation System. ICCCNT.
    """)
    st.caption("© 2025 — Nanda | Book Recommender Portfolio")
