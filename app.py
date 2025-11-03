# app.py
# Book Recommender Portfolio — Nanda
# FINAL MAKSIMAL: UI (UNAIR) + DB (Supabase API) + Model (Joblib 5 file)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os, hashlib
from textwrap import shorten

# --- Impor yang Diubah ---
# HAPUS: import sqlite3
from supabase import create_client, Client # <-- DIUBAH: Impor Supabase
# -------------------------

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
# HAPUS: DB_FILE = os.path.join(ROOT, "users.db")
BOOKS_CSV = os.path.join(ROOT, "data - books.csv")

# ---------------------------
# Style (Tetap Sama)
# ---------------------------
st.markdown("""
    <style>
        body { background-color: #F8FAFC; }
        /* ... (sisa CSS Anda tetap sama) ... */
        .stButton>button:hover { background-color: #00509E; }
        h1, h2, h3 { color: #003366; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# KONEKSI DATABASE (Supabase API)
# DIUBAH: Menggantikan seluruh blok 'sqlite3'
# ---------------------------
DB_CONNECTED = False
supabase: Client = None # Inisialisasi

try:
    # Membaca dari Streamlit Secrets
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(url, key)
    DB_CONNECTED = True
except Exception as e:
    st.sidebar.error("Koneksi Supabase gagal. Periksa secrets.toml Anda.")
    st.sidebar.error(f"Error: {e}")

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def register_user(username, password):
    if not DB_CONNECTED: return False, "Database tidak terhubung."
    hashed_pw = make_hashes(password)
    
    try:
        # Sintaks Supabase API
        response = supabase.table('users').insert({
            "username": username,
            "password": hashed_pw
        }).execute()
        
        if response.error:
            if "duplicate key value" in response.error.message or "UniqueViolation" in str(response.error):
                return False, "❌ Username sudah terdaftar."
            return False, f"Error: {response.error.message}"
        
        return True, "✅ Registrasi berhasil."
    except Exception as e:
        return False, f"Error: {e}"

def authenticate_user(username, password):
    if not DB_CONNECTED: return False
    hashed_pw = make_hashes(password)
    
    try:
        # Sintaks Supabase API
        response = supabase.table('users').select('*').eq('username', username).eq('password', hashed_pw).execute()
        
        if response.data and len(response.data) > 0:
            return True # Pengguna ditemukan
        return False # Pengguna tidak ditemukan
    except Exception as e:
        st.error(f"Error login: {e}")
        return False

def add_history(username, query, method):
    if not DB_CONNECTED: return
    timestamp = int(pd.Timestamp.now().timestamp())
    
    try:
        # Sintaks Supabase API
        response = supabase.table('history').insert({
            "username": username,
            "timestamp": timestamp,
            "query": query,
            "method": method
        }).execute()
        
        if response.error:
            st.error(f"Gagal menyimpan riwayat: {response.error.message}")
    except Exception as e:
        st.error(f"Gagal menyimpan riwayat: {e}")

def get_history(username):
    if not DB_CONNECTED: return pd.DataFrame() # Kembalikan DataFrame kosong
    
    try:
        # Sintaks Supabase API
        response = supabase.table('history').select('timestamp, query, method').eq('username', username).order('timestamp', desc=True).limit(20).execute()
        
        if response.data:
            # Kembalikan sebagai DataFrame agar kompatibel dengan sisa kode Anda
            return pd.DataFrame(response.data)
        return pd.DataFrame() # Kembalikan DataFrame kosong jika tidak ada riwayat
    except Exception as e:
        st.error(f"Gagal mengambil riwayat: {e}")
        return pd.DataFrame()

# HAPUS: create_usertable() (sudah dibuat di Supabase SQL Editor)
# ---------------------------

# ---------------------------
# Helpers for models & data (Definisi Global)
# ---------------------------
@st.cache_resource
def load_models():
    # Fungsi ini sekarang memuat SEMUA 5 file .pkl
    # dan juga model SBERT itu sendiri
    if HAS_SBERT:
        sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    else:
        sbert_model = None
        
    tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    kmeans_model = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    knn_model = joblib.load(os.path.join(MODELS_DIR, "knn_model.pkl"))
    embeddings = joblib.load(os.path.join(MODELS_DIR, "embeddings.pkl")) # NumPy array
    tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"))
    
    # st.success("Semua model (5 file .pkl + SBERT) berhasil dimuat.")
    return tfidf_vectorizer, kmeans_model, knn_model, embeddings, tfidf_matrix, sbert_model

def ensure_text_column(df):
    if 'text' not in df.columns:
        cols = [c for c in ['title','subtitle','authors','categories','description'] if c in df.columns]
        df['text'] = df[cols].fillna('').agg(' '.join, axis=1) if cols else df['title'].astype(str)
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
# Sidebar: Login / Register (Struktur dari File 1, Backend dari Supabase)
# ---------------------------
st.sidebar.title("Account")
menu = st.sidebar.selectbox("Menu", ["Login", "Register", "Profile"])

if menu == "Register":
    st.sidebar.subheader("🔐 Buat Akun")
    new_user = st.sidebar.text_input("Username", key="reg_user")
    new_pass = st.sidebar.text_input("Password", type="password", key="reg_pass")
    if st.sidebar.button("Daftar"):
        ok, msg = register_user(new_user, new_pass)
        (st.sidebar.success if ok else st.sidebar.error)(msg)

elif menu == "Login":
    st.sidebar.subheader("🔑 Login")
    in_user = st.sidebar.text_input("Username", key="login_user")
    in_pass = st.sidebar.text_input("Password", type="password", key="login_pass")
    if st.sidebar.button("Login"):
        if authenticate_user(in_user, in_pass):
            st.session_state['username'] = in_user
            st.session_state['logged_in'] = True
            st.sidebar.success(f"Selamat datang, {in_user}!")
            st.rerun() # Ganti experimental_rerun
        else:
            st.sidebar.error("Username/password salah")

elif menu == "Profile":
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        u = st.session_state['username']
        st.sidebar.write(f"Logged in as **{u}**")
        
        # DIUBAH: Membaca DataFrame dari Supabase
        history_df = get_history(u)
        
        if not history_df.empty:
            st.sidebar.markdown("📖 Riwayat Pencarian Terakhir:")
            # Format timestamp
            history_df['Waktu'] = pd.to_datetime(history_df['timestamp'], unit='s').dt.strftime("%Y-%m-%d %H:%M")
            # Loop DataFrame
            for _, row in history_df.head(8).iterrows():
                st.sidebar.caption(f"{row['Waktu']} — {row['query']} ({row['method']})")
        
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()
    else:
        st.sidebar.info("Anda belum login. Silakan login.")

# ----------------------------------------------------
# MAIN APP - HANYA JALAN JIKA SUDAH LOGIN
# ----------------------------------------------------
if 'logged_in' not in st.session_state or not st.session_state.get('logged_in'):
    st.info("ℹ️ Silakan login atau register melalui menu di sidebar untuk menggunakan aplikasi.")
else:
    # --- PANGGILAN MODEL & DATA DIPINDAHKAN KE SINI ---
    try:
        tfidf, kmeans_model, knn_model, embeddings,tfidf_matrix, sbert_model = load_models()
    except FileNotFoundError:
        st.error("Gagal memuat file model. Pastikan folder 'model' dan file .pkl ada.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.stop()

    books_df = load_books(BOOKS_CSV)
    
    # Cek kritis jika model dan data tidak sinkron
    if not books_df.empty and len(books_df) != embeddings.shape[0]:
        st.error(f"CRITICAL ERROR: Data dan Model tidak sinkron. Data CSV memiliki {len(books_df)} baris, tetapi model Embeddings memiliki {embeddings.shape[0]} baris.")
        st.info("Jalankan 'train.py' secara lokal dan upload ulang semua file .pkl di folder 'model/' Anda.")
        st.stop()
    
    # --- AKHIR DARI PEMUATAN ---

    # --- UI Utama (Tabs) sekarang ada di dalam 'else' ---
    tab = option_menu(
        menu_title=None, 
        options=["Home", "Recommender", "Clusters" ,"About"],
        icons=["house-door-fill", "star-fill", "search", "info-circle-fill"], 
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
            st.info("Dataset 'data - books.csv' tidak dimuat.")

    # ------- Recommender -------
    elif tab == "Recommender":
        st.header("🔎 Book Recommender")
        st.markdown("Cari buku anda.")
    
    # --- Input Kueri ---
        query = st.text_input("Cari judul buku:", value="")

    # --- Pengaturan Rekomendasi (Sebelum Tombol) ---
        st.subheader("Pengaturan Rekomendasi")
        col_k, col_alpha = st.columns(2)
        with col_k:
            top_k = st.slider("Jumlah Hasil", 3, 12, 6)
        with col_alpha:
        # 'alpha' sekarang menjadi slider utama, bukan di dalam 'if'
            alpha = st.slider("Bobot Embedding (alpha)", 0.0, 1.0, 0.5, 
                             help="0.0 = Hanya kata kunci (TF-IDF), 1.0 = Hanya makna (Embedding)")
    # --- Fuzzy Match (Deteksi Typo) ---
        cand = None
        if query and not books_df.empty:
        # 1. Dapatkan skor DARI DALAM BLOK INI
            match, score = fuzzy_match(query, books_df['title_norm'].tolist())
        
        # 2. Periksa skor (baris 291) - HARUS DI DALAM BLOK INI JUGA
            if score > 70: 
                cand = match
                st.caption(f"Mungkin maksud Anda: **{cand}** (skor {score})")
            else:
                st.caption("Tidak ada judul yang mirip, menggunakan pencarian teks penuh...")

    # --- Tombol Aksi ---
        if st.button("Dapatkan Rekomendasi"):
            with st.spinner("Menganalisis dan mencari rekomendasi terbaik... ⏳"):
            
                if books_df.empty:
                    st.error("Dataset tidak tersedia.")
            
            # Periksa apakah model & matriks (dari perbaikan error sebelumnya) sudah siap
                elif tfidf_matrix is None or embeddings is None or not HAS_SBERT:
                    st.error("Model yang dibutuhkan (TF-IDF Matrix / Embeddings / SBERT) tidak tersedia.")
            
                else:
                # --- Logika HYBRID (Satu-satunya metode) ---
                
                # 1. Tentukan Kueri (Hasil typo atau teks asli)
                    query_text = query if not cand else cand
                
                # 2. Skor TF-IDF
                    try:
                        vec_tfidf = tfidf.transform([query_text])
                        tfidf_sim = cosine_similarity(vec_tfidf, tfidf_matrix).flatten()
                    except Exception as e_tfidf:
                        st.error(f"Error TF-IDF: {e_tfidf}")
                        tfidf_sim = np.zeros(len(books_df))

                # 3. Skor Embedding
                    try:
                        emb_array = np.array(embeddings)
                        # s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') # Sudah di-load
                        q_emb = sbert_model.encode(query_text, convert_to_tensor=False).reshape(1,-1)
                        emb_sim = cosine_similarity(q_emb, emb_array).flatten()
                    except Exception as e_emb:
                        st.error(f"Error Embedding: {e_emb}")
                        emb_sim = np.zeros(len(books_df))

                # 4. Normalisasi dan Gabungkan
                    tf_norm = (tfidf_sim - tfidf_sim.min()) / (tfidf_sim.max() - tfidf_sim.min() + 1e-9)
                    eb_norm = (emb_sim - emb_sim.min()) / (emb_sim.max() - emb_sim.min() + 1e-9)
                    combined = alpha * eb_norm + (1 - alpha) * tf_norm
                
                    top_idx = combined.argsort()[::-1][:top_k]
                    scores = {i: combined[i] for i in top_idx}

                # --- 5. Tampilkan Hasil (UI Kartu) ---
                    st.subheader("Rekomendasi buku")
                
                    if len(top_idx) > 0:
                        for i in top_idx:
                            buku = books_df.iloc[i]
                        # Tampilan kartu yang lebih profesional
                            with st.container(border=True):
                                st.markdown(f"**{buku['title']}**")
                                st.caption(f"Penulis: {buku.get('authors', 'N/A')} | Kategori: {buku.get('categories', 'N/A')}")
                                st.write(shorten(str(buku.get('description', buku.get('text', ''))), width=200, placeholder="..."))
                                st.caption(f"Skor Gabungan: {scores.get(i, 0.0):.4f}")
                    
                    # Simpan riwayat
                        if 'username' in st.session_state:
                            add_history(st.session_state['username'], query, f"HYBRID (a={alpha})")
                    else:
                    # Tangani jika tidak ada hasil
                        st.info("Tidak ada buku yang cocok dengan kriteria Anda.")
                        
    # ------- Clusters -------
    elif tab == "Clusters":
        st.header("Jelajahi Cluster Buku (K-Means)")
    
        if books_df.empty:
            st.warning("Dataset tidak tersedia.")
    
        elif tfidf_matrix is None or kmeans_model is None:
            st.error("Model K-Means atau TF-IDF Matrix tidak dimuat. Fitur ini tidak tersedia.")
    
        else:
            with st.spinner("Menganalisis dan memprediksi cluster... ⏳"):
                try:
                    cluster_labels = kmeans_model.predict(tfidf_matrix)
                    books_df['cluster'] = cluster_labels
                except Exception as e:
                    st.error(f"Gagal memprediksi cluster: {e}")
                    st.stop() 

            st.subheader("Distribusi Buku per Cluster")
            cluster_counts = books_df["cluster"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Jumlah Buku"]
        
            try:
                import plotly.express as px
                fig = px.bar(cluster_counts.sort_values('Cluster'), 
                             x="Cluster", 
                             y="Jumlah Buku",
                             color="Cluster", 
                             title="Distribusi Buku per Cluster")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(cluster_counts.set_index('Cluster'))

            st.subheader("Jelajahi Isi Cluster")
        
            cluster_names = {
                1: "Fiction",
                2: "Juvenile Fiction",
                3: "Other",
                # ... tambahkan sesuai jumlah 'k' Anda
            }
        
            unique_clusters = sorted(books_df['cluster'].unique())
            display_options = [f"Cluster {i}: {cluster_names.get(i, 'Umum')}" for i in unique_clusters]
            selected_display_name = st.selectbox("Pilih cluster untuk dijelajahi:", display_options)
            
            selected_cluster_index = display_options.index(selected_display_name)
            selected_cluster = unique_clusters[selected_cluster_index]

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
