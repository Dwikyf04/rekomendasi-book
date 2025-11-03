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
tab = option_menu(
    menu_title=None, 
    options=["Home", "Recommender", "About"], # "Clusters",
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

# ---------------------------
# Sidebar: Login / Register (Struktur dari File 1, Backend dari File 2)
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
            st.rerun() # Ganti experimental_rerun
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

# ----------------------------------------------------
# MAIN APP - HANYA JALAN JIKA SUDAH LOGIN
# ----------------------------------------------------
if 'logged_in' not in st.session_state or not st.session_state.get('logged_in'):
    st.info("ℹ️ Silakan login atau register melalui menu di sidebar untuk menggunakan aplikasi.")
else:
    # --- PANGGILAN MODEL & DATA DIPINDAHKAN KE SINI ---
    try:
        tfidf, kmeans_model, knn_model, embeddings,tfidf_matrix = load_models()
    except FileNotFoundError:
        st.error("Gagal memuat file model. Pastikan folder 'model' dan file .pkl ada.")
        st.stop()
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.stop()

    books_df = load_books(BOOKS_CSV)
    # --- AKHIR DARI PEMUATAN ---

    # --- UI Utama (Tabs) sekarang ada di dalam 'else' ---
    #tab = st.selectbox("", ["Home", "Recommender", "Clusters", "Upload Data", "About"])

    # ------- Home -------
    if tab == "Home":
        if not books_df.empty:
            st.write(f"Dataset: {len(books_df)} buku")
            st.dataframe(books_df[['title','authors','categories']].head(8))
        else:
            st.info("Dataset belum dimuat. Pergi ke tab 'Upload Data' untuk mengunggah 'data - books.csv'.")

    # ------- Recommender -------
    # GANTI SELURUH BLOK 'elif tab == "Recommender":' DENGAN INI

    elif tab == "Recommender":
    st.header("🔎 Book Recommender")
    st.markdown("Cari buku (dengan toleransi typo). Sistem akan otomatis menggunakan metode **Hybrid** (TF-IDF + Embedding) untuk menemukan hasil terbaik.")
    
    # --- Input Kueri ---
    query = st.text_input("Cari judul buku (typo OK):", value="")

    # --- Pengaturan Rekomendasi (Sebelum Tombol) ---
    st.subheader("Pengaturan Rekomendasi")
    col_k, col_alpha = st.columns(2)
    with col_k:
        top_k = st.slider("Jumlah Hasil (Top K)", 3, 12, 6)
    with col_alpha:
        # 'alpha' sekarang menjadi slider utama, bukan di dalam 'if'
        alpha = st.slider("Bobot Embedding (alpha)", 0.0, 1.0, 0.5, 
                          help="0.0 = Hanya kata kunci (TF-IDF), 1.0 = Hanya makna (Embedding)")

    # --- Fuzzy Match (Deteksi Typo) ---
    cand = None
    if query and not books_df.empty:
        match, score = fuzzy_match(query, books_df['title_norm'].tolist())
        
        # Ambang batas agar tidak salah tebak
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
                    s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                    q_emb = s_model.encode(query_text, convert_to_tensor=False).reshape(1,-1)
                    emb_sim = cosine_similarity(q_emb, emb_array).flatten()
                except Exception as e_emb:
                    st.error(f"Error Embedding: {e_emb}")
                    emb_sim = np.zeros(len(books_df))

                # 4. Normalisasi dan Gabungkan
                # (Baris ini sekarang aman karena 'tfidf_matrix' dan 'embeddings' sinkron)
                tf_norm = (tfidf_sim - tfidf_sim.min()) / (tfidf_sim.max() - tfidf_sim.min() + 1e-9)
                eb_norm = (emb_sim - emb_sim.min()) / (emb_sim.max() - emb_sim.min() + 1e-9)
                combined = alpha * eb_norm + (1 - alpha) * tf_norm
                
                top_idx = combined.argsort()[::-1][:top_k]
                scores = {i: combined[i] for i in top_idx}

                # --- 5. Tampilkan Hasil (UI Kartu) ---
                st.subheader("Rekomendasi (Metode Hybrid Otomatis)")
                
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
        st.header("📂 K-Means Clustering")
        if books_df.empty:
            st.warning("Dataset tidak tersedia.")
        else:
            k = st.slider("Jumlah clusters (k)", 2, 20, 6)
            if st.button("Bangun/Bangun Ulang KMeans"):
                if tfidf is None:
                    st.error("Model TF-IDF dibutuhkan untuk membangun cluster.")
                else:
                    try:
                        with st.spinner("Menghitung TF-IDF Matrix..."):
                            X = tfidf.transform(books_df['text'].fillna('').tolist())
                        with st.spinner(f"Melatih K-Means dengan k={k}..."):
                            km = KMeans(n_clusters=k, random_state=42, n_init=10)
                            labels = km.fit_predict(X)
                        
                        books_df['cluster'] = labels
                        os.makedirs(MODELS_DIR, exist_ok=True)
                        joblib.dump(km, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
                        st.success(f"KMeans (k={k}) dibangun dan disimpan ke models/kmeans_model.pkl")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error K-Means: {e}")
            
            cluster_labels = None
            if 'cluster' in books_df.columns:
                 cluster_labels = books_df['cluster']
            elif kmeans_model and tfidf:
                try:
                    X = tfidf.transform(books_df['text'].fillna('').tolist())
                    cluster_labels = kmeans_model.predict(X)
                    books_df['cluster'] = cluster_labels
                except Exception:
                    st.info("Cluster default dari model tidak dapat dimuat.")

            if cluster_labels is not None:
                st.write(books_df.groupby('cluster').size().reset_index(name='count'))
                sel = st.selectbox("Pilih cluster", sorted(books_df['cluster'].unique()))
                st.dataframe(books_df[books_df['cluster']==sel][['title','authors','categories']].head(50))


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


















