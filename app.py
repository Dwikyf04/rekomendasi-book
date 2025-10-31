# [File: app.py]
# Portofolio App: Book Recommendation System (Versi Multi-Page)
# Disesuaikan: TANPA tfidf_matrix.pkl dan nama folder custom.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import matplotlib.pyplot as plt
import difflib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- Import opsional ---
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    st.warning("Library 'sentence-transformers' tidak ditemukan. Mode Embedding/Hybrid akan dinonaktifkan.")

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

try:
    from thefuzz import process
    HAS_THEFUZZ = True
except ImportError:
    HAS_THEFUZZ = False

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False
    st.error("Butuh 'streamlit_option_menu'. Install: pip install streamlit-option-menu")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    st.error("Scikit-learn (sklearn) tidak ditemukan. Install: pip install scikit-learn")

# ----------------------
# Konfigurasi Halaman
# ----------------------
st.set_page_config(page_title="Rekomendasi Buku", layout="wide")

# ----------------------
# 1. Fungsi Pemuatan Data (Cache)
# ----------------------

@st.cache_resource
def load_models():
    """
    Memuat semua model yang diperlukan dari folder custom Anda.
    """
    # -------------------------------------------------------------------
    # UBAH NAMA FOLDER DI BAWAH INI
    models_path = "folder_model_anda/" 
    # -------------------------------------------------------------------
    
    if not os.path.exists(models_path):
        st.error(f"Folder '{models_path}' tidak ditemukan. Pastikan nama folder sudah benar.")
        return None, None, None, None

    try:
        # Model yang Anda sebutkan:
        tfidf_vec = joblib.load(os.path.join(models_path, "tfidf.pkl"))
        kmeans_model = joblib.load(os.path.join(models_path, "kmean.pkl"))
        knn_model = joblib.load(os.path.join(models_path, "knn.pkl")) # Model KNN dari SBERT

        # Data yang DIBUTUHKAN oleh knn_model
        # Anda masih butuh file ini agar knn.pkl bisa membandingkan query
        sbert_embeddings = joblib.load(os.path.join(models_path, "sbert_embeddings.pkl"))
        
        st.success("Model (4 file) berhasil dimuat.")
        return tfidf_vec, kmeans_model, knn_model, sbert_embeddings
    
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}.")
        st.warning(f"Pastikan 4 file berikut ada di folder '{models_path}': tfidf.pkl, kmean.pkl, knn.pkl, sbert_embeddings.pkl")
        return None, None, None, None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None, None

@st.cache_data
def load_book_data(file_path="book.csv"):
    """
    Memuat data buku (ringkasan) dari file CSV.
    """
    try:
        df = pd.read_csv(file_path)
        df = df.drop_duplicates(subset=[c for c in ['isbn13','title'] if c in df.columns], keep='first')
        df['title'] = df['title'].astype(str)
        
        cols = []
        for c in ['title','subtitle','authors','categories','description']:
            if c in df.columns:
                cols.append(c)
        if cols:
            df['text'] = df[cols].fillna('').agg(' '.join, axis=1)
        else:
            df['text'] = df['title']
        df = df.reset_index(drop=True)
        return df
        
    except FileNotFoundError:
        st.error(f"File data '{file_path}' tidak ditemukan. Silakan upload.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data CSV (Buku): {e}")
        return pd.DataFrame()

# --- Fungsi Bantuan ---
def fuzzy_match(query, choices, limit=1):
    if HAS_THEFUZZ and process is not None:
        res = process.extractOne(query, choices)
        if res:
            return res[0], res[1]
    matches = difflib.get_close_matches(query, choices, n=limit)
    return (matches[0], 90) if matches else (None, 0)

# ----------------------
# 2. Memuat Semua Data & Model
# ----------------------
df_books = load_book_data("book.csv")
tfidf_vectorizer, kmeans_model, knn_model, sbert_embeddings = load_models()

# ----------------------
# 3. Navigasi Sidebar
# ----------------------
with st.sidebar:
    st.image("Logo.png", width=120) if os.path.exists('Logo.png') else st.title("Rekomendasi Buku")
    
    if HAS_OPTION_MENU:
        selected_page = option_menu(
            menu_title="Menu Utama",
            options=["Beranda", "Rekomendasi", "Analisis Teks", "About", "Feedback"],
            icons=["house-door-fill", "star-fill", "search", "info-circle-fill", "chat-left-text-fill"],
            menu_icon="compass-fill",
            default_index=0
        )
    else:
        selected_page = st.radio("Menu Utama", ["Beranda", "Rekomendasi", "Analisis Teks", "About", "Feedback"])

    st.sidebar.markdown("---")
    st.sidebar.caption("Dibuat oleh Nanda | 2025")

# ----------------------
# 4. Konten Halaman
# ----------------------

# ===============================================
# Halaman 1: BERANDA
# (Kode ini tidak berubah)
# ===============================================
if selected_page == "Beranda":
    st.markdown("""
        <div style='text-align:center; padding: 20px;'>
            <h1>Sistem Rekomendasi Buku</h1>
            <p style='font-size:18px;'>Temukan buku favorit Anda berikutnya berdasarkan preferensi dan analisis konten.</p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    if not df_books.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Judul Buku", f"{df_books['title'].nunique()} Judul")
        
        try:
            if 'authors' in df_books.columns:
                all_authors = df_books['authors'].dropna().astype(str).unique()
                col2.metric("Total Penulis", f"{len(all_authors)} Penulis")
            else:
                col2.metric("Total Penulis", "N/A")
        except Exception:
            col2.metric("Total Penulis", "N/A")

        if 'categories' in df_books.columns:
            col3.metric("Jumlah Kategori", f"{df_books['categories'].nunique()} Kategori")
        else:
            col3.metric("Jumlah Kategori", "N/A")
    else:
        st.info("Data buku belum dimuat...")

    st.divider()
    st.subheader("Sampel Dataset Buku")
    if not df_books.empty:
        st.dataframe(df_books[['title', 'authors', 'categories']].head(10), use_container_width=True)
    st.divider()

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Top 10 Kategori Buku")
        if not df_books.empty and 'categories' in df_books.columns:
            try:
                top_categories = df_books['categories'].dropna().astype(str).value_counts().nlargest(10)
                st.bar_chart(top_categories)
            except Exception as e:
                st.warning(f"Gagal membuat chart kategori: {e}")
        else:
            st.caption("Kolom 'categories' tidak ditemukan.")

    with col_chart2:
        st.subheader("Top 10 Penulis")
        if not df_books.empty and 'authors' in df_books.columns:
            try:
                top_authors = df_books['authors'].dropna().astype(str).value_counts().nlargest(10)
                st.bar_chart(top_authors)
            except Exception as e:
                st.warning(f"Gagal membuat chart penulis: {e}")
        else:
            st.caption("Kolom 'authors' tidak ditemukan.")
            
    st.divider()
    st.subheader("☁️ Word Cloud (dari Deskripsi & Judul)")
    if not df_books.empty and HAS_WORDCLOUD:
        text_reviews = " ".join(df_books['text'].astype(str))
        if text_reviews.strip():
            wc = WordCloud(width=800, height=400, background_color="white").generate(text_reviews)
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
        else:
            st.caption("Tidak ada teks untuk WordCloud.")

# ===============================================
# Halaman 2: REKOMENDASI (DIUBAH)
# ===============================================
elif selected_page == "Rekomendasi":
    st.header("🔎 Rekomendasi Buku (Pencarian Judul)")
    st.markdown("Cari buku berdasarkan judul (toleran terhadap typo) dan dapatkan rekomendasi berdasarkan kemiripan makna (Embedding).")
    
    if df_books.empty:
        st.error("book.csv tidak tersedia. Upload dataset terlebih dahulu.")
    
    else:
        # --- Opsi Metode (HANYA KNN) ---
        method_options = []
        if HAS_SBERT and knn_model is not None and sbert_embeddings is not None:
            method_options.append("Embedding + KNN (Semantic)")
        
        if not method_options:
            st.error("Model KNN atau SBERT Embeddings tidak dimuat. Fitur rekomendasi tidak tersedia.")
        else:
            cols = st.columns([3,1])
            query = cols[0].text_input("Cari judul buku (typo OK):", value="harry pottr and the chamber of secrets")
            method = cols[1].selectbox("Metode Rekomendasi", method_options)
            top_k = st.slider("Jumlah Rekomendasi (Top K)", min_value=3, max_value=20, value=5)

            # Fuzzy match
            titles = df_books['title'].astype(str).tolist()
            matched_title, score = fuzzy_match(query, titles)
            if matched_title:
                st.caption(f"Judul terdekat ditemukan: **{matched_title}** (Skor kemiripan: {score})")

            if st.button("Dapatkan Rekomendasi"):
                if matched_title is None:
                    st.warning('Tidak ada judul yang cocok. Coba query pencarian lain.')
                else:
                    try:
                        idx_query = df_books[df_books['title'] == matched_title].index[0]
                        
                        # --- Logika SBERT / Embedding ---
                        if method == 'Embedding + KNN (Semantic)' and HAS_SBERT:
                            
                            # Dapatkan embedding dari query (berdasarkan index judul yang di-match)
                            q_emb = sbert_embeddings[idx_query].reshape(1,-1)
                            
                            # Gunakan knn.pkl yang sudah di-load
                            dist, idxs = knn_model.kneighbors(q_emb, n_neighbors=top_k+1)
                            
                            idxs = idxs.flatten()[1:top_k+1] # Skip item pertama (dirinya sendiri)
                            
                            res = df_books.iloc[idxs][['title','authors','categories']].copy()
                            res['distance'] = dist.flatten()[1:top_k+1]
                            st.subheader("Hasil Rekomendasi (Semantic):")
                            st.dataframe(res.reset_index(drop=True), use_container_width=True)
                            
                    except IndexError:
                        st.error("Gagal menemukan index untuk judul yang cocok. Coba refresh.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {e}")

# ===============================================
# Halaman 3: ANALISIS TEKS (DIUBAH)
# ===============================================
elif selected_page == "Analisis Teks":
    st.markdown("""
        <div style='text-align:center; padding: 20px;'>
            <h1>Analisis Topik Teks</h1>
            <p style='font-size:18px;'>Jelaskan buku yang Anda cari, dan kami akan analisis topiknya!</p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Definisikan nama cluster Anda (HARUS SESUAI DENGAN HASIL TRAINING K-MEANS ANDA)
    nama_cluster_buku = {
        0: "Fiksi & Sastra",
        1: "Teknis & Sains",
        2: "Self-Help & Bisnis",
        3: "Biografi & Sejarah"
        # ... sesuaikan dengan jumlah K (kluster) Anda
    }

    if tfidf_vectorizer is None or kmeans_model is None:
        st.error("⚠️ Model (tfidf.pkl/kmean.pkl) gagal dimuat. Fitur ini tidak tersedia.")
    else:
        user_input = st.text_area(
            "Jelaskan buku yang Anda inginkan:", 
            "Saya mencari buku tentang petualangan di luar angkasa, sihir, dan ada naganya", 
            key="input_analisis_teks",
            height=150
        )

        if st.button("Analisis Topik Teks"):
            if user_input.strip():
                try:
                    # 1. Ubah input pengguna menjadi vektor TF-IDF
                    input_vector = tfidf_vectorizer.transform([user_input])

                    # 2. Prediksi Topik (K-Means)
                    cluster_pred = kmeans_model.predict(input_vector)[0]
                    cluster_name = nama_cluster_buku.get(cluster_pred, f"Cluster {cluster_pred}")

                    st.subheader("Hasil Analisis Teks Anda")
                    st.info(f"Topik utama yang Anda cari terdeteksi sebagai: **{cluster_name}**")
                    
                    # === Bagian Cosine Similarity DIHAPUS karena tfidf_matrix.pkl tidak ada ===

                except Exception as e:
                    st.error(f"❌ Gagal memproses: {e}")
            else:
                st.warning("Masukkan deskripsi buku yang Anda cari terlebih dahulu!")

# ===============================================
# Halaman 4: ABOUT (DIUBAH)
# ===============================================
elif selected_page == "About":
    st.markdown("""
        <div style='text-align:center; padding: 20px;'>
            <h1>About Aplikasi Ini</h1>
            <p style='font-size:18px;'>Metodologi dan teknologi di balik portofolio ini.</p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    st.markdown("""
    ### Metodologi
    Aplikasi ini adalah portofolio yang mendemonstrasikan beberapa teknik *Machine Learning* untuk sistem rekomendasi:
    
    1.  **Pencarian Judul (Tab Rekomendasi)**
        * **Fuzzy Search:** Menggunakan `thefuzz` untuk menemukan judul buku yang paling mirip dengan input pengguna.
        * **Rekomendasi Semantic:** Menggunakan model `knn.pkl` (Nearest Neighbors) yang telah dilatih pada data `sbert_embeddings.pkl` untuk menemukan item dengan "makna" terdekat.
        * **(Fitur TF-IDF Cosine dinonaktifkan)**

    2.  **Analisis Teks (Tab Analisis Teks)**
        * **Clustering Topik:** Model `kmean.pkl` (K-Means) dan `tfidf.pkl` (Vectorizer) digunakan untuk mengelompokkan input teks pengguna ke dalam topik utama.
        * **(Fitur Rekomendasi Cosine dinonaktifkan)**

    3.  **Deployment**
        * Seluruh model (`tfidf.pkl`, `kmean.pkl`, `knn.pkl`) dan data matriks (`sbert_embeddings.pkl`) dimuat ke memori menggunakan `@st.cache_resource`.
    
    4.  **Feedback Pengguna**
        * Menggunakan `gspread` dan `st.secrets` untuk terhubung ke Google Sheets API.
    
    ---
    *Dibuat oleh Nanda*
    """)

# ===============================================
# Halaman 5: FEEDBACK
# (Kode ini tidak berubah)
# ===============================================
elif selected_page == "Feedback":

    st.markdown("""
        <div style='text-align:center; padding: 15px;'>
            <h2>Formulir Feedback Pengguna</h2>
            <p style='font-size:17px;'>Masukan Anda sangat berharga bagi pengembangan aplikasi ini 🙌</p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    SCOPE = [
        "https.www.googleapis.com/auth/spreadsheets",
        "https.www.googleapis.com/auth/drive"
    ]
    sheet = None
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, SCOPE)
            client = gspread.authorize(creds)
            sheet = client.open("feedback_portofolio_buku").sheet1 # Ganti nama Google Sheet Anda
        else:
            st.warning("Fitur feedback dinonaktifkan. Secret 'gcp_service_account' tidak diatur.")
            
    except Exception as e:
        st.error(f"⚠️ Tidak dapat terhubung ke Google Sheets: {e}")
        st.info("Pastikan Anda telah membagikan Google Sheet Anda dengan email 'client_email' di file secrets.")

    with st.form("feedback_form"):
        user_name = st.text_input("Nama (opsional)")
        user_rating = st.slider("Seberapa puas Anda dengan aplikasi ini?", 1, 5, 5)
        user_feedback = st.text_area("Kritik / Saran Anda ✍️", placeholder="Aplikasi ini sangat membantu, tapi...")
        submitted = st.form_submit_button("Kirim Feedback ✅")

    if submitted:
        if not user_feedback.strip():
            st.warning("Mohon isi kritik atau saran terlebih dahulu.")
        elif sheet is None:
            st.error("❌ Feedback gagal dikirim. Koneksi ke database (Google Sheets) belum siap.")
        else:
            try:
                sheet.append_row([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    user_name,
                    user_rating,
                    user_feedback
                ])
                st.success("✨ Terima kasih! Feedback Anda berhasil dikirim.")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Gagal menyimpan feedback ke Google Sheets: {e}")
