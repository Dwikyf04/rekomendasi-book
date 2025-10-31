# app.py
# Streamlit Book Recommender — Nanda
# Features: simple login, TF-IDF, Embedding+KNN (optional), Hybrid, KMeans clustering
# UI styled to look similar to repository.unair.ac.id (header blue + navbar yellow)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from textwrap import shorten

# Optional: sentence-transformers (embedding)
HAS_SBERT = False
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SBERT = True
except Exception:
    HAS_SBERT = False

# -------------------------
# Config & constants
# -------------------------
st.set_page_config(page_title="UNAIR-like Book Recommender — Nanda",
                   page_icon="📚", layout="wide")

APP_TITLE = "📚 Book Recommender — Nanda"
TFIDF_PATH = "tfidf_vectorizer.pkl"
KNN_PATH = "knn_model.pkl"
KMEANS_PATH = "kmeans_model.pkl"
DF_PATH = "df_books.pkl"   # your df pickle (can fallback to CSV upload)
EMB_PATH = "embeddings.pkl"  # optional precomputed embeddings (numpy array or list)

# Simple user store (in-code). Replace with DB for production.
USERS = {"nanda": "admin123", "guest": "guest"}

# -------------------------
# Helper functions
# -------------------------
def safe_load_joblib(path):
    try:
        obj = joblib.load(path)
        st.sidebar.success(f"Loaded: {os.path.basename(path)}")
        return obj
    except FileNotFoundError:
        st.sidebar.warning(f"Not found: {os.path.basename(path)}")
        return None
    except Exception as e:
        st.sidebar.error(f"Failed to load {os.path.basename(path)}: {e}")
        return None

@st.cache_data
def load_df_pickle(path):
    try:
        df = pd.read_pickle(path)
        return df
    except Exception as e:
        return None

def ensure_text_column(df):
    # Create combined text column for TF-IDF/embedding if not present
    if 'text' not in df.columns:
        cols = [c for c in ['title','subtitle','authors','categories','description'] if c in df.columns]
        if cols:
            df['text'] = df[cols].fillna('').agg(' '.join, axis=1)
        else:
            df['text'] = df['title'].astype(str)
    return df

def fuzzy_match_title(query, titles):
    # basic fuzzy match: prefer thefuzz if installed, else difflib
    try:
        from thefuzz import process
        match = process.extractOne(query, titles)
        if match:
            return match[0], match[1]
    except Exception:
        import difflib
        matches = difflib.get_close_matches(query, titles, n=1)
        if matches:
            return matches[0], 90
    return None, 0

def render_book_card(row, score=None):
    title = row.get('title', '')
    authors = row.get('authors', '')
    categories = row.get('categories', '')
    desc = row.get('description', '') if 'description' in row else row.get('text','')
    desc_short = shorten(str(desc), width=180, placeholder="...")
    score_display = f"{score:.3f}" if score is not None else ""
    st.markdown(f"**{title}**  \n_{authors}_  \n**Category:** {categories}  \n{desc_short}  \n**Score:** {score_display}")
    st.markdown("---")

# -------------------------
# Load models (try to load .pkl artifacts)
# -------------------------
with st.sidebar:
    st.markdown("## ⚙️ Models & Data (sidebar)")
tfidf = safe_load_joblib(TFIDF_PATH)
knn_model = safe_load_joblib(KNN_PATH)
kmeans_model = safe_load_joblib(KMEANS_PATH)
embeddings = safe_load_joblib(EMB_PATH)  # optional
df = None
# load df pickle:
df = load_df_pickle(DF_PATH)
if df is None:
    # try CSV fallback if exists
    if os.path.exists("books.csv"):
        try:
            df = pd.read_csv("books.csv")
            st.sidebar.success("Loaded books.csv")
        except Exception as e:
            st.sidebar.error(f"Failed to load books.csv: {e}")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

if not df.empty:
    df = ensure_text_column(df)
    # ensure lowercase title for matching convenience
    df['title_norm'] = df['title'].astype(str).str.strip()
else:
    st.sidebar.info("No dataset loaded. Use Upload Data tab or place df_books.pkl in repo.")

# If embeddings not provided but SBERT installed, we'll offer to compute lazily
sbert_model = None
if HAS_SBERT and embeddings is None and not df.empty:
    # don't compute automatically; compute when user requests due to cost/time
    pass

# -------------------------
# CSS styling to emulate UNAIR-like look
# -------------------------
st.markdown("""
<style>
.header {
  background-color: #002855;
  color: white;
  padding: 18px;
  border-radius: 6px;
  text-align: center;
}
.navbar {
  background-color: #ffd100;
  padding: 10px;
  border-radius: 6px;
  display:flex;
  gap:12px;
  justify-content: center;
}
.searchbox {
  margin-top: 20px;
  text-align: center;
}
.search-input {
  width: 60%;
  padding: 12px;
  font-size: 18px;
  border-radius: 10px;
  border: 1px solid #ccc;
}
.card {
  border: 1px solid #eee;
  padding: 12px;
  border-radius: 8px;
  background-color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"<div class='header'><h2>{APP_TITLE}</h2></div>", unsafe_allow_html=True)
st.markdown("<div class='navbar'> <b>🏠 Home</b> &nbsp;&nbsp; <b>📖 Recommender</b> &nbsp;&nbsp; <b>📊 Clusters</b> &nbsp;&nbsp; <b>🔧 Upload</b> &nbsp;&nbsp; <b>ℹ️ About</b></div>", unsafe_allow_html=True)
st.write("")

# -------------------------
# Simple Login (stateless)
# -------------------------
st.sidebar.markdown("### 🔐 Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_btn = st.sidebar.button("Login")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if login_btn:
    if username in USERS and USERS[username] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.sidebar.success(f"Logged in as {username}")
    else:
        st.sidebar.error("Invalid username or password")

# Logout
if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.experimental_rerun()

# -------------------------
# Main layout (tabs)
# -------------------------
tab = st.selectbox("", ["Home", "Recommender", "Clusters", "Upload Data", "About"])

# ---- HOME ----
if tab == "Home":
    st.subheader("Welcome to the Book Recommender Portfolio")
    st.write("This demo app uses TF-IDF, (optional) sentence embeddings, KNN and KMeans to recommend books.")
    if df.empty:
        st.info("No dataset loaded. Please go to 'Upload Data' or add df_books.pkl to the repository.")
    else:
        st.write(f"Dataset contains **{len(df)}** books.")
        st.dataframe(df[['title','authors','categories']].head(8))
        # wordcloud sample
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            text = ' '.join(df['text'].astype(str).tolist())
            if len(text.strip())>20:
                wc = WordCloud(width=800, height=300, background_color='white').generate(text)
                fig, ax = plt.subplots(figsize=(10,3))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        except Exception:
            pass

# ---- RECOMMENDER ----
elif tab == "Recommender":
    st.header("🔎 Recommender")
    if not st.session_state.logged_in:
        st.warning("Please login in the sidebar to get personalized recommendations.")
    query_col, method_col, k_col = st.columns([6,3,1])
    query = query_col.text_input("Search book title (typo OK):", value="")
    method = method_col.selectbox("Method", ["TF-IDF Cosine", "Embedding + KNN (if available)", "Hybrid (TF-IDF+Embedding)"])
    top_k = k_col.slider("Top K", 3, 20, 6)

    # fuzzy suggest
    if df is not None and not df.empty and query.strip():
        matched_title, score = fuzzy_match_title(query, df['title_norm'].tolist())
        if matched_title:
            st.caption(f"Did you mean: **{matched_title}**  (score: {score})")

    if st.button("Recommend"):
        if query.strip() == "":
            st.warning("Please enter a search query.")
        elif df.empty:
            st.error("Dataset not loaded.")
        else:
            if method == "TF-IDF Cosine":
                if tfidf is None:
                    st.error("TF-IDF vectorizer not found.")
                else:
                    # create vector for all texts if vectorizer supports .transform on df['text']
                    try:
                        if 'text' in df.columns:
                            corpus = df['text'].fillna('').tolist()
                            X = tfidf.transform(corpus)
                        else:
                            X = tfidf.transform(df['title'].astype(str).tolist())
                        # try to match title first, else use query as pseudo-document
                        if matched_title:
                            idx = df[df['title_norm'] == matched_title].index[0]
                            vec = X[idx]
                        else:
                            vec = tfidf.transform([query])
                            # we'll compute similarity with X
                        sims = cosine_similarity(vec, X).flatten()
                        top_idxs = sims.argsort()[::-1][1:top_k+1] if matched_title else sims.argsort()[::-1][:top_k]
                        st.subheader("Recommendations (TF-IDF)")
                        for i in top_idxs:
                            render_book_card(df.iloc[i], sims[i])
                    except Exception as e:
                        st.error(f"Failed TF-IDF recommendation: {e}")

            elif method == "Embedding + KNN (if available)":
                if embeddings is None and not HAS_SBERT:
                    st.error("No embeddings available and sentence-transformers not installed.")
                else:
                    try:
                        # load or compute embeddings
                        if embeddings is None:
                            # compute on the fly (may be slow)
                            s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                            emb = s_model.encode(df['text'].tolist(), convert_to_tensor=False)
                        else:
                            emb = embeddings  # assumed numpy array / list
                        # build KNN
                        knn_local = NearestNeighbors(metric='cosine', algorithm='brute')
                        knn_local.fit(emb)
                        # encode query (use s_model if needed)
                        if HAS_SBERT:
                            s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                            qv = s_model.encode(query, convert_to_tensor=False).reshape(1,-1)
                        else:
                            # fallback: use TF-IDF vectorization to seed (less ideal)
                            qv = tfidf.transform([query]).toarray()
                        dists, idxs = knn_local.kneighbors(qv, n_neighbors=top_k+1)
                        idxs = idxs.flatten()[1:top_k+1]
                        st.subheader("Recommendations (Embedding + KNN)")
                        for i, idx in enumerate(idxs):
                            render_book_card(df.iloc[idx], score=1.0 - float(dists.flatten()[i]))
                    except Exception as e:
                        st.error(f"Embedding KNN failed: {e}")

            else:  # Hybrid
                # combine TF-IDF and embedding scores with alpha weight
                alpha = st.slider("Embedding weight (alpha)", 0.0, 1.0, 0.5)
                try:
                    # TF-IDF sim
                    if 'text' in df.columns:
                        corpus = df['text'].fillna('').tolist()
                        X = tfidf.transform(corpus)
                    else:
                        X = tfidf.transform(df['title'].astype(str).tolist())
                    vec = tfidf.transform([query])
                    tfidf_sim = cosine_similarity(vec, X).flatten()
                    # Embedding sim
                    if embeddings is None:
                        if not HAS_SBERT:
                            st.error("No embeddings and sentence-transformers not available.")
                            emb_sim = np.zeros_like(tfidf_sim)
                        else:
                            s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                            emb = s_model.encode(df['text'].tolist(), convert_to_tensor=False)
                            qv = s_model.encode(query, convert_to_tensor=False).reshape(1,-1)
                            emb_sim = cosine_similarity(qv, emb).flatten()
                    else:
                        emb = embeddings
                        # compute qv using same model (if saved) - here we assume SBERT available
                        if HAS_SBERT:
                            s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                            qv = s_model.encode(query, convert_to_tensor=False).reshape(1,-1)
                            emb_sim = cosine_similarity(qv, emb).flatten()
                        else:
                            emb_sim = np.zeros_like(tfidf_sim)
                    # normalize both
                    tfidf_norm = (tfidf_sim - tfidf_sim.min()) / (tfidf_sim.max() - tfidf_sim.min() + 1e-9)
                    emb_norm = (emb_sim - emb_sim.min()) / (emb_sim.max() - emb_sim.min() + 1e-9)
                    combined = alpha * emb_norm + (1-alpha) * tfidf_norm
                    top_idxs = combined.argsort()[::-1][:top_k]
                    st.subheader("Recommendations (Hybrid)")
                    for i in top_idxs:
                        render_book_card(df.iloc[i], combined[i])
                except Exception as e:
                    st.error(f"Hybrid recommendation failed: {e}")

# ---- CLUSTERS ----
elif tab == "Clusters":
    st.header("📂 K-Means Clustering")
    if df.empty:
        st.warning("Dataset not loaded.")
    else:
        default_k = 6
        k = st.slider("Choose number of clusters", 2, 20, default_k)
        if st.button("Build / Rebuild KMeans"):
            try:
                # Use TF-IDF matrix if available
                if tfidf is None:
                    st.error("TF-IDF vectorizer not available to build clusters.")
                else:
                    if 'text' in df.columns:
                        corpus = df['text'].fillna('').tolist()
                        X = tfidf.transform(corpus)
                    else:
                        X = tfidf.transform(df['title'].astype(str).tolist())
                    km = KMeans(n_clusters=k, random_state=42)
                    labels = km.fit_predict(X)
                    df['cluster'] = labels
                    # persist model
                    os.makedirs('models', exist_ok=True)
                    joblib.dump(km, 'models/kmeans_model.pkl')
                    st.success("KMeans built and saved to models/kmeans_model.pkl")
            except Exception as e:
                st.error(f"KMeans build failed: {e}")
        if 'cluster' in df.columns:
            st.write(df.groupby('cluster').size().reset_index(name='count'))
            chosen = st.selectbox("Choose cluster to inspect", sorted(df['cluster'].unique()))
            st.dataframe(df[df['cluster']==chosen][['title','authors','categories']].head(50))

# ---- UPLOAD DATA ----
elif tab == "Upload Data":
    st.header("📁 Upload dataset (books.csv or df_books.pkl)")
    uploaded = st.file_uploader("Upload CSV or Pickle (books_df.pkl / df_books.pkl)", type=['csv','pkl','pkl'])
    if uploaded is not None:
        try:
            # if csv
            if uploaded.name.lower().endswith('.csv'):
                new_df = pd.read_csv(uploaded)
                new_df.to_pickle('df_books.pkl')
                st.success("CSV uploaded and saved as df_books.pkl in repo root (persisted in container). Reload page.")
            else:
                # assume pickle
                content = uploaded.read()
                with open('df_books.pkl','wb') as f:
                    f.write(content)
                st.success("Pickle saved as df_books.pkl in repo root. Reload page.")
        except Exception as e:
            st.error(f"Upload failed: {e}")

# ---- ABOUT ----
elif tab == "About":
    st.header("About this Portfolio App")
    st.write("""
    - Built by **Nanda** — Book Recommender portfolio combining TF-IDF, optional SBERT embeddings, KNN, and KMeans.
    - Login is a simple demo (replace with secure auth for production).
    - Place model files (tfidf_vectorizer.pkl, knn_model.pkl, kmeans_model.pkl, df_books.pkl, embeddings.pkl) in repo root or models/ folder.
    - To deploy: push to GitHub and connect to Streamlit Cloud (share.streamlit.io).
    """)
    st.markdown("**Model files currently detected in repo:**")
    present = {TFIDF_PATH: os.path.exists(TFIDF_PATH), KNN_PATH: os.path.exists(KNN_PATH),
               KMEANS_PATH: os.path.exists(KMEANS_PATH), DF_PATH: os.path.exists(DF_PATH),
               EMB_PATH: os.path.exists(EMB_PATH)}
    st.json(present)

# footer
st.markdown("---")
st.caption("© Nanda — Book Recommender Portfolio. For production, secure your credentials and hosts.")
