# app.py
# Book Recommender Portfolio — Nanda
# Uses joblib to load models from models/ folder
# Login & Register saved in users.json
# Recommendation methods: TF-IDF, Embedding+KNN, Hybrid, KMeans

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os, json, hashlib, time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from textwrap import shorten

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
st.set_page_config(page_title="Book Recommender — Nanda", page_icon="📚", layout="wide")
ROOT = os.getcwd()
MODELS_DIR = os.path.join(ROOT, "models")
USERS_FILE = os.path.join(ROOT, "users.json")
BOOKS_CSV = os.path.join(ROOT, "data - books.csv")

# ---------------------------
# Utilities: users.json
# ---------------------------
def ensure_users_file():
    if not os.path.exists(USERS_FILE):
        init = {"users": {}}
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(init, f, indent=2)

def load_users():
    ensure_users_file()
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(data):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def hash_password(password: str):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def register_user(username, password):
    data = load_users()
    if username in data["users"]:
        return False, "Username sudah terdaftar."
    data["users"][username] = {"password": hash_password(password), "history": []}
    save_users(data)
    return True, "Registrasi berhasil."

def authenticate_user(username, password):
    data = load_users()
    pwd_hash = hash_password(password)
    user = data["users"].get(username)
    if user and user.get("password") == pwd_hash:
        return True
    return False

def add_history(username, query, method):
    data = load_users()
    if username not in data["users"]:
        return
    entry = {"time": int(time.time()), "query": query, "method": method}
    data["users"][username].setdefault("history", []).insert(0, entry)
    # limit history length
    data["users"][username]["history"] = data["users"][username]["history"][:200]
    save_users(data)

# ---------------------------
# Helpers for models & data
# ---------------------------
@st.cache_resource
def load_models():
    # returns (tfidf, kmeans, knn, embeddings)
    def safe_load(path):
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                st.sidebar.error(f"Failed to load {os.path.basename(path)}: {e}")
                return None
        else:
            st.sidebar.warning(f"Missing: {os.path.basename(path)}")
            return None

    tfidf = safe_load(os.path.join(MODELS_DIR, "model/tfidf_vectorizer.pkl"))
    kmeans = safe_load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    knn = safe_load(os.path.join(MODELS_DIR, "knn_model.pkl"))
    embeddings = safe_load(os.path.join(MODELS_DIR, "embeddings.pkl"))
    return tfidf, kmeans, knn, embeddings

def ensure_text_column(df):
    if 'text' not in df.columns:
        cols = [c for c in ['title','subtitle','authors','categories','description'] if c in df.columns]
        if cols:
            df['text'] = df[cols].fillna('').agg(' '.join, axis=1)
        else:
            df['text'] = df['title'].astype(str)
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
# Load models & data
# ---------------------------
tfidf, kmeans_model, knn_model, embeddings = load_models()

# Load books.csv if exists
books_df = pd.DataFrame()
if os.path.exists(BOOKS_CSV):
    try:
        books_df = pd.read_csv(BOOKS_CSV)
        books_df = ensure_text_column(books_df)
        books_df['title_norm'] = books_df['title'].astype(str).str.strip()
    except Exception as e:
        st.sidebar.error(f"Failed to read books.csv: {e}")
else:
    st.sidebar.warning("books.csv not found. Use Upload Data tab to upload dataset.")

# ---------------------------
# UI styling (UNAIR-like)
# ---------------------------
st.markdown("""
<style>
.header { background-color:#002855; color:white; padding:16px; border-radius:6px; text-align:center; }
.navbar { background-color:#ffd100; padding:8px; border-radius:6px; display:flex; gap:12px; justify-content:center; font-weight:600;}
.search { text-align:center; margin-top:18px; }
.search-input{ width:60%; padding:12px; font-size:18px; border-radius:10px; border:1px solid #ccc; }
.card { background:white; padding:12px; border-radius:10px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h2>UNIVERSITAS AIRLANGGA — Book Recommender (Nanda)</h2></div>', unsafe_allow_html=True)
st.markdown('<div class="navbar">🏠 Home &nbsp; 📖 Recommender &nbsp; 📊 Clusters &nbsp; ℹ️ About</div>', unsafe_allow_html=True)
st.write("")

# ---------------------------
# Sidebar: Login / Register
# ---------------------------
st.sidebar.title("Account")
ensure_users_file()
menu = st.sidebar.selectbox("Menu", ["Login","Register","Profile"])

if menu == "Register":
    st.sidebar.subheader("🔐 Create account")
    new_user = st.sidebar.text_input("Username", key="reg_user")
    new_pass = st.sidebar.text_input("Password", type="password", key="reg_pass")
    if st.sidebar.button("Register"):
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
            st.sidebar.success(f"Welcome, {in_user}")
        else:
            st.sidebar.error("Invalid username/password")

elif menu == "Profile":
    if 'logged_in' in st.session_state and st.session_state['logged_in']:
        u = st.session_state['username']
        st.sidebar.write(f"Logged in as **{u}**")
        # show history
        data = load_users()
        history = data["users"].get(u, {}).get("history", [])
        if history:
            st.sidebar.write("Recent queries:")
            for h in history[:8]:
                t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(h["time"]))
                st.sidebar.write(f"- {t} — {h['query']} ({h['method']})")
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()
    else:
        st.sidebar.info("You are not logged in. Use Login tab.")

# ---------------------------
# Main Tabs
# ---------------------------
tab = st.selectbox("", ["Home","Recommender","Clusters","Upload Data","About"])

# ------- Home -------
if tab == "Home":
    st.markdown('<div class="card"><h3>Welcome</h3><p>This portfolio app demonstrates TF-IDF and semantic recommendations for books. Login to get personalized recommendations.</p></div>', unsafe_allow_html=True)
    if not books_df.empty:
        st.write(f"Dataset: {len(books_df)} books")
        st.dataframe(books_df[['title','authors','categories']].head(8))
    else:
        st.info("No dataset loaded. Go to Upload Data to upload books.csv.")

# ------- Recommender -------
elif tab == "Recommender":
    st.header("🔎 Book Recommender")
    if 'logged_in' not in st.session_state or not st.session_state.get('logged_in'):
        st.warning("Please login to get personalized features. But you can still try recommendations (non-personal).")
    col1, col2 = st.columns([3,1])
    query = col1.text_input("Search book title (typo OK):", value="")
    method = col2.selectbox("Method", ["TF-IDF Cosine","Embedding + KNN","Hybrid (TFIDF+Embedding)"])
    top_k = st.slider("Top K", 3, 12, 6)

    # fuzzy suggestion
    if query and not books_df.empty:
        cand, score = fuzzy_match(query, books_df['title_norm'].tolist())
        if cand:
            st.caption(f"Did you mean: **{cand}** (score {score})")

    if st.button("Recommend"):
        if books_df.empty:
            st.error("Dataset not available.")
        else:
            # prepare TF-IDF matrix if tfidf available
            tfidf_matrix = None
            if tfidf is not None:
                try:
                    tfidf_matrix = tfidf.transform(books_df['text'].fillna('').tolist())
                except Exception:
                    try:
                        tfidf_matrix = tfidf.transform(books_df['title'].astype(str).tolist())
                    except Exception:
                        tfidf_matrix = None
            # TF-IDF Cosine
            if method == "TF-IDF Cosine":
                if tfidf is None or tfidf_matrix is None:
                    st.error("TF-IDF model not available.")
                else:
                    if cand:
                        idx = books_df[books_df['title_norm']==cand].index[0]
                        vec = tfidf_matrix[idx]
                    else:
                        vec = tfidf.transform([query])
                    sims = cosine_similarity(vec, tfidf_matrix).flatten()
                    top_idx = sims.argsort()[::-1][:top_k]
                    st.subheader("Recommendations (TF-IDF)")
                    for i in top_idx:
                        st.markdown(f"**{books_df.iloc[i]['title']}** — {books_df.iloc[i].get('authors','')}")
                        st.write(shorten(str(books_df.iloc[i].get('description',books_df.iloc[i].get('text',''))), width=180))
                        st.caption(f"Score: {sims[i]:.4f}")
                        st.markdown("---")
                    if 'username' in st.session_state:
                        add_history(st.session_state['username'], query, "TF-IDF")

            # Embedding + KNN
            elif method == "Embedding + KNN":
                if embeddings is None and not HAS_SBERT:
                    st.error("Embeddings not available and sentence-transformers not installed.")
                else:
                    try:
                        # get embeddings array (if dict or array)
                        if embeddings is None:
                            s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                            emb_array = s_model.encode(books_df['text'].tolist(), convert_to_tensor=False)
                            q_emb = s_model.encode(query, convert_to_tensor=False).reshape(1,-1)
                        else:
                            # embeddings might be dict mapping titles->vec OR numpy array aligned with DF
                            if isinstance(embeddings, dict):
                                titles = list(embeddings.keys())
                                emb_array = np.array(list(embeddings.values()))
                                # if query matches a title use that vector else encode using sbert if available
                                if query in embeddings:
                                    q_emb = np.array(embeddings[query]).reshape(1,-1)
                                elif HAS_SBERT:
                                    s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                                    q_emb = s_model.encode(query, convert_to_tensor=False).reshape(1,-1)
                                else:
                                    st.error("Cannot encode query: sentence-transformers not available.")
                                    q_emb = None
                            else:
                                emb_array = np.array(embeddings)
                                if HAS_SBERT:
                                    s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                                    q_emb = s_model.encode(query, convert_to_tensor=False).reshape(1,-1)
                                else:
                                    st.error("Embeddings provided as array but sentence-transformers not available to encode query.")
                                    q_emb = None

                        if q_emb is not None:
                            knn = NearestNeighbors(metric="cosine", algorithm="brute")
                            knn.fit(emb_array)
                            dists, idxs = knn.kneighbors(q_emb, n_neighbors=top_k)
                            st.subheader("Recommendations (Embedding + KNN)")
                            for dist, idx in zip(dists.flatten(), idxs.flatten()):
                                st.markdown(f"**{books_df.iloc[idx]['title']}** — {books_df.iloc[idx].get('authors','')}")
                                st.caption(f"Similarity: {1.0 - dist:.4f}")
                                st.write(shorten(str(books_df.iloc[idx].get('description',books_df.iloc[idx].get('text',''))), width=160))
                                st.markdown("---")
                            if 'username' in st.session_state:
                                add_history(st.session_state['username'], query, "EMBEDDING_KNN")
                    except Exception as e:
                        st.error(f"Embedding KNN error: {e}")

            # Hybrid
            elif method == "Hybrid (TFIDF+Embedding)":
                alpha = st.slider("Embedding weight (alpha)", 0.0, 1.0, 0.5)
                if tfidf is None or (embeddings is None and not HAS_SBERT):
                    st.error("Required models not available for Hybrid.")
                else:
                    # compute tfidf sim
                    try:
                        vec = tfidf.transform([query])
                        tfidf_sim = cosine_similarity(vec, tfidf_matrix).flatten()
                    except Exception:
                        tfidf_sim = np.zeros(len(books_df))
                    # compute emb sim
                    try:
                        if embeddings is None:
                            s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                            emb_array = s_model.encode(books_df['text'].tolist(), convert_to_tensor=False)
                            q_emb = s_model.encode(query, convert_to_tensor=False).reshape(1,-1)
                            emb_sim = cosine_similarity(q_emb, emb_array).flatten()
                        else:
                            if isinstance(embeddings, dict):
                                emb_array = np.array(list(embeddings.values()))
                                if query in embeddings:
                                    q_emb = np.array(embeddings[query]).reshape(1,-1)
                                elif HAS_SBERT:
                                    s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                                    q_emb = s_model.encode(query, convert_to_tensor=False).reshape(1,-1)
                                else:
                                    q_emb = None
                            else:
                                emb_array = np.array(embeddings)
                                if HAS_SBERT:
                                    s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                                    q_emb = s_model.encode(query, convert_to_tensor=False).reshape(1,-1)
                                else:
                                    q_emb = None
                            if q_emb is not None:
                                emb_sim = cosine_similarity(q_emb, emb_array).flatten()
                            else:
                                emb_sim = np.zeros(len(books_df))
                    except Exception:
                        emb_sim = np.zeros(len(books_df))

                    # normalize and combine
                    tf_norm = (tfidf_sim - tfidf_sim.min()) / (tfidf_sim.max() - tfidf_sim.min() + 1e-9)
                    eb_norm = (emb_sim - emb_sim.min()) / (emb_sim.max() - emb_sim.min() + 1e-9)
                    combined = alpha * eb_norm + (1 - alpha) * tf_norm
                    top_idx = combined.argsort()[::-1][:top_k]
                    st.subheader("Recommendations (Hybrid)")
                    for i in top_idx:
                        st.markdown(f"**{books_df.iloc[i]['title']}** — {books_df.iloc[i].get('authors','')}")
                        st.caption(f"Score: {combined[i]:.4f}")
                        st.write(shorten(str(books_df.iloc[i].get('description',books_df.iloc[i].get('text',''))), width=160))
                        st.markdown("---")
                    if 'username' in st.session_state:
                        add_history(st.session_state['username'], query, "HYBRID")

# ------- Clusters -------
elif tab == "Clusters":
    st.header("📂 K-Means Clustering")
    if books_df.empty:
        st.warning("Dataset not available.")
    else:
        k = st.slider("Number of clusters (k)", 2, 20, 6)
        if st.button("Build/Rebuild KMeans"):
            if tfidf is None:
                st.error("TF-IDF model required to build clusters.")
            else:
                try:
                    X = tfidf.transform(books_df['text'].fillna('').tolist())
                    km = KMeans(n_clusters=k, random_state=42)
                    labels = km.fit_predict(X)
                    books_df['cluster'] = labels
                    os.makedirs(MODELS_DIR, exist_ok=True)
                    joblib.dump(km, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
                    st.success("KMeans built and saved to models/kmeans_model.pkl")
                except Exception as e:
                    st.error(f"KMeans error: {e}")
        if 'cluster' in books_df.columns:
            st.write(books_df.groupby('cluster').size().reset_index(name='count'))
            sel = st.selectbox("Pick cluster", sorted(books_df['cluster'].unique()))
            st.dataframe(books_df[books_df['cluster']==sel][['title','authors','categories']].head(50))

# ------- Upload Data -------
elif tab == "Upload Data":
    st.header("📁 Upload dataset (books.csv)")
    uploaded = st.file_uploader("Upload CSV containing columns title, authors, categories, description (optional)", type=['csv'])
    if uploaded is not None:
        try:
            df_new = pd.read_csv(uploaded)
            df_new.to_csv(BOOKS_CSV, index=False)
            st.success("books.csv uploaded. Reload the app to pickup new dataset.")
        except Exception as e:
            st.error(f"Upload failed: {e}")

# ------- About -------
elif tab == "About":
    st.header("About")
    st.write("Portfolio app by Nanda. Shows TF-IDF, embeddings, KNN and KMeans based book recommendation.")
    st.write("Model files loaded from /models. If you want embeddings, precompute using SentenceTransformer and save as models/embeddings.pkl (aligned with books.csv rows).")
    st.write("For production: secure credentials, use remote DB, and do not commit private datasets to GitHub.")

# Footer
st.markdown("---")
st.caption("© Nanda — Book Recommender Portfolio. Use responsibly.")




