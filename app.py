
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re 
import difflib
from thefuzz import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer, util
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Optional: semantic embeddings (SentenceTransformers). If not available the app still runs with TF-IDF.
try:
    HAS_SBERT = True
except Exception:
    HAS_SBERT = False

# fuzzy matching
try:
except Exception:
    process = None

# ----------------------
# App configuration
# ----------------------
st.set_page_config(page_title="Book Recommender — Nanda", layout="wide")
st.title("📚 Book Recommendation Portfolio")
st.markdown("A compact portfolio app: search books (with typo-tolerance), get content-based and embedding-based recommendations, explore clusters, and inspect metadata.")

# ----------------------
# Utilities
# ----------------------
@st.cache_data
def load_books(csv_path="books.csv"):
    if not os.path.exists(csv_path):
        st.error(f"books.csv not found at path: {csv_path}. Please upload dataset to repository root or use File Uploader in the Recommender tab.")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # minimal cleaning
    df = df.drop_duplicates(subset=[c for c in ['isbn13','title'] if c in df.columns], keep='first')
    df['title'] = df['title'].astype(str)
    # create combined text field
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

@st.cache_resource
def build_tfidf(df, max_features=5000):
    vec = TfidfVectorizer(stop_words='english', max_features=max_features)
    mat = vec.fit_transform(df['text'].fillna(''))
    return vec, mat

@st.cache_resource
def build_kmeans(tfidf_matrix, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)
    return kmeans

@st.cache_resource
def build_sbert_embeddings(df, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df['text'].tolist(), convert_to_tensor=True)
    return model, embeddings

# fuzzy helper
def fuzzy_match(query, choices, limit=1):
    if process is not None:
        res = process.extractOne(query, choices)
        if res:
            return res[0], res[1]
        return None, 0
    else:
        # use difflib sequence matcher
        matches = difflib.get_close_matches(query, choices, n=limit)
        return (matches[0], 90) if matches else (None, 0)

# ----------------------
# Load data & models
# ----------------------
df = load_books('books.csv')
if not df.empty:
    tfidf_vectorizer, tfidf_matrix = build_tfidf(df)
else:
    tfidf_vectorizer, tfidf_matrix = None, None

# Try to load optional precomputed artifacts (model files) if present
KMEANS_PATH = 'model/kmeans.pkl'
SBERT_PATH = 'model/book_embeddings.pkl'  # optional

kmeans_model = None
if os.path.exists(KMEANS_PATH):
    try:
        kmeans_model = joblib.load(KMEANS_PATH)
    except Exception:
        kmeans_model = None

sbert_model = None
book_embeddings = None
if HAS_SBERT:
    if os.path.exists(SBERT_PATH):
        try:
            book_embeddings = joblib.load(SBERT_PATH)
            # note: we do not persist the SentenceTransformer object, only embeddings
        except Exception:
            book_embeddings = None
    else:
        # Build embeddings lazily if user asks
        book_embeddings = None

# ----------------------
# Sidebar menu
# ----------------------
with st.sidebar:
    st.image("Logo.png", width=120) if os.path.exists('Logo.png') else st.write("")
    page = st.radio("Navigation", ["Home","Recommender","Clusters","Upload Data","About"])
    st.markdown("---")
    st.caption("Portofolio — Nanda | Book Recommender")

# ----------------------
# Home
# ----------------------
if page == 'Home':
    st.header("Welcome")
    st.write("This app showcases a book recommender system built for a portfolio. Use the Recommender tab to search and get recommendations.")

    if df.empty:
        st.warning("Dataset not found. Go to Upload Data tab to upload your books.csv or push dataset to repo.")
    else:
        st.subheader("Dataset sample")
        st.dataframe(df[['title','authors','categories']].head(10))
        st.subheader("Word Cloud (from titles & descriptions)")
        wc_text = ' '.join(df['text'].astype(str).tolist())
        if wc_text.strip():
            wc = WordCloud(width=800, height=400).generate(wc_text)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

# ----------------------
# Recommender
# ----------------------
elif page == 'Recommender':
    st.header("🔎 Book Recommender")
    if df.empty:
        st.error("books.csv not available. Upload dataset first.")
    else:
        cols = st.columns([3,1])
        query = cols[0].text_input("Search book title (typo OK):", value="harry pottr and the chamber of secrets")
        method = cols[1].selectbox("Method", ["TF-IDF Cosine","Embedding + KNN (semantic)","Hybrid (TF-IDF+Embedding)"])
        top_k = st.slider("Top K", min_value=3, max_value=20, value=5)

        # fuzzy match to find canonical title
        titles = df['title'].astype(str).tolist()
        matched_title, score = fuzzy_match(query, titles)
        if matched_title:
            st.caption(f"Did you mean: **{matched_title}**  (score: {score})")

        if st.button("Recommend"):
            if method == 'TF-IDF Cosine':
                # find index
                if matched_title is None:
                    st.warning('No close title matched. Try different input.')
                else:
                    idx = df[df['title'] == matched_title].index[0]
                    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
                    top_idx = sim.argsort()[::-1][1:top_k+1]
                    res = df.iloc[top_idx][['title','authors','categories']].copy()
                    res['score'] = sim[top_idx]
                    st.dataframe(res.reset_index(drop=True))

            elif method == 'Embedding + KNN (semantic)':
                if not HAS_SBERT:
                    st.error('SentenceTransformer not installed. Install sentence-transformers in requirements to use embedding mode.')
                else:
                    # build embeddings if not present
                    if book_embeddings is None:
                        with st.spinner('Building sentence embeddings (may take a while)...'):
                            s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                            book_embeddings = s_model.encode(df['text'].tolist(), convert_to_tensor=False)
                            # save to memory cache (not persisted to disk here)
                    # build knn
                    knn = NearestNeighbors(metric='cosine', algorithm='brute')
                    knn.fit(book_embeddings)
                    # encode query
                    s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                    q_emb = s_model.encode(query, convert_to_tensor=False).reshape(1,-1)
                    dist, idxs = knn.kneighbors(q_emb, n_neighbors=top_k+1)
                    idxs = idxs.flatten()[1:top_k+1]
                    res = df.iloc[idxs][['title','authors','categories']].copy()
                    res['distance'] = dist.flatten()[1:top_k+1]
                    st.dataframe(res.reset_index(drop=True))

            else: # Hybrid
                # compute tfidf scores and embedding scores then combine
                if not HAS_SBERT:
                    st.error('SentenceTransformer not installed. Install sentence-transformers in requirements to use hybrid mode.')
                else:
                    # TF-IDF part
                    if matched_title is None:
                        st.warning('No close title matched. Try different input.')
                    else:
                        idx = df[df['title'] == matched_title].index[0]
                        tfidf_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
                        # Embedding part
                        s_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                        emb = s_model.encode(df['text'].tolist(), convert_to_tensor=False)
                        q_emb = s_model.encode(query, convert_to_tensor=False).reshape(1,-1)
                        emb_sim = cosine_similarity(q_emb, emb).flatten()
                        alpha = st.slider('Alpha (embedding weight)', 0.0, 1.0, 0.5)
                        combined = alpha * emb_sim + (1-alpha) * tfidf_sim
                        top_idx = combined.argsort()[::-1][1:top_k+1]
                        res = df.iloc[top_idx][['title','authors','categories']].copy()
                        res['score'] = combined[top_idx]
                        st.dataframe(res.reset_index(drop=True))

# ----------------------
# Clusters
# ----------------------
elif page == 'Clusters':
    st.header('📂 Clustering (K-Means)')
    if df.empty:
        st.error('books.csv not available. Upload dataset first.')
    else:
        n_clusters = st.slider('Number of clusters (k)', 2, 20, value=6)
        if st.button('Build / Rebuild KMeans'):
            with st.spinner('Fitting KMeans...'):
                kmeans_model = build_kmeans(tfidf_matrix, n_clusters=n_clusters)
                df['cluster'] = kmeans_model.labels_
                # persist model for reuse
                os.makedirs('models', exist_ok=True)
                joblib.dump(kmeans_model, 'models/kmeans.pkl')
                st.success('KMeans built and saved to models/kmeans.pkl')
        if 'cluster' in df.columns:
            st.write(df.groupby('cluster')['title'].count().reset_index(name='count'))
            chosen = st.selectbox('Select cluster to inspect', sorted(df['cluster'].unique()))
            st.dataframe(df[df['cluster']==chosen][['title','authors','categories']].head(50))

# ----------------------
# Upload Data
# ----------------------
elif page == 'Upload Data':
    st.header('📁 Upload books.csv')
    uploaded = st.file_uploader('Upload your books.csv (columns: title, authors, categories, description, etc.)', type=['csv'])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            uploaded_df.to_csv('books.csv', index=False)
            st.success('books.csv saved to app root. Reload page to use dataset.')
        except Exception as e:
            st.error(f'Failed to read uploaded file: {e}')

# ----------------------
# About
# ----------------------
elif page == 'About':
    st.header('About this Portfolio App')
    st.markdown('''
    - **Author:** Nanda
    - **What:** Book Recommendation System combining TF-IDF, semantic embeddings, clustering, and fuzzy search.
    - **How to run locally:**
        1. Put `books.csv` in the project root
        2. Install requirements (see requirements.txt)
        3. Run `streamlit run streamlit_book_recommender_app.py`
    ''')

# ----------------------
# Footer: requirements reminder
# ----------------------
st.markdown('---')
st.caption('Requirements example: pandas, scikit-learn, streamlit, thefuzz, wordcloud, sentence-transformers (optional).')

