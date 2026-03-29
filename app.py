import streamlit as st
import pandas as pd
import numpy as np
import chromadb
import os
import time

st.set_page_config(
    page_title="Energy Semantic Search",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0a0e17;
    --surface: #111827;
    --border: #1e2d40;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --text: #e2e8f0;
    --muted: #64748b;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero p {
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

/* Result card */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    position: relative;
    transition: border-color 0.2s;
}
.result-card:hover { border-color: var(--accent); }

.rank-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--muted);
    background: #1e2d40;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 0.5rem;
}
.country-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
}
.year-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent);
    margin-left: 0.5rem;
}
.doc-text {
    font-size: 0.82rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.5rem;
    line-height: 1.6;
}

/* Similarity bar */
.sim-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-top: 0.7rem;
}
.sim-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    width: 70px;
    flex-shrink: 0;
}
.sim-bar-bg {
    flex: 1;
    height: 5px;
    background: #1e2d40;
    border-radius: 99px;
    overflow: hidden;
}
.sim-bar-fill {
    height: 100%;
    border-radius: 99px;
}
.sim-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    width: 36px;
    text-align: right;
    flex-shrink: 0;
}

/* Meta tags */
.meta-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.6rem;
}
.meta-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid;
}
.tag-fossil { color: var(--amber); border-color: rgba(245,158,11,0.3); background: rgba(245,158,11,0.06); }
.tag-renew  { color: var(--green); border-color: rgba(16,185,129,0.3); background: rgba(16,185,129,0.06); }
.tag-coal   { color: var(--red);   border-color: rgba(239,68,68,0.3);  background: rgba(239,68,68,0.06); }

/* Sidebar */
.sidebar-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.sidebar-title {
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}

/* Override Streamlit input */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.8rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.6rem !important;
    width: 100%;
}
.stSlider > div { padding-top: 0; }

/* No results */
.no-results {
    text-align: center;
    padding: 3rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}

/* Status */
.status-bar {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    text-align: right;
    margin-bottom: 0.5rem;
}

/* Hide Streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0; max-width: 1100px; }
</style>
""", unsafe_allow_html=True)


# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner=False)
def load_data_and_index():
    """Load CSV, build documents, embed, and store in Chroma (in-memory)."""
    df = pd.read_csv("owid-energy-data.csv")

    # Filter: real countries with core columns
    required = ['coal_share_energy', 'renewables_share_energy', 'fossil_share_energy']
    df = df[
        df['iso_code'].notna() &
        (df['iso_code'].str.len() == 3) &
        df[required].notna().all(axis=1)
    ].copy()

    def fossil_label(v):
        if v > 85: return 'very high fossil dependency'
        if v > 70: return 'high fossil dependency'
        if v > 50: return 'moderate fossil dependency'
        return 'low fossil dependency'

    def renew_label(v):
        if v > 40: return 'very high renewable share'
        if v > 20: return 'high renewable share'
        if v > 10: return 'growing renewable share'
        return 'low renewable share'

    def row_to_document(row):
        parts = [f"{row['country']} energy profile {int(row['year'])}"]
        if pd.notna(row.get('fossil_share_energy')):
            v = row['fossil_share_energy']
            parts.append(f"fossil fuels {v:.1f}% ({fossil_label(v)})")
        if pd.notna(row.get('coal_share_energy')) and row['coal_share_energy'] > 0.5:
            parts.append(f"coal {row['coal_share_energy']:.1f}%")
        if pd.notna(row.get('oil_share_energy')) and row['oil_share_energy'] > 0.5:
            parts.append(f"oil {row['oil_share_energy']:.1f}%")
        if pd.notna(row.get('gas_share_energy')) and row['gas_share_energy'] > 0.5:
            parts.append(f"gas {row['gas_share_energy']:.1f}%")
        if pd.notna(row.get('renewables_share_energy')):
            v = row['renewables_share_energy']
            parts.append(f"renewables {v:.1f}% ({renew_label(v)})")
        for src in ['solar', 'wind', 'nuclear', 'hydro']:
            col = f'{src}_share_energy'
            if pd.notna(row.get(col)) and row[col] > 0.1:
                parts.append(f"{src} {row[col]:.1f}%")
        if pd.notna(row.get('greenhouse_gas_emissions')):
            parts.append(f"greenhouse gas emissions {row['greenhouse_gas_emissions']:.1f} MtCO2")
        if pd.notna(row.get('primary_energy_consumption')):
            parts.append(f"total energy consumption {row['primary_energy_consumption']:.1f} TWh")
        if pd.notna(row.get('energy_per_capita')):
            parts.append(f"energy per capita {row['energy_per_capita']:.0f} kWh per person")
        return '. '.join(parts) + '.'

    documents, metadatas, ids = [], [], []
    for _, row in df.iterrows():
        doc = row_to_document(row)
        if len(doc) < 50:
            continue
        documents.append(doc)
        metadatas.append({
            'country':      str(row['country']),
            'year':         int(row['year']),
            'iso_code':     str(row['iso_code']),
            'fossil_share': round(float(row['fossil_share_energy']), 1) if pd.notna(row['fossil_share_energy']) else -1.0,
            'renew_share':  round(float(row['renewables_share_energy']), 1) if pd.notna(row['renewables_share_energy']) else -1.0,
            'solar_share':  round(float(row['solar_share_energy']), 2) if pd.notna(row.get('solar_share_energy')) else -1.0,
            'wind_share':   round(float(row['wind_share_energy']), 2) if pd.notna(row.get('wind_share_energy')) else -1.0,
            'coal_share':   round(float(row['coal_share_energy']), 1) if pd.notna(row['coal_share_energy']) else -1.0,
        })
        ids.append(f"{row['iso_code']}_{int(row['year'])}")

    model = load_model()
    embeddings = model.encode(documents, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    client = chromadb.Client()
    try:
        client.delete_collection("energy_profiles")
    except:
        pass
    collection = client.create_collection("energy_profiles", metadata={"hnsw:space": "cosine"})

    BATCH = 500
    for i in range(0, len(documents), BATCH):
        end = min(i + BATCH, len(documents))
        collection.add(
            documents=documents[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
            ids=ids[i:end]
        )

    return collection, documents, metadatas, df


def similarity_color(sim):
    if sim >= 0.75: return "#10b981"
    if sim >= 0.55: return "#00d4ff"
    if sim >= 0.40: return "#f59e0b"
    return "#ef4444"


def render_result(i, doc, meta, dist):
    sim = round(1 - dist, 3)
    color = similarity_color(sim)
    pct = int(sim * 100)

    fossil = meta.get('fossil_share', -1)
    renew  = meta.get('renew_share', -1)
    coal   = meta.get('coal_share', -1)

    tags_html = ""
    if fossil >= 0:
        tags_html += f'<span class="meta-tag tag-fossil">fossil {fossil}%</span>'
    if renew >= 0:
        tags_html += f'<span class="meta-tag tag-renew">renew {renew}%</span>'
    if coal >= 0:
        tags_html += f'<span class="meta-tag tag-coal">coal {coal}%</span>'

    st.markdown(f"""
    <div class="result-card">
        <div class="rank-badge">#{i+1}</div>
        <div>
            <span class="country-name">{meta['country']}</span>
            <span class="year-tag">{meta['year']}</span>
        </div>
        <div class="sim-row">
            <span class="sim-label">similarity</span>
            <div class="sim-bar-bg">
                <div class="sim-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
            <span class="sim-val" style="color:{color};">{sim}</span>
        </div>
        <div class="meta-tags">{tags_html}</div>
        <div class="doc-text">{doc[:180]}…</div>
    </div>
    """, unsafe_allow_html=True)


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⚡ Energy Semantic Search</h1>
  <p>4,459 country-year energy profiles · all-MiniLM-L6-v2 · ChromaDB · OWID Dataset</p>
</div>
""", unsafe_allow_html=True)

# Check data file
if not os.path.exists("owid-energy-data.csv"):
    st.error("**`owid-energy-data.csv` not found.** Place the OWID energy CSV in the same folder as this app, then restart.")
    st.info("Download from: https://github.com/owid/energy-data")
    st.stop()

# Loading
with st.spinner("Building vector index — first run takes ~60 seconds…"):
    collection, documents, metadatas, df = load_data_and_index()

model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙ filters</div>', unsafe_allow_html=True)

    year_min = int(df['year'].min())
    year_max = int(df['year'].max())
    year_range = st.slider("Year range", year_min, year_max, (2000, year_max), step=1)

    n_results = st.slider("Results to show", 3, 15, 6)

    st.markdown("---")
    st.markdown('<div class="sidebar-title">💡 example queries</div>', unsafe_allow_html=True)

    examples = [
        "countries rapidly transitioning away from fossil fuels",
        "petrostates with almost no renewable energy",
        "clean energy leaders in Europe",
        "low energy consumption developing nations",
        "explosive solar growth after 2018",
        "high fossil dependency but growing renewables",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["query_input"] = ex

    st.markdown("---")
    st.markdown(f'<div class="sidebar-title">📊 index stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; color: #64748b; line-height: 2;">
    Documents: <span style="color:#e2e8f0">{collection.count():,}</span><br>
    Countries: <span style="color:#e2e8f0">{df['country'].nunique()}</span><br>
    Year span: <span style="color:#e2e8f0">{df['year'].min()}–{df['year'].max()}</span><br>
    Embedding dim: <span style="color:#e2e8f0">384</span><br>
    Distance: <span style="color:#e2e8f0">cosine</span>
    </div>
    """, unsafe_allow_html=True)

# ── Main search ───────────────────────────────────────────────────────────────
query = st.text_input(
    "",
    placeholder="e.g.  'coal-dependent nations with low renewables'  or  'offshore wind pioneers'",
    key="query_input",
    label_visibility="collapsed"
)

col1, col2 = st.columns([3, 1])
with col1:
    search_btn = st.button("Search", use_container_width=True)

if query and search_btn:
    t0 = time.time()

    filters = {"$and": [
        {"year": {"$gte": year_range[0]}},
        {"year": {"$lte": year_range[1]}}
    ]}

    query_vec = model.encode([query], convert_to_numpy=True)
    results = collection.query(
        query_embeddings=query_vec.tolist(),
        n_results=n_results,
        include=['documents', 'metadatas', 'distances'],
        where=filters
    )

    elapsed = round(time.time() - t0, 3)

    docs_r  = results['documents'][0]
    metas_r = results['metadatas'][0]
    dists_r = results['distances'][0]

    st.markdown(f'<div class="status-bar">query: "{query}" · {len(docs_r)} results · {elapsed}s</div>', unsafe_allow_html=True)

    if not docs_r:
        st.markdown('<div class="no-results">No results found. Try a different query or expand the year range.</div>', unsafe_allow_html=True)
    else:
        for i, (doc, meta, dist) in enumerate(zip(docs_r, metas_r, dists_r)):
            render_result(i, doc, meta, dist)

elif not query:
    st.markdown("""
    <div class="no-results" style="padding: 4rem 2rem;">
        <div style="font-size: 2rem; margin-bottom: 1rem;">🔍</div>
        Type a natural language query above.<br><br>
        <span style="color: #334155;">This system understands <em>meaning</em>, not just keywords.<br>
        "coal-dependent nations" finds them even if that phrase never appears in the data.</span>
    </div>
    """, unsafe_allow_html=True)
