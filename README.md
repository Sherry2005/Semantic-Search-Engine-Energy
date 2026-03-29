# ⚡ Energy Semantic Search

Natural language search over **4,459 country-year energy profiles** from the [Our World in Data](https://github.com/owid/energy-data) global energy dataset (1900–2026).

Built with `sentence-transformers`, `ChromaDB`, and `Streamlit`.

---

## What it demonstrates

| Technique | Description |
|---|---|
| **Document synthesis** | Converts numeric columns into searchable text |
| **Semantic embeddings** | `all-MiniLM-L6-v2` (384-dim vectors) |
| **Vector search** | ChromaDB with cosine distance + HNSW index |
| **Metadata filtering** | Year-range filters applied server-side |
| **Streamlit UI** | Clean, portfolio-ready interface |

---

## Setup

### 1. Get the dataset

Download `owid-energy-data.csv` from:
```
https://github.com/owid/energy-data
```
Place it in the same folder as `app.py`.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
streamlit run app.py
```

The first run builds the vector index (~60 seconds). Subsequent runs reuse the cached index instantly.

---

## Example queries

These all work because the model understands **meaning**, not just keywords:

- `countries rapidly transitioning away from fossil fuels`
- `petrostates with almost no renewable energy`
- `clean energy leaders in Europe`
- `explosive solar growth after 2018`
- `low energy consumption developing nations`
- `high fossil dependency but growing renewables`

---

## How it works

```
User query → embed (MiniLM) → 384-dim vector
                                     ↓
                          ChromaDB cosine search
                                     ↓
                     Top-k most similar country-year profiles
```

Each row in the dataset is first converted to a text document like:
> "Germany energy profile 2023. fossil fuels 75.4% (high fossil dependency). coal 12.1%. gas 22.3%. renewables 22.8% (high renewable share). wind 10.3%."

This lets the embedding model link numeric data to human language concepts.

---

## Author

Built by [Sherry](https://github.com/Sherry2005) as a portfolio project.
