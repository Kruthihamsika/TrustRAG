import json
import re
import math
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# =========================
# CONFIG
# =========================
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INDEX_PATH = "data/faiss_index.index"
METADATA_PATH = "data/processed_chunks/metadata.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"

TOP_K = 4
PROMPT_CHAR_LIMIT = 6000

# =========================
# GLOBAL BACKEND OBJECTS
# =========================
embed_model = None
reranker = None
index = None
metadata = None
bm25 = None


# =========================
# LOAD BACKEND (SAFE FOR STREAMLIT)
# =========================
def load_backend():
    global embed_model, reranker, index, metadata, bm25

    if embed_model is not None:
        return

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading reranker model...")
    reranker = CrossEncoder(RERANK_MODEL_NAME)

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    print("Loading metadata...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("Building BM25 index...")
    corpus = [m["text"] for m in metadata]
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    print("RAG backend ready.")


# =========================
# HYBRID RETRIEVAL
# =========================
def retrieve(query, top_k=TOP_K):

    load_backend()

    # ---- Dense Retrieval ----
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    dense_results = []

    for dist, idx in zip(distances[0], indices[0]):

        # Convert L2 distance to similarity
        dense_score = 1 / (1 + float(dist))

        dense_results.append({
            "text": metadata[idx]["text"],
            "source": metadata[idx]["source"],
            "page": metadata[idx]["page"],
            "dense_score": dense_score
        })

    # ---- Sparse Retrieval ----
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k]

    sparse_results = []

    for idx in top_bm25_idx:
        sparse_results.append({
            "text": metadata[idx]["text"],
            "source": metadata[idx]["source"],
            "page": metadata[idx]["page"],
            "bm25_score": float(bm25_scores[idx])
        })

    # ---- Combine Unique Results ----
    combined = {}

    for r in dense_results + sparse_results:
        key = (r["text"], r["source"], r["page"])
        if key not in combined:
            combined[key] = r

    combined_results = list(combined.values())

    # ---- Reranking ----
    if not combined_results:
        return []

    pairs = [(query, r["text"]) for r in combined_results]
    rerank_scores = reranker.predict(pairs)

    if isinstance(rerank_scores, float):
        rerank_scores = [rerank_scores]

    for r, score in zip(combined_results, rerank_scores):

        r["rerank_score"] = float(score)

        # Weighted hybrid scoring
        dense_part = r.get("dense_score", 0) * 1.0
        sparse_part = r.get("bm25_score", 0) * 0.05

        r["final_score"] = (
            r["rerank_score"] * 0.8 +
            dense_part * 0.15 +
            sparse_part
        )

    combined_results = sorted(
        combined_results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    return combined_results[:top_k]


# =========================
# PROMPT BUILDER
# =========================
def build_prompt(query, contexts):

    context_text = "\n\n".join([
        f"[Source: {c['source']} | Page {c['page']}]\n{c['text']}"
        for c in contexts
    ])

    prompt = f"""
You are a strict technical document extraction assistant.

Rules:
1. Use ONLY the provided context.
2. Extract values exactly as written.
3. Do NOT add external knowledge.
4. If answer is not clearly available, say:
   "The answer is not clearly available in the provided documents."
5. Be structured and concise.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:
"""

    return prompt[:PROMPT_CHAR_LIMIT]


# =========================
# NUMERIC VERIFICATION
# =========================
def verify_numeric_answer(answer, contexts):

    numbers = re.findall(r"\b\d[\d,\.]*\b", answer)
    context_text = " ".join([c["text"] for c in contexts])

    for num in numbers:

        # Ignore tiny numbers (list bullets etc.)
        if len(num) <= 2:
            continue

        if num not in context_text:
            return False

    return True


# =========================
# CONFIDENCE SCORING
# =========================
def compute_confidence(contexts, answer=None):

    if not contexts:
        return "LOW", 0.0

    top_score = contexts[0]["final_score"]

    # Smooth normalization
    normalized = 1 / (1 + math.exp(-top_score))

    numeric_valid = True
    if answer:
        numeric_valid = verify_numeric_answer(answer, contexts)

    # More realistic thresholds
    if normalized > 0.8 and numeric_valid:
        level = "HIGH"
    elif normalized > 0.6 and numeric_valid:
        level = "MEDIUM"
    else:
        level = "LOW"

    return level, round(normalized, 2)


# =========================
# OLLAMA GENERATION
# =========================
def generate_with_ollama(prompt):

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        if response.status_code != 200:
            return f"Model generation failed ({response.status_code})."

        data = response.json()

        if "response" not in data:
            return f"Ollama error: {data}"

        return data["response"].strip()

    except requests.exceptions.RequestException as e:
        return f"Ollama communication error: {str(e)}"