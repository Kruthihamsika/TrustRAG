import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/processed_chunks/faiss.index"
METADATA_PATH = "data/processed_chunks/metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    print("Loading metadata...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("\n✅ FAISS Semantic Search Ready")

    while True:
        query = input("\nEnter your query (exit/quit/q to stop): ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting search...")
            break

        print("Embedding query...")
        query_embedding = model.encode(query).astype("float32")
        query_embedding = normalize_vector(query_embedding)

        query_embedding = np.expand_dims(query_embedding, axis=0)

        scores, indices = index.search(query_embedding, TOP_K)

        print("\n🔎 Top Results:\n")

        for score, idx in zip(scores[0], indices[0]):
            print(f"Similarity Score: {score:.4f}")
            print(f"Source: {metadata[idx]['source']}")
            print(f"Page: {metadata[idx]['page']}")
            print("Text Preview:")
            print(metadata[idx]['text'][:400])
            print("-" * 80)


if __name__ == "__main__":
    main()