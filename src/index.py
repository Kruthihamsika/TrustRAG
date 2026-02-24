import numpy as np
import faiss
import os

EMBEDDINGS_PATH = "data/processed_chunks/embeddings.npy"
INDEX_PATH = "data/processed_chunks/faiss.index"


def normalize_vectors(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def main():
    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    print("Normalizing embeddings...")
    embeddings = normalize_vectors(embeddings)

    dimension = embeddings.shape[1]

    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(dimension)  # Inner Product (for cosine similarity)
    index.add(embeddings)

    print(f"Total vectors indexed: {index.ntotal}")

    print("Saving index...")
    faiss.write_index(index, INDEX_PATH)

    print("✅ FAISS index created successfully!")


if __name__ == "__main__":
    main()