import numpy as np
import faiss

EMBEDDINGS_PATH = "data/processed_chunks/embeddings.npy"
INDEX_PATH = "data/faiss_index.index"

print("Loading embeddings...")
embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

dimension = embeddings.shape[1]
print("Embedding dimension:", dimension)

print("Building FAISS index...")
index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine-like)
index.add(embeddings)

print("Saving FAISS index...")
faiss.write_index(index, INDEX_PATH)

print("✅ FAISS index built and saved successfully!")