import json
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed_chunks/chunks.json"
EMBEDDINGS_PATH = "data/processed_chunks/embeddings.npy"
METADATA_PATH = "data/processed_chunks/metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading chunks...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    total_chunks = len(chunks)
    print(f"Total chunks loaded: {total_chunks}")

    if total_chunks == 0:
        print("❌ No chunks found. Exiting.")
        return

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"Embeddings shape: {embeddings.shape}")

    if embeddings.shape[0] != total_chunks:
        print("❌ Mismatch between chunks and embeddings!")
        return

    print("Saving embeddings...")
    np.save(EMBEDDINGS_PATH, embeddings.astype("float32"))

    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append({
            "id": i,
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"]
        })

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Embeddings + metadata saved successfully")


if __name__ == "__main__":
    main()