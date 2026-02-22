import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/processed_chunks/chunks.json"
OUTPUT_PATH = "data/processed_chunks/embeddings.npy"
METADATA_PATH = "data/processed_chunks/metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading chunks...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    print("Saving embeddings...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, embeddings)

    metadata = [
        {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"]
        }
        for chunk in chunks
    ]

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Embeddings created successfully")
    print("Shape:", embeddings.shape)


if __name__ == "__main__":
    main()