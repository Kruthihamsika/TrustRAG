import json
import os
import tiktoken

INPUT_PATH = "data/processed_chunks/extracted_pages.json"
OUTPUT_PATH = "data/processed_chunks/chunks.json"

CHUNK_SIZE = 300
OVERLAP = 50

encoder = tiktoken.get_encoding("cl100k_base")


def chunk_text(text, chunk_size=300, overlap=50):
    tokens = encoder.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        source = page["source"]
        page_num = page["page"]

        text_chunks = chunk_text(text)

        for chunk in text_chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "page": page_num,
                "text": chunk
            })
            chunk_id += 1

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"✅ Created {len(all_chunks)} chunks → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
