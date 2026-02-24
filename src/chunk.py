import os
import re
import json
from PyPDF2 import PdfReader

# =========================
# CONFIG
# =========================
RAW_PDF_DIR = "data/raw_pdfs"
OUTPUT_DIR = "data/processed_chunks"
OUTPUT_FILE = "chunks.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# =========================
# CLEANING FUNCTION
# =========================
def clean_text(text):
    # Fix OCR numeric issues like 23.4 miles -> 23,400 miles
    text = re.sub(r"(\d+)\.(\d+)\s*miles", r"\1,\2 miles", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =========================
# CHUNKING FUNCTION
# =========================
def chunk_text(text, source_name, page_number, chunk_id_start):
    chunks = []
    start = 0
    chunk_id = chunk_id_start

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk,
            "source": source_name,
            "page": page_number
        })

        chunk_id += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks, chunk_id

# =========================
# MAIN
# =========================
def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_chunks = []
    chunk_id_counter = 0

    pdf_files = [f for f in os.listdir(RAW_PDF_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("⚠ No PDFs found in raw_pdfs folder.")
        return

    print(f"📂 Found {len(pdf_files)} PDF files.\n")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(RAW_PDF_DIR, pdf_file)
        print(f"Processing {pdf_file}...")

        reader = PdfReader(pdf_path)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if not text:
                continue

            text = clean_text(text)

            chunks, chunk_id_counter = chunk_text(
                text,
                source_name=pdf_file,
                page_number=page_num,
                chunk_id_start=chunk_id_counter
            )

            all_chunks.extend(chunks)

    # Save ALL chunks to chunks.json
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Created {len(all_chunks)} cleaned chunks from all PDFs.")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()