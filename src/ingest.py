from pathlib import Path
from pypdf import PdfReader
import json

RAW_PDF_DIR = Path("data/raw_pdfs")
OUTPUT_DIR = Path("data/processed_chunks")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_pages(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": text.strip()
            })

    return pages


def main():
    all_pages = []

    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in data/raw_pdfs/")
        return

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        all_pages.extend(extract_pages(pdf_file))

    output_file = OUTPUT_DIR / "extracted_pages.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_pages)} pages to {output_file}")


if __name__ == "__main__":
    main()
