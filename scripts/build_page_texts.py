from pathlib import Path
import json
import fitz

RAW_DIR = Path("data/raw_pdfs")
META_PATH = Path("data/metadata/metadata.jsonl")
OUTPUT_PATH = Path("data/metadata/metadata_with_text.jsonl")

def load_metadata(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def main():
    records = load_metadata(META_PATH)
    enriched_records = []

    pdf_cache = {}

    for record in records:
        pdf_path = record["pdf_path"]
        page_num = record["page_num"]

        if pdf_path not in pdf_cache:
            pdf_cache[pdf_path] = fitz.open(pdf_path)

        doc = pdf_cache[pdf_path]
        page = doc.load_page(page_num - 1)
        text = page.get_text("text").strip()

        record["page_text"] = text
        enriched_records.append(record)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in enriched_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] saved to {OUTPUT_PATH}")
    print(f"[OK] total pages: {len(enriched_records)}")

if __name__ == "__main__":
    main()
