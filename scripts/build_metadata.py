from pathlib import Path
import json

RAW_DIR = Path("data/raw_pdfs")
IMG_DIR = Path("data/page_images")
META_DIR = Path("data/metadata")
META_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = META_DIR / "metadata.jsonl"

def main():
    records = []

    for pdf_dir in sorted(IMG_DIR.iterdir()):
        if not pdf_dir.is_dir():
            continue

        pdf_name = pdf_dir.name
        pdf_path = RAW_DIR / f"{pdf_name}.pdf"

        image_files = sorted(pdf_dir.glob("page_*.png"))
        for img_path in image_files:
            page_num = int(img_path.stem.split("_")[1])

            record = {
                "doc_name": pdf_name,
                "page_num": page_num,
                "image_path": str(img_path),
                "pdf_path": str(pdf_path)
            }
            records.append(record)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] metadata saved to {OUTPUT_PATH}")
    print(f"[OK] total pages: {len(records)}")

if __name__ == "__main__":
    main()
