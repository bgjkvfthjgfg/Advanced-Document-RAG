from pathlib import Path
import fitz

RAW_DIR = Path("data/raw_pdfs")
IMG_DIR = Path("data/page_images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

def convert_pdf(pdf_path: Path):
    doc = fitz.open(pdf_path)
    pdf_name = pdf_path.stem
    out_dir = IMG_DIR / pdf_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_path = out_dir / f"page_{i+1:03d}.png"
        pix.save(str(img_path))

    print(f"[OK] {pdf_name}: {len(doc)} pages converted")

def main():
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/raw_pdfs")
        return

    for pdf_file in pdf_files:
        convert_pdf(pdf_file)

if __name__ == "__main__":
    main()
