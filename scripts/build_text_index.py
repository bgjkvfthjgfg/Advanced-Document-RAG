from pathlib import Path
import json
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

META_PATH = Path("data/metadata/metadata_with_text.jsonl")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = INDEX_DIR / "page_text_embeddings.npy"
RECORDS_PATH = INDEX_DIR / "page_text_records.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_records(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def main():
    print(f"[INFO] loading metadata from {META_PATH}")
    records = load_records(META_PATH)

    texts = []
    clean_records = []

    for record in records:
        text = record.get("page_text", "").strip()
        if not text:
            continue
        texts.append(text[:4000])
        clean_records.append(record)

    print(f"[INFO] valid pages with text: {len(clean_records)}")
    print(f"[INFO] loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("[INFO] encoding page texts...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    np.save(EMB_PATH, embeddings)
    with open(RECORDS_PATH, "wb") as f:
        pickle.dump(clean_records, f)

    print(f"[OK] embeddings saved to {EMB_PATH}")
    print(f"[OK] records saved to {RECORDS_PATH}")
    print(f"[OK] embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    main()
