import os
import shutil
import pandas as pd

BULK_DIR = "data/raw_pdfs_bulk"
OPENALEX_DIR = "data/raw_pdfs_openalex"
MERGED_DIR = "data/raw_pdfs_all"
META_PATH = "data/raw_pdfs_all_metadata.csv"

os.makedirs(MERGED_DIR, exist_ok=True)

def collect_files(src_dir, source_name):
    rows = []
    if not os.path.exists(src_dir):
        return rows

    for fname in os.listdir(src_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        rows.append({
            "filename": fname,
            "source": source_name,
            "src_path": os.path.join(src_dir, fname),
        })
    return rows

def main():
    rows = []
    rows.extend(collect_files(BULK_DIR, "semantic_scholar"))
    rows.extend(collect_files(OPENALEX_DIR, "openalex"))

    print(f"[INFO] collected raw file records: {len(rows)}")

    seen = set()
    merged_rows = []

    for row in rows:
        fname = row["filename"]
        fname_key = fname.lower()

        if fname_key in seen:
            continue
        seen.add(fname_key)

        src_path = row["src_path"]
        dst_path = os.path.join(MERGED_DIR, fname)

        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

        merged_rows.append({
            "filename": fname,
            "source": row["source"],
            "merged_path": dst_path
        })

    df = pd.DataFrame(merged_rows)
    df.to_csv(META_PATH, index=False, encoding="utf-8-sig")

    print(f"[OK] merged pdf count: {len(df)}")
    print(f"[OK] merged dir: {MERGED_DIR}")
    print(f"[OK] metadata saved to: {META_PATH}")

if __name__ == "__main__":
    main()
