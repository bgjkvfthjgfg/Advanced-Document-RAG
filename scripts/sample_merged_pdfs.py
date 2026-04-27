import os
import shutil
import pandas as pd

SOURCE_META = "data/raw_pdfs_all_metadata.csv"
TARGET_DIR = "data/raw_pdfs"
SAMPLE_SIZE = 100

os.makedirs(TARGET_DIR, exist_ok=True)

def main():
    df = pd.read_csv(SOURCE_META)
    sample_df = df.head(SAMPLE_SIZE).copy()

    # 清空目标目录中旧的 PDF
    for fname in os.listdir(TARGET_DIR):
        if fname.lower().endswith(".pdf"):
            os.remove(os.path.join(TARGET_DIR, fname))

    copied = 0
    for _, row in sample_df.iterrows():
        src_path = row["merged_path"]
        filename = row["filename"]
        dst_path = os.path.join(TARGET_DIR, filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied += 1

    print(f"[OK] sampled pdf count: {copied}")
    print(f"[OK] target dir: {TARGET_DIR}")

if __name__ == "__main__":
    main()
