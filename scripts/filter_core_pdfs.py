import os
import shutil
import pandas as pd

SOURCE_META = "data/raw_pdfs_all_metadata.csv"
TARGET_DIR = "data/raw_pdfs_core"
TARGET_META = "data/raw_pdfs_core_metadata.csv"

os.makedirs(TARGET_DIR, exist_ok=True)

# 这些关键词偏向自动驾驶 / 轨迹预测 / motion forecasting / interaction
KEEP_KEYWORDS = [
    "trajectory prediction",
    "trajectory forecasting",
    "motion prediction",
    "motion forecasting",
    "autonomous driving",
    "autonomous vehicle",
    "vehicle trajectory",
    "interactive prediction",
    "interaction-aware",
    "interaction aware",
    "multi-agent",
    "driving graph",
    "behavior prediction",
    "intention prediction",
    "trajectory planning",
    "driving scene",
    "waymo",
    "nuplan",
    "argoverse",
    "highway",
    "lane change",
    "overtaking",
    "hypergraph",
    "gameformer",
    "transformer for trajectory",
]

# 这些关键词偏向明显无关领域，先排掉
DROP_KEYWORDS = [
    "alzheimer",
    "tourism",
    "eeg",
    "glacier",
    "ocean",
    "poverty",
    "railway",
    "aquatic",
    "holography",
    "pv power",
    "emotion recognition",
    "social sciences",
    "arts and humanities",
    "crowdworkers",
    "biology",
    "medical",
]

def normalize(text: str) -> str:
    return str(text).lower().strip()

def should_keep(filename: str) -> bool:
    text = normalize(filename)

    if any(k in text for k in DROP_KEYWORDS):
        return False

    if any(k in text for k in KEEP_KEYWORDS):
        return True

    return False

def main():
    df = pd.read_csv(SOURCE_META)

    keep_rows = []
    copied = 0

    # 清空旧 core 目录里的 pdf
    for fname in os.listdir(TARGET_DIR):
        if fname.lower().endswith(".pdf"):
            os.remove(os.path.join(TARGET_DIR, fname))

    for _, row in df.iterrows():
        filename = row["filename"]
        merged_path = row["merged_path"]

        if not should_keep(filename):
            continue

        if not os.path.exists(merged_path):
            continue

        dst_path = os.path.join(TARGET_DIR, filename)
        shutil.copy2(merged_path, dst_path)
        copied += 1

        keep_rows.append({
            "filename": filename,
            "source": row["source"],
            "merged_path": dst_path,
        })

    core_df = pd.DataFrame(keep_rows)
    core_df.to_csv(TARGET_META, index=False, encoding="utf-8-sig")

    print(f"[OK] core pdf count: {copied}")
    print(f"[OK] core dir: {TARGET_DIR}")
    print(f"[OK] core metadata saved to: {TARGET_META}")

if __name__ == "__main__":
    main()
