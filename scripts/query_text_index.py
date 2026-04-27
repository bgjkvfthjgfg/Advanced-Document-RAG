from pathlib import Path
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("data/index")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = INDEX_DIR / "page_text_embeddings.npy"
RECORDS_PATH = INDEX_DIR / "page_text_records.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print(f"[INFO] loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"[INFO] loading embeddings from {EMB_PATH}")
    embeddings = np.load(EMB_PATH)

    print(f"[INFO] loading records from {RECORDS_PATH}")
    with open(RECORDS_PATH, "rb") as f:
        records = pickle.load(f)

    query = input("请输入你的问题：").strip()
    if not query:
        print("问题不能为空。")
        return

    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_emb
    top_k = 3
    top_indices = np.argsort(scores)[::-1][:top_k]

    output = {
        "query": query,
        "top_k": top_k,
        "results": []
    }

    print("\n========== 检索结果 Top-3 ==========\n")
    for rank, idx in enumerate(top_indices, start=1):
        record = records[idx]
        preview = record.get("page_text", "")[:200].replace("\n", " ")

        item = {
            "rank": rank,
            "score": float(scores[idx]),
            "doc_name": record["doc_name"],
            "page_num": record["page_num"],
            "image_path": record["image_path"],
            "pdf_path": record["pdf_path"],
            "text_preview": preview
        }
        output["results"].append(item)

        print(f"[Top {rank}] score = {scores[idx]:.4f}")
        print(f"doc_name    : {record['doc_name']}")
        print(f"page_num    : {record['page_num']}")
        print(f"image_path  : {record['image_path']}")
        print(f"text_preview: {preview}")
        print("-" * 80)

    save_path = RESULTS_DIR / "retrieval_result.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] retrieval result saved to {save_path}")

if __name__ == "__main__":
    main()
