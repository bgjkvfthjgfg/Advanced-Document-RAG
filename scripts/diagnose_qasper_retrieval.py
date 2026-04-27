import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def cosine_similarity(query_vec, doc_vecs):
    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    return np.dot(doc_norms, query_norm)

def contains_gold_evidence(chunk_text, gold_evidence_list):
    chunk_lower = chunk_text.lower()
    for ev in gold_evidence_list:
        ev_lower = ev.lower().strip()
        if not ev_lower:
            continue
        # 宽松一点：只要 gold evidence 的前一段关键词片段在 chunk 里
        key_part = ev_lower[:120]
        if key_part in chunk_lower:
            return True
    return False

def main():
    subset = load_jsonl("data/benchmark/qasper_subset.jsonl")
    corpus = load_jsonl("data/benchmark/qasper_corpus.jsonl")

    # 先只看前 5 条
    subset = subset[:5]

    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for sample in subset:
        question = sample["question"]
        target_doc = sample["doc_id"]
        gold_evidence = sample["gold_evidence"]

        doc_chunks = [x for x in corpus if x["doc_id"] == target_doc]
        texts = [x["text"] for x in doc_chunks]

        text_embs = emb_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        q_emb = emb_model.encode(question, convert_to_numpy=True, show_progress_bar=False)

        sims = cosine_similarity(q_emb, text_embs)
        ranked_idx = np.argsort(-sims)

        hit_rank = None
        hit_chunk = None

        for rank, idx in enumerate(ranked_idx[:10], 1):
            chunk = doc_chunks[idx]
            if contains_gold_evidence(chunk["text"], gold_evidence):
                hit_rank = rank
                hit_chunk = chunk
                break

        print("=" * 80)
        print("sample_id:", sample["sample_id"])
        print("question:", question)
        print("gold_answer:", sample["gold_answer"])

        if hit_rank is None:
            print("gold evidence hit: NOT FOUND in Top-10")
        else:
            print(f"gold evidence hit: Top-{hit_rank}")
            print(f"section: {hit_chunk['section_name']} | paragraph_id: {hit_chunk['paragraph_id']}")
            print("chunk preview:", hit_chunk["text"][:300])

if __name__ == "__main__":
    main()