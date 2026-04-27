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

def keyword_overlap_score(question, text):
    q_words = set(question.lower().split())
    t_words = set(text.lower().split())
    return len(q_words & t_words)

def definition_bonus(question, text):
    q = question.lower()
    t = text.lower()
    bonus = 0.0
    if q.startswith("what is") or q.startswith("what are"):
        for p in ["consists of", "is a", "is an", "are", "refers to", "defined as"]:
            if p in t:
                bonus += 0.12
    return bonus

def main():
    subset = load_jsonl("data/benchmark/qasper_subset.jsonl")
    corpus = load_jsonl("data/benchmark/qasper_corpus.jsonl")

    sample = subset[0]
    question = sample["question"]
    target_doc = sample["doc_id"]

    doc_chunks = [x for x in corpus if x["doc_id"] == target_doc]
    texts = [x["text"] for x in doc_chunks]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    text_embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    q_emb = model.encode(question, convert_to_numpy=True, show_progress_bar=False)

    sims = cosine_similarity(q_emb, text_embs)

    # 原始 retrieval Top-5
    baseline_top5_idx = np.argsort(-sims)[:5]
    baseline_top5_chunks = [doc_chunks[i] for i in baseline_top5_idx]

    # rerank：embedding score + keyword overlap + definition bonus
    combined_scores = []
    for i, chunk in enumerate(doc_chunks):
        overlap = keyword_overlap_score(question, chunk["text"])
        def_bonus = definition_bonus(question, chunk["text"])
        score = sims[i] + 0.05 * overlap + def_bonus
        combined_scores.append(score)

    combined_scores = np.array(combined_scores)
    rerank_top5_idx = np.argsort(-combined_scores)[:5]

    print("=== Question ===")
    print(question)

    print("\n=== Gold Answer ===")
    print(sample["gold_answer"])

    print("\n=== Top 5 Retrieved Chunks ===")
    for rank, idx in enumerate(baseline_top5_idx, 1):
        chunk = doc_chunks[idx]
        print(f"\n[Top {rank}] score={sims[idx]:.4f}")
        print(f"section={chunk['section_name']} | paragraph_id={chunk['paragraph_id']}")
        print(chunk["text"][:500])

    print("\n=== Top 5 Retrieved Chunks After Definition-Aware Rerank ===")
    for rank, idx in enumerate(rerank_top5_idx, 1):
        chunk = doc_chunks[idx]
        overlap = keyword_overlap_score(question, chunk["text"])
        def_bonus = definition_bonus(question, chunk["text"])
        print(
            f"\n[Top {rank}] rerank_score={combined_scores[idx]:.4f} "
            f"| sim={sims[idx]:.4f} | overlap={overlap} | def_bonus={def_bonus:.2f}"
        )
        print(f"section={chunk['section_name']} | paragraph_id={chunk['paragraph_id']}")
        print(chunk["text"][:500])

if __name__ == "__main__":
    main()
