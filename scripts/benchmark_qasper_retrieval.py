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

def contains_gold_evidence(chunk_text, gold_evidence_list):
    chunk_lower = chunk_text.lower()
    for ev in gold_evidence_list:
        ev_lower = ev.lower().strip()
        if not ev_lower:
            continue
        key_part = ev_lower[:120]
        if key_part in chunk_lower:
            return True
    return False

def get_hit_rank(question, doc_chunks, gold_evidence, model, use_rerank=False):
    texts = [x["text"] for x in doc_chunks]
    text_embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    q_emb = model.encode(question, convert_to_numpy=True, show_progress_bar=False)
    sims = cosine_similarity(q_emb, text_embs)

    if use_rerank:
        combined_scores = []
        for i, chunk in enumerate(doc_chunks):
            overlap = keyword_overlap_score(question, chunk["text"])
            def_bonus = definition_bonus(question, chunk["text"])
            score = sims[i] + 0.05 * overlap + def_bonus
            combined_scores.append(score)
        scores = np.array(combined_scores)
    else:
        scores = sims

    ranked_idx = np.argsort(-scores)

    for rank, idx in enumerate(ranked_idx[:10], 1):
        chunk = doc_chunks[idx]
        if contains_gold_evidence(chunk["text"], gold_evidence):
            return rank
    return None

def main():
    benchmark = load_jsonl("data/benchmark/qasper_benchmark_50.jsonl")
    corpus = load_jsonl("data/benchmark/qasper_corpus.jsonl")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    output_path = "results/qasper_retrieval_benchmark.jsonl"

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in benchmark:
            question = sample["question"]
            target_doc = sample["doc_id"]
            gold_evidence = sample["gold_evidence"]

            doc_chunks = [x for x in corpus if x["doc_id"] == target_doc]

            baseline_rank = get_hit_rank(
                question, doc_chunks, gold_evidence, model, use_rerank=False
            )
            rerank_rank = get_hit_rank(
                question, doc_chunks, gold_evidence, model, use_rerank=True
            )

            improved = False
            if baseline_rank is None and rerank_rank is not None:
                improved = True
            elif baseline_rank is not None and rerank_rank is not None and rerank_rank < baseline_rank:
                improved = True

            record = {
                "sample_id": sample["sample_id"],
                "doc_id": sample["doc_id"],
                "question": question,
                "baseline_hit_rank": baseline_rank,
                "rerank_hit_rank": rerank_rank,
                "improved": improved
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved retrieval benchmark results to {output_path}")

if __name__ == "__main__":
    main()