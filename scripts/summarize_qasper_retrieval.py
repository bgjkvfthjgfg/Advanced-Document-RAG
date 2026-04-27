import json
import math

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def mean(values):
    if not values:
        return None
    return sum(values) / len(values)

def main():
    records = load_jsonl("results/qasper_retrieval_benchmark.jsonl")

    total = len(records)

    baseline_hits = [r for r in records if r["baseline_hit_rank"] is not None]
    rerank_hits = [r for r in records if r["rerank_hit_rank"] is not None]

    baseline_ranks = [r["baseline_hit_rank"] for r in records if r["baseline_hit_rank"] is not None]
    rerank_ranks = [r["rerank_hit_rank"] for r in records if r["rerank_hit_rank"] is not None]

    improved_count = sum(1 for r in records if r["improved"])

    print("=== Retrieval Benchmark Summary ===")
    print(f"total_samples: {total}")
    print(f"baseline_hit@10: {len(baseline_hits)}/{total} = {len(baseline_hits)/total:.4f}")
    print(f"rerank_hit@10:   {len(rerank_hits)}/{total} = {len(rerank_hits)/total:.4f}")
    print(f"baseline_avg_hit_rank: {mean(baseline_ranks):.4f}" if baseline_ranks else "baseline_avg_hit_rank: None")
    print(f"rerank_avg_hit_rank:   {mean(rerank_ranks):.4f}" if rerank_ranks else "rerank_avg_hit_rank: None")
    print(f"improved_count: {improved_count}/{total} = {improved_count/total:.4f}")

if __name__ == "__main__":
    main()