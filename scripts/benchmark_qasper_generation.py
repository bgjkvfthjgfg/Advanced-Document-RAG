import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def cosine_similarity(query_vec, doc_vecs):
    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    return np.dot(doc_norms, query_norm)


def keyword_overlap_score(question, text):
    return len(set(question.lower().split()) & set(text.lower().split()))


def definition_bonus(question, text):
    q = question.lower()
    t = text.lower()
    if q.startswith("what is") or q.startswith("what are"):
        for p in ["consists of", "is a", "is an", "defined as"]:
            if p in t:
                return 0.12
    return 0.0


def split_sentences(text):
    return [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]


def extractive_fallback(question, evidences):
    q = question.lower()

    if not (q.startswith("what is") or q.startswith("what are")):
        return None

    target = q.replace("what is ", "").replace("what are ", "").strip(" ?.")

    sentences = []
    for e in evidences:
        sentences += split_sentences(e)

    for s in sentences:
        if target in s.lower() and "consists of" in s.lower():
            return s.strip()

    return None


def build_prompt(question, evidences):
    ev_text = "\n\n".join(evidences)
    return f"""Answer based only on the evidence.

Question: {question}

Evidence:
{ev_text}

Answer:"""


def run_system(question, doc_chunks, emb_model, reader_type):
    texts = [x["text"] for x in doc_chunks]
    text_embs = emb_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    q_emb = emb_model.encode(question, convert_to_numpy=True, show_progress_bar=False)

    sims = cosine_similarity(q_emb, text_embs)

    # baseline / rerank
    if "rerank" in reader_type:
        scores = []
        for i, chunk in enumerate(doc_chunks):
            score = sims[i] + 0.05 * keyword_overlap_score(question, chunk["text"]) + definition_bonus(question, chunk["text"])
            scores.append(score)
        scores = np.array(scores)
    else:
        scores = sims

    top_idx = np.argsort(-scores)[:2]
    evidences = [doc_chunks[i]["text"] for i in top_idx]

    # fallback only for 14B
    if reader_type == "rerank_14B":
        fb = extractive_fallback(question, evidences)
        if fb:
            return fb, evidences

    # reader
    if "3B" in reader_type:
        model_path = "/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
    else:
        model_path = "/root/autodl-tmp/models/Qwen2.5-14B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    prompt = build_prompt(question, evidences)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ans = ans.split("Answer:")[-1].strip()

    return ans, evidences


def main():
    benchmark = load_jsonl("data/benchmark/qasper_benchmark_50.jsonl")
    corpus = load_jsonl("data/benchmark/qasper_corpus.jsonl")

    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    systems = ["baseline_3B", "rerank_3B", "rerank_14B"]

    with open("results/qasper_generation_outputs.jsonl", "w", encoding="utf-8") as fout:
        for sample in benchmark:
            doc_chunks = [x for x in corpus if x["doc_id"] == sample["doc_id"]]

            for sys in systems:
                ans, evs = run_system(sample["question"], doc_chunks, emb_model, sys)

                record = {
                    "sample_id": sample["sample_id"],
                    "system": sys,
                    "question": sample["question"],
                    "pred_answer": ans,
                    "evidence": evs
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Saved generation outputs.")


if __name__ == "__main__":
    main()