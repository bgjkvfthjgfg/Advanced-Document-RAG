import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


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


def split_sentences(text):
    text = text.replace("\n", " ").strip()
    seps = [". ", "? ", "! ", "; "]
    sentences = [text]
    for sep in seps:
        new_sentences = []
        for s in sentences:
            parts = s.split(sep)
            for i, p in enumerate(parts):
                p = p.strip()
                if not p:
                    continue
                if i < len(parts) - 1:
                    new_sentences.append(p + sep.strip())
                else:
                    new_sentences.append(p)
        sentences = new_sentences
    return [s.strip() for s in sentences if s.strip()]


def extractive_fallback(question, evidences):
    q = question.lower().strip()

    # 先只处理定义类问题
    if not (q.startswith("what is") or q.startswith("what are")):
        return None

    # 从问题里抽一个简单的核心短语
    target_phrase = None
    if q.startswith("what is "):
        target_phrase = q.replace("what is ", "").strip(" ?.")
    elif q.startswith("what are "):
        target_phrase = q.replace("what are ", "").strip(" ?.")

    candidate_sentences = []
    for ev in evidences:
        candidate_sentences.extend(split_sentences(ev))

    definition_patterns = [
        "consists of",
        "is a",
        "is an",
        "refers to",
        "defined as",
    ]

    # 第一优先级：同时包含 target_phrase + 定义模式
    for sent in candidate_sentences:
        s = sent.lower()
        if target_phrase and target_phrase in s:
            for p in definition_patterns:
                if p in s:
                    return sent.strip()

    # 第二优先级：包含 target_phrase 的句子
    for sent in candidate_sentences:
        s = sent.lower()
        if target_phrase and target_phrase in s:
            return sent.strip()

    # 第三优先级：退回原始定义模式
    for sent in candidate_sentences:
        s = sent.lower()
        for p in definition_patterns:
            if p in s:
                return sent.strip()

    return None


def build_prompt(question, evidences):
    evidence_block = "\n\n".join(
        [f"[Evidence {i+1}]\n{e}" for i, e in enumerate(evidences)]
    )
    return f"""You are a strict document-grounded QA assistant.

Your task:
- Extract the answer directly from the evidence.
- Use the exact wording from the evidence whenever possible.

Rules:
1. Answer MUST be supported by the evidence.
2. If a definition is asked, return the definition sentence.
3. Do NOT summarize broadly.
4. If answer exists in evidence, DO NOT say "Insufficient evidence".
5. Only say "Insufficient evidence" if nothing relevant is found.

Question:
{question}

Evidence:
{evidence_block}

Answer:"""


def main():
    subset = load_jsonl("data/benchmark/qasper_subset.jsonl")
    corpus = load_jsonl("data/benchmark/qasper_corpus.jsonl")

    # 当前先固定跑第 1 条样本，后面做 benchmark 时再扩成循环
    sample = subset[0]
    question = sample["question"]
    target_doc = sample["doc_id"]

    doc_chunks = [x for x in corpus if x["doc_id"] == target_doc]
    texts = [x["text"] for x in doc_chunks]

    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    text_embs = emb_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    q_emb = emb_model.encode(question, convert_to_numpy=True, show_progress_bar=False)

    sims = cosine_similarity(q_emb, text_embs)

    # 先取 Top-5 baseline retrieval
    initial_top5_idx = np.argsort(-sims)[:5]
    initial_top5_chunks = [doc_chunks[i] for i in initial_top5_idx]

    # 在 Top-5 内做 rerank
    rerank_scores = []
    for idx in initial_top5_idx:
        chunk = doc_chunks[idx]
        overlap = keyword_overlap_score(question, chunk["text"])
        def_bonus = definition_bonus(question, chunk["text"])
        score = sims[idx] + 0.05 * overlap + def_bonus
        rerank_scores.append(score)

    rerank_scores = np.array(rerank_scores)
    reranked_order = np.argsort(-rerank_scores)
    final_top2_chunks = [initial_top5_chunks[i] for i in reranked_order[:2]]

    print("=== Question ===")
    print(question)

    print("\n=== Gold Answer ===")
    print(sample["gold_answer"])

    print("\n=== Initial Top-5 Retrieved Evidence ===")
    for rank, chunk in enumerate(initial_top5_chunks, 1):
        print(f"\n[Top {rank}] {chunk['section_name']} | paragraph_id={chunk['paragraph_id']}")
        print(chunk["text"][:400])

    print("\n=== Final Top-2 Evidence After Rerank ===")
    for rank, chunk in enumerate(final_top2_chunks, 1):
        print(f"\n[Rerank Top {rank}] {chunk['section_name']} | paragraph_id={chunk['paragraph_id']}")
        print(chunk["text"][:400])

    retrieved_texts = [x["text"] for x in final_top2_chunks]

    # 先尝试 extractive fallback
    fallback_answer = extractive_fallback(question, retrieved_texts)

    if fallback_answer is not None:
        answer = fallback_answer
        print("\n=== Answer Source ===")
        print("extractive_fallback")
    else:
        print("\n=== Answer Source ===")
        print("llm_generation")

        # 这里默认用 14B，后面 benchmark 时可以按 system_name 切换
        model_path = "/root/autodl-tmp/models/Qwen2.5-14B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

        prompt = build_prompt(question, retrieved_texts)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("Answer:")[-1].strip()

    print("\n=== Pred Answer After Rerank ===")
    print(answer)


if __name__ == "__main__":
    main()
