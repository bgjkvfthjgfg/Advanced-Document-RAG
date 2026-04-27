import argparse
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_index(index_dir):
    print(f"[INFO] loading retrieval records: {index_dir}/page_text_records.pkl")
    with open(f"{index_dir}/page_text_records.pkl", "rb") as f:
        records = pickle.load(f)

    print(f"[INFO] loaded records: {len(records)}")

    print(f"[INFO] loading embeddings: {index_dir}/page_text_embeddings.npy")
    embeddings = np.load(f"{index_dir}/page_text_embeddings.npy")
    print(f"[INFO] embeddings shape: {embeddings.shape}")

    return records, embeddings


def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def route_query(question: str):
    q = question.lower()

    safety_keywords = [
        "safety", "risk", "risks", "harm", "harms", "misuse",
        "alignment", "misaligned", "danger", "dangerous",
        "security", "policy", "preparedness", "jailbreak"
    ]

    matched = [kw for kw in safety_keywords if kw in q]

    if matched:
        route_info = {
            "route_name": "safety",
            "index_dir": "data/index",
            "reason": f"matched safety keywords: {matched}"
        }
    else:
        route_info = {
            "route_name": "default",
            "index_dir": "data/index",
            "reason": "no safety keyword matched; fallback to default index"
        }

    return route_info


def retrieve(query, model, records, embeddings, top_k=5):
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    emb_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.allclose(emb_norm.mean(), 1.0, atol=1e-1):
        embeddings = l2_normalize(embeddings)

    scores = embeddings @ query_emb
    topk_idx = np.argsort(scores)[::-1]

    results = []
    seen_prefix = set()

    for idx in topk_idx:
        r = records[idx]
        text = (r.get("page_text", "") or "").strip()
        if len(text) < 80:
            continue

        prefix = text[:200]
        if prefix in seen_prefix:
            continue
        seen_prefix.add(prefix)

        doc_name = r.get("doc_name", "unknown_doc")
        page_num = r.get("page_num", -1)
        display_page = page_num + 1 if isinstance(page_num, int) and page_num >= 0 else page_num

        results.append({
            "score": float(scores[idx]),
            "doc": doc_name,
            "page": display_page,
            "text": text[:1200]
        })

        if len(results) >= top_k:
            break

    return results


def build_context(evidence_list, max_context_chars=10000):
    parts = []
    total = 0

    for i, e in enumerate(evidence_list, start=1):
        block = (
            f"[Evidence {i}]\n"
            f"Document: {e['doc']}\n"
            f"Page: {e['page']}\n"
            f"Text:\n{e['text']}\n\n"
        )

        if total + len(block) > max_context_chars:
            break

        parts.append(block)
        total += len(block)

    return "".join(parts)


def build_prompt(question, context):
    prompt = f"""You are a safety-focused document QA assistant.

Use the provided evidence to answer the question.

Question:
{question}

Evidence:
{context}

Instructions:
- First write a concise summary of the answer based ONLY on the evidence.
- Then list the key risks.
- Each risk MUST be supported by the evidence.
- Do NOT introduce information not present in the evidence.

Output format:

Answer:
<2-3 sentence summary grounded in the evidence>

Key Risks:
- <risk 1> (supported by: <document> | page <n>)
- <risk 2> (supported by: <document> | page <n>)

Evidence Pages:
- <document> | page <n>
- <document> | page <n>
"""
    return prompt


def post_process_answer(answer: str) -> str:
    answer = answer.strip()
    idx = answer.find("Answer:")
    if idx != -1:
        answer = answer[idx:]
    return answer.strip()


def generate_answer(model, tokenizer, prompt, max_new_tokens=300):
    messages = [
        {"role": "system", "content": "You are a concise document-grounded assistant."},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    answer = post_process_answer(answer)
    return answer


def score_answer(answer: str) -> dict:
    score = 0
    reasons = []

    if "Answer:" in answer:
        score += 2
        reasons.append("contains Answer section")

    if "Key Risks:" in answer:
        score += 2
        reasons.append("contains Key Risks section")

    if "Evidence Pages:" in answer:
        score += 2
        reasons.append("contains Evidence Pages section")

    supported_count = answer.count("supported by:")
    score += min(supported_count, 5)
    reasons.append(f"supported-by count = {supported_count}")

    evidence_line_count = 0
    if "Evidence Pages:" in answer:
        tail = answer.split("Evidence Pages:", 1)[1]
        evidence_line_count = sum(
            1 for line in tail.splitlines()
            if line.strip().startswith("- ")
        )
        score += min(evidence_line_count, 5)
        reasons.append(f"evidence page lines = {evidence_line_count}")

    if "Insufficient evidence" in answer:
        score -= 1
        reasons.append("contains Insufficient evidence")

    if len(answer.strip()) < 80:
        score -= 2
        reasons.append("answer too short")

    return {
        "score": score,
        "reasons": reasons
    }


def print_agent_workflow():
    print("\n================ Agent Workflow ================\n")
    print("Step 1: Query Understanding")
    print("Step 2: Route Selection")
    print("Step 3: Retrieval Tool Call")
    print("Step 4: Candidate Answer Generation")
    print("Step 5: Answer Scoring")
    print("Step 6: Final Selection")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--llm_model_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--max_context_chars", type=int, default=10000)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    args = parser.parse_args()

    print_agent_workflow()

    print("\n[AGENT] understanding user query...")
    route_info = route_query(args.question)

    print("\n================ Query Routing ================\n")
    print(f"route_name={route_info['route_name']}")
    print(f"index_dir={route_info['index_dir']}")
    print(f"reason={route_info['reason']}")

    print("\n[AGENT] calling retrieval tool...")
    records, embeddings = load_index(route_info["index_dir"])

    print("[INFO] loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(f"\n[INFO] loading LLM: {args.llm_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    candidate_topks = [3, args.top_k]
    candidate_answers = []

    for idx, cur_top_k in enumerate(candidate_topks, start=1):
        print(f"\n[AGENT] generating candidate {idx} with retrieval top_k={cur_top_k} ...")
        evidence = retrieve(args.question, embed_model, records, embeddings, cur_top_k)

        print(f"\n================ Retrieved Evidence (Candidate {idx}, top_k={cur_top_k}) ================\n")
        for i, e in enumerate(evidence, start=1):
            print(f"[{i}] score={e['score']:.4f}")
            print(f"doc={e['doc']} | page={e['page']}")
            print(f"text_preview={e['text'][:500]}")
            print("-" * 80)

        context = build_context(evidence, max_context_chars=args.max_context_chars)
        prompt = build_prompt(args.question, context)

        answer = generate_answer(model, tokenizer, prompt, args.max_new_tokens)
        score_info = score_answer(answer)

        candidate_answers.append({
            "candidate_id": idx,
            "top_k": cur_top_k,
            "answer": answer,
            "score": score_info["score"],
            "reasons": score_info["reasons"]
        })

    print("\n[AGENT] scoring candidate answers...")

    print("\n================ Answer Scoring ================\n")
    for c in candidate_answers:
        print(f"Candidate {c['candidate_id']} | top_k={c['top_k']} | score={c['score']}")
        for reason in c["reasons"]:
            print(f"- {reason}")
        print("-" * 80)

    best = sorted(candidate_answers, key=lambda x: x["score"], reverse=True)[0]

    print("\n[AGENT] selecting final answer...")

    print("\n================ Final Selected Answer ================\n")
    print(f"selected_candidate={best['candidate_id']}")
    print(f"selected_top_k={best['top_k']}")
    print(f"selected_score={best['score']}")

    print("\n================ Final Structured Answer ================\n")
    print(best["answer"])


if __name__ == "__main__":
    main()