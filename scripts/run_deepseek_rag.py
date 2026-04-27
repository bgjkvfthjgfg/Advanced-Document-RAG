import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images

# ====== paths ======
INDEX_DIR = Path("data/index")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = INDEX_DIR / "page_text_embeddings.npy"
RECORDS_PATH = INDEX_DIR / "page_text_records.pkl"

RETRIEVAL_SAVE_PATH = RESULTS_DIR / "retrieval_result.json"
FINAL_SAVE_PATH = RESULTS_DIR / "deepseek_rag_result.json"

# ====== models ======
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VLM_MODEL_NAME = "deepseek-ai/deepseek-vl2-small"

TOP_K = 3


def retrieve(query: str, top_k: int = 3):
    print(f"[INFO] loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"[INFO] loading embeddings from {EMB_PATH}")
    embeddings = np.load(EMB_PATH)

    print(f"[INFO] loading records from {RECORDS_PATH}")
    with open(RECORDS_PATH, "rb") as f:
        records = pickle.load(f)

    query_emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_emb
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    print("\n========== Retrieval Top-K ==========\n")
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
        results.append(item)

        print(f"[Top {rank}] score = {scores[idx]:.4f}")
        print(f"doc_name    : {record['doc_name']}")
        print(f"page_num    : {record['page_num']}")
        print(f"image_path  : {record['image_path']}")
        print(f"text_preview: {preview}")
        print("-" * 80)

    retrieval_output = {
        "query": query,
        "top_k": top_k,
        "results": results
    }

    with open(RETRIEVAL_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(retrieval_output, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] retrieval result saved to {RETRIEVAL_SAVE_PATH}")
    return retrieval_output


def generate_answer(query: str, top_results):
    image_paths = [item["image_path"] for item in top_results]

    print(f"\n[INFO] loading processor from {VLM_MODEL_NAME}")
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(VLM_MODEL_NAME)
    tokenizer = vl_chat_processor.tokenizer

    print(f"[INFO] loading model from {VLM_MODEL_NAME}")
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        VLM_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    vl_gpt = vl_gpt.eval()

    image_tokens = "\n".join(["<image>" for _ in image_paths])
    conversation = [
        {
            "role": "<|User|>",
            "content": f"{image_tokens}\nPlease answer the question based on these retrieved document pages:\n{query}",
            "images": image_paths,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    pil_images = load_pil_images(conversation)

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer


def main():
    query = input("请输入你的问题：").strip()
    if not query:
        print("问题不能为空。")
        return

    retrieval_output = retrieve(query=query, top_k=TOP_K)
    answer = generate_answer(query=query, top_results=retrieval_output["results"])

    print("\n========== DeepSeek-VL2 Final Answer ==========\n")
    print(answer)

    final_output = {
        "query": query,
        "top_k": TOP_K,
        "retrieval_results": retrieval_output["results"],
        "answer": answer
    }

    with open(FINAL_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] final rag result saved to {FINAL_SAVE_PATH}")


if __name__ == "__main__":
    main()
