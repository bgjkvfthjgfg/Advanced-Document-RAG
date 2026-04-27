import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images

MODEL_NAME = "deepseek-ai/deepseek-vl2-small"
RESULT_PATH = "results/retrieval_result.json"
SAVE_PATH = Path("results/deepseek_answer_top1.json")

def main():
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    query = data["query"]
    top1 = data["results"][0]
    top1_image = top1["image_path"]

    print("[INFO] query:", query)
    print("[INFO] top1 image:", top1_image)

    print(f"[INFO] loading processor from {MODEL_NAME}")
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(MODEL_NAME)
    tokenizer = vl_chat_processor.tokenizer

    print(f"[INFO] loading model from {MODEL_NAME}")
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    vl_gpt = vl_gpt.eval()

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\nPlease answer the question based on this document page:\n{query}",
            "images": [top1_image],
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

    print("\n========== DeepSeek-VL2 Answer ==========\n")
    print(answer)

    output = {
        "query": query,
        "top1_result": top1,
        "answer": answer
    }

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] answer saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
