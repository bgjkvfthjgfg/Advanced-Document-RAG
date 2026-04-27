import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor

MODEL_NAME = "deepseek-ai/deepseek-vl2-small"

def main():
    print(f"[INFO] loading processor from {MODEL_NAME}")
    processor = DeepseekVLV2Processor.from_pretrained(MODEL_NAME)

    print(f"[INFO] loading model from {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("[OK] processor loaded")
    print("[OK] model loaded")
    print("[INFO] model device:", next(model.parameters()).device)

if __name__ == "__main__":
    main()
