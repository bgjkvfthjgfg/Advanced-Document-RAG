import json
from datasets import load_dataset

def load_benchmark_doc_ids(benchmark_path):
    doc_ids = set()
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            doc_ids.add(item["doc_id"])
    return doc_ids

def main():
    dataset = load_dataset(
        "parquet",
        data_files={"train": "data/qasper/train-00000-of-00001.parquet"}
    )

    target_doc_ids = load_benchmark_doc_ids("data/benchmark/qasper_benchmark_50.jsonl")

    output_path = "data/benchmark/qasper_corpus.jsonl"
    count = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in dataset["train"]:
            title = sample["title"]

            if title not in target_doc_ids:
                continue

            section_names = sample["full_text"]["section_name"]
            paragraphs = sample["full_text"]["paragraphs"]

            for sec_idx, (sec_name, para_list) in enumerate(zip(section_names, paragraphs)):
                for para_idx, para in enumerate(para_list):
                    para = para.strip()
                    if not para:
                        continue

                    record = {
                        "doc_id": title,
                        "section_id": sec_idx,
                        "section_name": sec_name,
                        "paragraph_id": para_idx,
                        "text": para
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

    print(f"Saved {count} corpus chunks to {output_path}")

if __name__ == "__main__":
    main()