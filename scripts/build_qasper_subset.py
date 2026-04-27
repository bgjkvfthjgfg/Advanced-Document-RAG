import json
from datasets import load_dataset

TARGET_SIZE = 50


def main():
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": "data/qasper/train-00000-of-00001.parquet"
        }
    )

    output = []
    sample_count = 0

    for doc in dataset["train"]:
        title = doc["title"]
        qas = doc["qas"]

        num_q = len(qas["question"])

        for i in range(num_q):
            answers = qas["answers"][i]["answer"]

            # 取第一个有效标注
            ann = answers[0]

            if ann["unanswerable"]:
                continue

            question = qas["question"][i]
            answer = ann["free_form_answer"]

            if answer == "":
                continue

            evidence = ann["evidence"]

            sample = {
                "sample_id": f"qasper_{sample_count:04d}",
                "doc_id": title,
                "question": question,
                "gold_answer": answer,
                "gold_evidence": evidence
            }

            output.append(sample)
            sample_count += 1

            if sample_count >= TARGET_SIZE:
                break

        if sample_count >= TARGET_SIZE:
            break

    with open("data/benchmark/qasper_subset.jsonl", "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(output)} samples to data/benchmark/qasper_subset.jsonl")


if __name__ == "__main__":
    main()