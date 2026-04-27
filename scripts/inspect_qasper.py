from datasets import load_dataset
from pprint import pprint

def main():
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": "data/qasper/train-00000-of-00001.parquet"
        }
    )

    sample = dataset["train"][0]

    qa_idx = 0

    print("=== Title ===")
    print(sample["title"])

    print("\n=== Question ===")
    print(sample["qas"]["question"][qa_idx])

    print("\n=== Raw Answers ===")
    pprint(sample["qas"]["answers"][qa_idx])

    print("\n=== First Annotation ===")
    first_ann = sample["qas"]["answers"][qa_idx]["answer"][0]
    pprint(first_ann)

    print("\n=== Free-form Answer ===")
    print(first_ann["free_form_answer"])

    print("\n=== Extractive Spans ===")
    print(first_ann["extractive_spans"])

    print("\n=== Evidence ===")
    pprint(first_ann["evidence"])

if __name__ == "__main__":
    main()