from datasets import load_dataset

def main():
    dataset = load_dataset(
        "parquet",
        data_files={"train": "data/qasper/train-00000-of-00001.parquet"}
    )

    sample = dataset["train"][0]
    title = sample["title"]
    section_names = sample["full_text"]["section_name"]
    paragraphs = sample["full_text"]["paragraphs"]

    print("TITLE:", title)
    print("=" * 80)

    for i, (sec, para_list) in enumerate(zip(section_names, paragraphs)):
        print(f"\n[Section {i}] {sec}")
        for j, para in enumerate(para_list[:2]):
            print(f"  - Paragraph {j}: {para[:300]}")
        if i >= 2:
            break

if __name__ == "__main__":
    main()