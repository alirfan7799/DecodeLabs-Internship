import pandas as pd

TEXT_COLUMNS = ["Responsibilities", "Minimum Qualifications", "Preferred Qualifications"]


def load_and_process(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    for c in TEXT_COLUMNS:
        df[c] = df[c].fillna("")

    df["full_text"] = df[TEXT_COLUMNS].agg(" ".join, axis=1)
    df["word_count"] = df["full_text"].str.split().str.len()

    # drop near-empty postings, they'd just be dead weight in the vector space
    df = df[df["word_count"] >= 15].reset_index(drop=True)

    return df


if __name__ == "__main__":
    processed = load_and_process("job_skills.csv")
    print(f"Processed {len(processed)} postings")
    print(processed[["Title", "Category", "word_count"]].head(10).to_string())
