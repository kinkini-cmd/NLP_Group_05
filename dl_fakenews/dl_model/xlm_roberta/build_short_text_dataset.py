import argparse
import csv
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a balanced short-text dataset from real/fake article titles."
    )
    parser.add_argument("--true-csv", default="datasets/raw/True.csv")
    parser.add_argument("--fake-csv", default="datasets/raw/Fake.csv")
    parser.add_argument("--output", default="data/short_text_training.csv")
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=1000,
        help="Maximum title samples to keep for each label.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_titles(csv_path, label, max_rows, seed):
    df = pd.read_csv(csv_path)

    if "title" not in df.columns:
        raise ValueError(f"{csv_path} must contain a title column.")

    titles = (
        df["title"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
    )
    titles = titles[titles != ""]

    if len(titles) > max_rows:
        titles = titles.sample(n=max_rows, random_state=seed)

    return [
        {
            "label": label,
            "text": title,
            "source": csv_path,
        }
        for title in titles
    ]


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["label", "text", "source"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = []
    rows.extend(load_titles(args.true_csv, "REAL", args.max_per_label, args.seed))
    rows.extend(load_titles(args.fake_csv, "FAKE", args.max_per_label, args.seed))

    write_csv(rows, args.output)

    print(f"saved={args.output}")
    print(f"total={len(rows)}")
    print(f"REAL={sum(row['label'] == 'REAL' for row in rows)}")
    print(f"FAKE={sum(row['label'] == 'FAKE' for row in rows)}")


if __name__ == "__main__":
    main()
