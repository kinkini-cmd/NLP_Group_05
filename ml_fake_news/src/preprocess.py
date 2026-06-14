import re

import pandas as pd
from sklearn.model_selection import train_test_split

URL_PATTERN = re.compile(r"http\S+")
SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text):
    """Keep more signals: punctuation matters for fake news detection"""
    text = str(text).lower()
    text = URL_PATTERN.sub(" url ", text)  # Replace URLs with placeholder
    text = SPACE_PATTERN.sub(" ", text)
    return text.strip()


def clean_text_aggressive(text):
    """Old aggressive cleaning for backward compatibility"""
    text = str(text).lower()
    text = URL_PATTERN.sub("", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = SPACE_PATTERN.sub(" ", text)
    return text.strip()


def build_content(row):
    title = clean_text(row.get("title", ""))
    text = clean_text(row.get("text", ""))
    return f"{title} {text}".strip()


def load_data():
    fake = pd.read_csv("data/raw/Fake.csv")
    true = pd.read_csv("data/raw/True.csv")

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], ignore_index=True)
    return df


def clean_data(df):
    # remove missing values
    df = df.dropna(subset=["text"])

    # remove duplicates
    df = df.drop_duplicates()

    # Apply the same cleaning used during prediction.
    if "title" in df.columns:
        df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)
    df["content"] = df.apply(build_content, axis=1)

    # REMOVE EMPTY TEXT
    df = df[df["content"].str.strip() != ""]

    df = df.reset_index(drop=True)

    return df


def split_data(df):
    train, temp = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["label"]
    )

    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=42,
        stratify=temp["label"]
    )

    return train, val, test


def save_data():
    print("Loading data...")
    df = load_data()

    print("Cleaning data...")
    df = clean_data(df)

    print("Splitting data...")
    train, val, test = split_data(df)

    print("Saving files...")

    df.to_csv("data/processed/final_clean.csv", index=False)
    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    print("Preprocessing completed!")


if __name__ == "__main__":
    save_data()
