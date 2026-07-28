import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


BASE_MODEL_PATH = Path(__file__).resolve().parent / "saved_model"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "claim_model"
LABEL_TO_ID = {
    "FAKE": 0,
    "REAL": 1,
}
ID_TO_LABEL = {
    0: "FAKE",
    1: "REAL",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune the saved XLM-RoBERTa model on short fact-check claims."
    )
    parser.add_argument(
        "--claims-csv",
        default="data/politifact_claims.csv",
        help="CSV with text,label columns.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Directory where the fine-tuned claim model will be saved.",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-base",
        action="store_true",
        help="Train only the classification head. Faster on CPU, less accurate.",
    )
    return parser.parse_args()


def load_claims(csv_path):
    df = pd.read_csv(csv_path)
    required = {"text", "label"}

    if not required.issubset(df.columns):
        raise ValueError("Claims CSV must contain text and label columns.")

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df = df[df["text"] != ""]
    df = df[df["label"].isin(LABEL_TO_ID)]
    df["labels"] = df["label"].map(LABEL_TO_ID)

    if df["labels"].nunique() != 2:
        raise ValueError("Claims CSV must contain both REAL and FAKE labels.")

    return df[["text", "labels"]]


def tokenize_dataset(dataset, tokenizer, max_length):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return dataset.map(tokenize, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def freeze_base_model(model):
    for name, parameter in model.named_parameters():
        trainable_parts = ("classifier", "score", "pooler")

        if not any(part in name for part in trainable_parts):
            parameter.requires_grad = False


def main():
    args = parse_args()
    df = load_claims(args.claims_csv)

    train_df, eval_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["labels"],
    )

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BASE_MODEL_PATH),
        local_files_only=True,
    )
    model.config.id2label = ID_TO_LABEL
    model.config.label2id = LABEL_TO_ID

    if args.freeze_base:
        freeze_base_model(model)

    train_dataset = tokenize_dataset(Dataset.from_pandas(train_df), tokenizer, args.max_length)
    eval_dataset = tokenize_dataset(Dataset.from_pandas(eval_df), tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"saved_claim_model={output_dir}")


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    main()
