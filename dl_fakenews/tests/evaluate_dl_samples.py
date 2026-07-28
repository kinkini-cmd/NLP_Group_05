import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.app import MIN_RELIABLE_CHARS
from dl_model.xlm_roberta.predict_xlmr import has_claim_model, predict_claim, predict_news


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the XLM-RoBERTa detector on labeled CSV samples."
    )
    parser.add_argument(
        "csv_path",
        help="CSV with text,label columns. Label must be REAL or FAKE.",
    )
    parser.add_argument(
        "--include-short",
        action="store_true",
        help="Evaluate short title/claim samples instead of marking them UNCERTAIN.",
    )
    parser.add_argument(
        "--claim-model",
        action="store_true",
        help="Use the fine-tuned claim model for short claim samples.",
    )
    return parser.parse_args()


def load_samples(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as sample_file:
        reader = csv.DictReader(sample_file)
        required = {"text", "label"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain text and label columns.")

        return list(reader)


def main():
    args = parse_args()
    samples = load_samples(args.csv_path)

    evaluated = 0
    skipped = 0
    correct = 0

    for index, sample in enumerate(samples, start=1):
        text = sample["text"].strip()
        expected = sample["label"].strip().upper()

        if expected not in {"REAL", "FAKE"}:
            raise ValueError(f"Row {index} has unsupported label: {expected}")

        is_short = len(text) < MIN_RELIABLE_CHARS

        if is_short and args.claim_model and not has_claim_model():
            raise RuntimeError(
                "Claim model not found. Train it with "
                "dl_model/xlm_roberta/train_claim_xlmr.py first."
            )

        if is_short and not args.include_short and not args.claim_model:
            skipped += 1
            print(
                f"{index}\texpected={expected}\tpredicted=UNCERTAIN"
                f"\tchars={len(text)}\tcorrect=skipped"
            )
            continue

        if is_short and args.claim_model:
            predicted, confidence = predict_claim(text)
        else:
            predicted, confidence = predict_news(text)
        is_correct = predicted == expected
        evaluated += 1
        correct += int(is_correct)

        print(
            f"{index}\texpected={expected}\tpredicted={predicted}"
            f"\tconfidence={confidence}\tchars={len(text)}\tcorrect={is_correct}"
        )

    accuracy = (correct / evaluated * 100) if evaluated else 0.0

    print()
    print(f"evaluated={evaluated}")
    print(f"skipped_short={skipped}")
    print(f"accuracy={correct}/{evaluated}={accuracy:.2f}%")


if __name__ == "__main__":
    main()
