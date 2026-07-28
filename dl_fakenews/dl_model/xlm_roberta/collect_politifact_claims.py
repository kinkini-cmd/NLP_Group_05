import argparse
import csv
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.politifact.com/factchecks/list/"
RULINGS = {
    "true": "REAL",
    "false": "FAKE",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect short labeled claims from PolitiFact list pages."
    )
    parser.add_argument(
        "--output",
        default="data/politifact_claims.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=5,
        help="Number of pages per label to collect.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to wait between requests.",
    )
    return parser.parse_args()


def build_url(ruling, page):
    query = urlencode({"ruling": ruling, "page": page})
    return f"{BASE_URL}?{query}"


def clean_text(text):
    return " ".join(text.split())


def extract_claims(html, label, source_url):
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for quote in soup.select(".m-statement__quote"):
        claim = clean_text(quote.get_text(" ", strip=True).strip("\""))

        if claim:
            rows.append(
                {
                    "label": label,
                    "text": claim,
                    "source": source_url,
                }
            )

    return rows


def collect_claims(pages, sleep_seconds):
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (compatible; dl-fakenews-claim-collector/1.0)"
            )
        }
    )

    rows = []

    for ruling, label in RULINGS.items():
        for page in range(1, pages + 1):
            url = build_url(ruling, page)
            response = session.get(url, timeout=30)
            response.raise_for_status()

            page_rows = extract_claims(response.text, label, url)
            rows.extend(page_rows)
            print(f"{ruling}\tpage={page}\tclaims={len(page_rows)}\turl={url}")
            time.sleep(sleep_seconds)

    return dedupe_rows(rows)


def dedupe_rows(rows):
    seen = set()
    unique = []

    for row in rows:
        key = (row["label"], row["text"].lower())

        if key in seen:
            continue

        seen.add(key)
        unique.append(row)

    return unique


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["label", "text", "source"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = collect_claims(args.pages, args.sleep)
    write_csv(rows, args.output)

    label_counts = {}
    for row in rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    print(f"saved={args.output}")
    print(f"total={len(rows)}")
    print(f"label_counts={label_counts}")


if __name__ == "__main__":
    main()
