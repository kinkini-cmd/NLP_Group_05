import csv
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATABASE = PROJECT_ROOT / "data" / "source_reputation.csv"


def normalize_domain(url_or_domain):
    value = str(url_or_domain).strip().lower()

    if not value:
        return ""

    parsed = urlparse(value)

    if not parsed.netloc:
        parsed = urlparse("//" + value)

    domain = parsed.netloc or parsed.path
    domain = domain.split("/")[0].split(":")[0]

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def load_source_reputation():
    if not SOURCE_DATABASE.exists():
        return {}

    sources = {}

    with SOURCE_DATABASE.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            domain = normalize_domain(row.get("domain", ""))
            label = row.get("label", "").strip().lower()

            if domain and label in {"real", "fake"}:
                sources[domain] = label

    return sources


def source_label(url):
    domain = normalize_domain(url)
    sources = load_source_reputation()

    return sources.get(domain)


def source_score(url):
    label = source_label(url)

    if label == "real":
        return 2

    if label == "fake":
        return -2

    return 0
