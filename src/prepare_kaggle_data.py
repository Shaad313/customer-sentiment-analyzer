"""Create a balanced CSV sample from Kaggle's FastText Amazon review file."""

from __future__ import annotations

import argparse
import bz2
import csv
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"__label__1": "negative", "__label__2": "positive"}


def prepare(input_path: Path, output_path: Path, samples_per_class: int = 5000, seed: int = 42) -> int:
    rng = random.Random(seed)
    reservoirs: dict[str, list[str]] = {label: [] for label in LABELS.values()}
    seen: Counter[str] = Counter()

    with bz2.open(input_path, "rt", encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, start=1):
            label, separator, review = line.partition(" ")
            sentiment = LABELS.get(label)
            review = review.strip()
            if not sentiment or not separator or not review:
                continue

            seen[sentiment] += 1
            bucket = reservoirs[sentiment]
            if len(bucket) < samples_per_class:
                bucket.append(review)
            else:
                slot = rng.randrange(seen[sentiment])
                if slot < samples_per_class:
                    bucket[slot] = review

            if line_number % 500_000 == 0:
                print(f"Read {line_number:,} source reviews...")

    missing = [label for label, rows in reservoirs.items() if len(rows) < samples_per_class]
    if missing:
        raise ValueError(f"Not enough source rows for requested sample size. Counts: {dict(seen)}")

    rows = [(review, label) for label, reviews in reservoirs.items() for review in reviews]
    rng.shuffle(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("text", "sentiment"))
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} balanced reviews to {output_path}")
    print(f"Source counts: {dict(seen)}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw" / "train.ft.txt.bz2")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "reviews.csv")
    parser.add_argument("--samples-per-class", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.input, args.output, args.samples_per_class, args.seed)


if __name__ == "__main__":
    main()
