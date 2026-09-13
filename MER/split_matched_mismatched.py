#!/usr/bin/env python3
"""
Split MER train_features.csv into matched vs mismatched pairs.

- Mismatched rows -> train_features_mismatch.csv (new file, same directory)
- Matched rows only -> train_features.csv (overwritten)

Match/mismatch can be determined by:
  A) Annotation CSV (EmoMV DS1 style): first column is folder path containing
     'MISMATCH' or 'MATCH'; second column is the key. Keys in MISMATCH rows
     are written to the mismatch file.
  B) Two label columns in the CSV: if two columns (e.g. audio_label and
     video_label) are given, rows where they differ are mismatched.

Usage:
  # Using annotation CSV (recommended for Dataset1)
  python MER/split_matched_mismatched.py \
    --input-csv MER/EmoMV_features/mert/Dataset1/train_features.csv \
    --annotation-csv /path/to/DS1_TRAIN_MATCH_MISMATCH_labels.csv

  # Using two label columns in the CSV
  python MER/split_matched_mismatched.py \
    --input-csv MER/EmoMV_features/af3/Dataset1/train_features.csv \
    --label-col-a audio_label --label-col-b video_label
"""
import argparse
import csv
from pathlib import Path


def get_mismatched_keys_from_annotation(annotation_path: Path) -> set:
    """Parse EmoMV DS1-style annotation; return set of keys that are mismatched."""
    mismatched = set()
    with open(annotation_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            folder, key = row[0].strip(), row[1].strip()
            if "MISMATCH" in folder.upper():
                mismatched.add(key)
    return mismatched


def run_with_annotation(input_csv: Path, annotation_csv: Path, dry_run: bool) -> None:
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    mismatched_keys = get_mismatched_keys_from_annotation(annotation_csv)
    matched_rows = [r for r in rows if r.get("key", "").strip() not in mismatched_keys]
    mismatched_rows = [r for r in rows if r.get("key", "").strip() in mismatched_keys]

    out_dir = input_csv.parent
    mismatch_path = out_dir / "train_features_mismatch.csv"

    if dry_run:
        print(f"Dry run: would write {len(matched_rows)} matched -> {input_csv}")
        print(f"         would write {len(mismatched_rows)} mismatched -> {mismatch_path}")
        return

    with open(input_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(matched_rows)
    print(f"Wrote {len(matched_rows)} matched rows to {input_csv}")

    with open(mismatch_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(mismatched_rows)
    print(f"Wrote {len(mismatched_rows)} mismatched rows to {mismatch_path}")


def run_with_label_columns(
    input_csv: Path, col_a: str, col_b: str, dry_run: bool
) -> None:
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if col_a not in fieldnames or col_b not in fieldnames:
        raise SystemExit(
            f"Columns {col_a!r} and/or {col_b!r} not in CSV. Available: {fieldnames}"
        )

    matched_rows = [r for r in rows if str(r.get(col_a, "")).strip() == str(r.get(col_b, "")).strip()]
    mismatched_rows = [r for r in rows if str(r.get(col_a, "")).strip() != str(r.get(col_b, "")).strip()]

    out_dir = input_csv.parent
    mismatch_path = out_dir / "train_features_mismatch.csv"

    if dry_run:
        print(f"Dry run: would write {len(matched_rows)} matched -> {input_csv}")
        print(f"         would write {len(mismatched_rows)} mismatched -> {mismatch_path}")
        return

    with open(input_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(matched_rows)
    print(f"Wrote {len(matched_rows)} matched rows to {input_csv}")

    with open(mismatch_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(mismatched_rows)
    print(f"Wrote {len(mismatched_rows)} mismatched rows to {mismatch_path}")


def main():
    p = argparse.ArgumentParser(
        description="Split train_features.csv into matched (keep in file) and mismatched (separate file)."
    )
    p.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to train_features.csv (will be overwritten with matched-only rows)",
    )
    p.add_argument(
        "--annotation-csv",
        type=Path,
        default=None,
        help="EmoMV DS1 annotation CSV (folder,col2=key). Rows in MISMATCH folder are mismatched.",
    )
    p.add_argument(
        "--label-col-a",
        default=None,
        help="First label column (use with --label-col-b to split by equality)",
    )
    p.add_argument(
        "--label-col-b",
        default=None,
        help="Second label column (use with --label-col-a to split by equality)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print counts and paths, do not write files",
    )
    args = p.parse_args()

    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    if args.annotation_csv is not None:
        if not args.annotation_csv.exists():
            raise SystemExit(f"Annotation CSV not found: {args.annotation_csv}")
        run_with_annotation(args.input_csv, args.annotation_csv, args.dry_run)
    elif args.label_col_a and args.label_col_b:
        run_with_label_columns(
            args.input_csv, args.label_col_a, args.label_col_b, args.dry_run
        )
    else:
        raise SystemExit(
            "Provide either --annotation-csv OR both --label-col-a and --label-col-b."
        )


if __name__ == "__main__":
    main()
