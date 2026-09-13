#!/usr/bin/env python3
"""Remap existing modality ``{split}_features.csv`` roots to a new partition.

``multimodal_classifier.py`` never reads EmoMV annotation CSVs. It only needs::

    --video-root-dir   DIR   # DIR/{train,val,test}_features.csv
    --audio-root-dir   DIR
    --lyrics-root-dir  DIR
    --comments-root-dir DIR

Each CSV has columns ``key,feature_path,<label>``. Samples are joined **by key**.
``.pt`` files can stay where they are; only the split membership CSVs must change.

This script takes ``partition_map.csv`` from ``make_song_disjoint_partition.py``
(or the shipped ``partitions/emomv_ds1_song_disjoint/partition_map.csv``)
and rewrites modality roots so the same keys land in the new train/val/test
CSVs (compatible with ``--*-root-dir``).

Example
-------
  # Lyrics (local) — prefer the shipped map
  python MER/remap_modality_roots_for_partition.py \\
    --partition-map partitions/emomv_ds1_song_disjoint/partition_map.csv \\
    --src-root TER/csvs \\
    --dst-root TER/csvs_song_disjoint \\
    --modality lyrics

  # On the cluster, remap each modality the classifier uses:
  python MER/remap_modality_roots_for_partition.py \\
    --partition-map partitions/emomv_ds1_song_disjoint/partition_map.csv \\
    --src-root $DATA/VER/EmoMV_mean_features_-1_10/Dataset1 \\
    --dst-root $DATA/VER/EmoMV_mean_features_-1_10/Dataset1_song_disjoint \\
    --modality video

  python multimodal_classifier.py --mode train \\
    --video-root-dir   .../Dataset1_song_disjoint \\
    --audio-root-dir   .../af3/Dataset1_song_disjoint \\
    --lyrics-root-dir  .../lyrics_csvs_song_disjoint \\
    --comments-root-dir .../comment_csvs_song_disjoint \\
    ...
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SPLITS = ("train", "val", "test")
LABEL_COLS = (
    "audio_label",
    "video_label",
    "lyrics_label",
    "comments_label",
    "text_label",
)


def load_partition_map(path: Path) -> Dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, str] = {}
    for r in rows:
        key = (r.get("key") or "").strip()
        sp = (r.get("new_split") or "").strip()
        if not key or sp not in SPLITS:
            continue
        out[key] = sp
    return out


def load_all_features(src_root: Path) -> Tuple[List[str], Dict[str, dict]]:
    """Merge train/val/test_features.csv into key -> row. Returns (fieldnames, by_key)."""
    by_key: Dict[str, dict] = {}
    fieldnames: Optional[List[str]] = None
    sources: Dict[str, str] = {}
    for split in SPLITS:
        path = src_root / f"{split}_features.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                key = (row.get("key") or "").strip()
                if not key:
                    continue
                row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                row["key"] = key
                if key in by_key and by_key[key].get("feature_path") != row.get("feature_path"):
                    raise SystemExit(
                        f"Duplicate key {key!r} with different feature_path in {src_root} "
                        f"({sources[key]} vs {split})"
                    )
                by_key[key] = row
                sources[key] = split
    if fieldnames is None:
        raise SystemExit(f"No {{train,val,test}}_features.csv under {src_root}")
    if "key" not in fieldnames or "feature_path" not in fieldnames:
        raise SystemExit(f"Expected key,feature_path in {src_root}; got {fieldnames}")
    return fieldnames, by_key


def pick_label_col(fieldnames: List[str], modality: str) -> Optional[str]:
    preferred = f"{modality}_label"
    if preferred in fieldnames:
        return preferred
    for c in LABEL_COLS:
        if c in fieldnames:
            return c
    return None


def remap(
    partition: Dict[str, str],
    fieldnames: List[str],
    by_key: Dict[str, dict],
    match_only: bool,
) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {s: [] for s in SPLITS}
    skipped_not_in_map = 0
    for key, row in by_key.items():
        if key not in partition:
            if match_only:
                skipped_not_in_map += 1
                continue
            # keep old split if column somehow present — else drop
            skipped_not_in_map += 1
            continue
        out[partition[key]].append(row)
    if skipped_not_in_map:
        print(f"Note: skipped {skipped_not_in_map} keys not in partition_map (e.g. MISMATCH-only)")
    for sp in SPLITS:
        out[sp].sort(key=lambda r: r["key"])
    return out


def write_root(
    dst_root: Path,
    fieldnames: List[str],
    split_rows: Dict[str, List[dict]],
) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for sp in SPLITS:
        path = dst_root / f"{sp}_features.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(split_rows[sp])
        print(f"  wrote {path} n={len(split_rows[sp])}")


def verify_paths(split_rows: Dict[str, List[dict]], check_exists: bool) -> None:
    if not check_exists:
        return
    missing = 0
    for sp, rows in split_rows.items():
        for r in rows:
            p = Path(r["feature_path"])
            if not p.is_file():
                missing += 1
                if missing <= 5:
                    print(f"  missing .pt [{sp}] {p}")
    print(f"  missing feature files: {missing}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition-map", type=Path, required=True)
    ap.add_argument("--src-root", type=Path, required=True, help="Old modality root with {{split}}_features.csv")
    ap.add_argument("--dst-root", type=Path, required=True, help="New root to write remapped CSVs")
    ap.add_argument(
        "--modality",
        choices=["audio", "video", "lyrics", "comments", "auto"],
        default="auto",
        help="Used only to prefer the right label column name when summarizing",
    )
    ap.add_argument(
        "--match-only",
        action="store_true",
        default=True,
        help="Only keep keys present in partition_map (MATCH clips). Default: True",
    )
    ap.add_argument("--no-match-only", action="store_false", dest="match_only")
    ap.add_argument("--check-exists", action="store_true", help="Warn if feature_path .pt is missing")
    args = ap.parse_args()

    partition = load_partition_map(args.partition_map)
    print(f"partition_map keys: {len(partition)}  counts={dict(Counter(partition.values()))}")

    fieldnames, by_key = load_all_features(args.src_root)
    print(f"src_root {args.src_root}: unique keys={len(by_key)} cols={fieldnames}")

    overlap = sum(1 for k in by_key if k in partition)
    print(f"overlap with partition_map: {overlap}/{len(by_key)}")

    split_rows = remap(partition, fieldnames, by_key, match_only=args.match_only)
    write_root(args.dst_root, fieldnames, split_rows)
    verify_paths(split_rows, args.check_exists)

    mod = args.modality
    label_col = pick_label_col(fieldnames, mod if mod != "auto" else "text")
    print("summary:")
    for sp in SPLITS:
        rows = split_rows[sp]
        print(f"  {sp}: {len(rows)}")
    print(f"Done. Point multimodal_classifier --*-root-dir at {args.dst_root}")


if __name__ == "__main__":
    main()
