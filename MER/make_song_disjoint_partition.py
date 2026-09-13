#!/usr/bin/env python3
"""Build a song/video-disjoint EmoMV Dataset1 partition (strategy B).

Constraints
-----------
- Clips that share a source ``video_id`` (filename stem before ``-k-of-m``)
  stay in the same split.
- Clips that share a song id ``(artist.lower(), title.lower())`` from AudD /
  lyrics JSON stay in the same split.
- **test** (and **val**) contain only size-1 song components: one clip per
  song id, and that song does not appear in any other split.
- Annotation CSV format is unchanged (7 columns, no header), so
  ``MER/extract_EmoMV.py`` / ``VER/extract_EmoMV.py`` keep working.
  Column 0 still points at the **on-disk** folder (old split name); only
  which of TRAIN/VAL/TEST CSVs a row appears in defines the new split.

Outputs (default under ``EmoMV/DS1_EmoMV_A/annotation_song_disjoint/``)
----------------------------------------------------------------------
- ``DS1_{TRAIN,VAL,TEST}_MATCH_MISMATCH_labels.csv`` — same schema as stock
- ``partition_map.csv`` — per MATCH clip: key, old/new split, song, video_id
- ``partition_summary.txt`` — counts and integrity checks

The paper split is **already shipped** at
``partitions/emomv_ds1_song_disjoint/`` (prefer that over regenerating).
To point existing ``{split}_features.csv`` roots at this partition without
re-extracting ``.pt`` files, use ``MER/remap_modality_roots_for_partition.py``.

Example
-------
  python MER/make_song_disjoint_partition.py \\
    --annotation-dir EmoMV/DS1_EmoMV_A/annotation \\
    --index-csv MER/outputs/music_recognition/index.csv \\
    --out-dir EmoMV/DS1_EmoMV_A/annotation_song_disjoint \\
    --seed 42 --test-frac 0.10 --val-frac 0.10
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MOODS = ["exciting", "fear", "tense", "sad", "relax"]
MOOD_TO_ID = {m: i for i, m in enumerate(["exciting", "fear", "tense", "sad", "relax"])}
CLIP_PAT = re.compile(r"^(.+?\(\d+\))-\d+-of-\d+$")
SPLIT_DIRS = {
    "train": "DS1_TRAIN_MATCH",
    "val": "DS1_VAL_MATCH",
    "test": "DS1_TEST_MATCH",
}


def video_id_from_stem(stem: str) -> str:
    m = CLIP_PAT.match(stem)
    return m.group(1) if m else stem


def mood_from_stem(stem: str) -> str:
    for m in MOODS:
        if stem.startswith(m):
            return m
    return "unknown"


def load_song_lookup(index_csv: Path, repo_root: Path) -> Dict[str, Tuple[Optional[str], Optional[str], bool]]:
    """stem -> (artist, title, lyrics_ok)."""
    out: Dict[str, Tuple[Optional[str], Optional[str], bool]] = {}
    with index_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row["rel_path"]
            stem = Path(rel).stem
            artist = title = None
            lyrics_ok = False
            lp = (row.get("lyrics_path") or "").strip()
            if lp.startswith("outputs/"):
                p = repo_root / "MER" / lp if not (repo_root / lp).exists() else repo_root / lp
                # index paths are relative to MER/
                candidates = [repo_root / "MER" / lp, repo_root / lp]
                for cand in candidates:
                    if cand.exists():
                        d = json.loads(cand.read_text(encoding="utf-8"))
                        artist, title = d.get("artist"), d.get("title")
                        lyrics_ok = d.get("status") == "success"
                        break
            if title is None:
                rj = (row.get("result_json") or "").strip()
                if rj:
                    candidates = [repo_root / "MER" / rj, repo_root / rj]
                    for cand in candidates:
                        if cand.exists():
                            d = json.loads(cand.read_text(encoding="utf-8"))
                            artist = d.get("artist") or (d.get("result") or {}).get("artist")
                            title = d.get("title") or (d.get("result") or {}).get("title")
                            break
            out[stem] = (artist, title, lyrics_ok)
    return out


def song_key(artist: Optional[str], title: Optional[str]) -> Optional[Tuple[str, str]]:
    if not title:
        return None
    return ((artist or "").strip().lower(), title.strip().lower())


def load_annotation_rows(annotation_dir: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    for name in (
        "DS1_TRAIN_MATCH_MISMATCH_labels.csv",
        "DS1_VAL_MATCH_MISMATCH_labels.csv",
        "DS1_TEST_MATCH_MISMATCH_labels.csv",
    ):
        path = annotation_dir / name
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 7:
                    rows.append(row)
    return rows


def is_mismatch(folder: str) -> bool:
    return "MISMATCH" in folder.upper()


def old_split_from_folder(folder: str) -> str:
    u = folder.upper()
    if "TRAIN" in u:
        return "train"
    if "VAL" in u:
        return "val"
    if "TEST" in u:
        return "test"
    raise ValueError(f"Cannot parse split from folder: {folder}")


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def assign_match_components(
    comps: List[dict],
    test_frac: float,
    val_frac: float,
    seed: int,
) -> Dict[int, str]:
    """Mood-stratified assignment; test/val only get size-1 song components."""
    rng = random.Random(seed)
    cmap: Dict[int, str] = {}
    by_mood: Dict[str, List[dict]] = defaultdict(list)
    for c in comps:
        by_mood[c["primary"]].append(c)

    for mood, clist in by_mood.items():
        clist = clist[:]
        rng.shuffle(clist)
        # Prefer singleton song comps for test/val.
        clist.sort(
            key=lambda c: (
                0 if (c["has_song"] and c["n"] == 1) else 1 if c["has_song"] else 2,
                c["n"],
            )
        )
        mood_n = sum(c["n"] for c in clist)
        t_bud = int(round(test_frac * mood_n))
        v_bud = int(round(val_frac * mood_n))
        t = v = 0

        for c in clist:
            if c["id"] in cmap:
                continue
            if c["has_song"] and c["n"] == 1 and (
                t + c["n"] <= t_bud or (t < t_bud and t + c["n"] <= t_bud + 2)
            ):
                cmap[c["id"]] = "test"
                t += c["n"]

        remain = [c for c in clist if c["id"] not in cmap]
        remain.sort(
            key=lambda c: (
                0 if (c["has_song"] and c["n"] == 1) else 1 if c["has_song"] else 2,
                c["n"],
            )
        )
        for c in remain:
            if c["has_song"] and c["n"] == 1 and (
                v + c["n"] <= v_bud or (v < v_bud and v + c["n"] <= v_bud + 2)
            ):
                cmap[c["id"]] = "val"
                v += c["n"]

        for c in clist:
            if c["id"] not in cmap:
                cmap[c["id"]] = "train"

    for c in comps:
        if c["id"] not in cmap:
            cmap[c["id"]] = "train"
    return cmap


def assign_mismatch_rows(
    mm_rows: List[List[str]],
    video_id_to_split: Dict[str, str],
    test_frac: float,
    val_frac: float,
    seed: int,
) -> List[Tuple[List[str], str]]:
    """Place MISMATCH rows: follow linked video_id split if known, else mood-strat fill.

    Never put an unlinked MISMATCH into test (keeps test song-unique / clean).
    """
    rng = random.Random(seed + 7)
    placed: List[Tuple[List[str], str]] = []
    unlinked: List[List[str]] = []

    for row in mm_rows:
        key = row[1]
        if "_PAIR_" not in key:
            unlinked.append(row)
            continue
        left, right = key.split("_PAIR_", 1)
        left_vid, right_vid = video_id_from_stem(left), video_id_from_stem(right)
        splits = set()
        if left_vid in video_id_to_split:
            splits.add(video_id_to_split[left_vid])
        if right_vid in video_id_to_split:
            splits.add(video_id_to_split[right_vid])
        if len(splits) == 1:
            placed.append((row, next(iter(splits))))
        elif len(splits) > 1:
            # Conflicting sides → train (never test)
            placed.append((row, "train"))
        else:
            unlinked.append(row)

    # Fill unlinked into train/val only, mood-stratified on video label (col3)
    by_mood: Dict[str, List[List[str]]] = defaultdict(list)
    for row in unlinked:
        by_mood[row[3] if row[3] in MOOD_TO_ID else "exciting"].append(row)

    for mood, rows in by_mood.items():
        rows = rows[:]
        rng.shuffle(rows)
        n = len(rows)
        v_bud = int(round(val_frac * n))
        # no test for unlinked
        for i, row in enumerate(rows):
            if i < v_bud:
                placed.append((row, "val"))
            else:
                placed.append((row, "train"))

    return placed


def verify(
    match_items: List[dict],
    assign: Dict[int, str],
) -> List[str]:
    msgs: List[str] = []
    # video isolation
    vid_sp: Dict[str, set] = defaultdict(set)
    song_sp: Dict[Tuple[str, str], set] = defaultdict(set)
    for it in match_items:
        sp = assign[it["idx"]]
        vid_sp[it["video_id"]].add(sp)
        if it["song"]:
            song_sp[it["song"]].add(sp)

    cross_v = sum(1 for s in vid_sp.values() if len(s) > 1)
    cross_s = sum(1 for s in song_sp.values() if len(s) > 1)
    if cross_v:
        msgs.append(f"FAIL video_cross_split={cross_v}")
    else:
        msgs.append("OK video_cross_split=0")
    if cross_s:
        msgs.append(f"FAIL song_cross_split={cross_s}")
    else:
        msgs.append("OK song_cross_split=0")

    for split in ("test", "val"):
        idxs = [it for it in match_items if assign[it["idx"]] == split]
        songs = [it["song"] for it in idxs if it["song"]]
        if len(songs) != len(idxs):
            msgs.append(f"FAIL {split}: {len(idxs) - len(songs)} clips lack song id")
        if len(songs) != len(set(songs)):
            msgs.append(
                f"FAIL {split}: duplicate songs within split "
                f"({len(songs) - len(set(songs))} extras)"
            )
        else:
            msgs.append(f"OK {split}: {len(idxs)} clips, {len(set(songs))} unique songs")
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("EmoMV/DS1_EmoMV_A/annotation"),
        help="Directory with stock DS1_*_MATCH_MISMATCH_labels.csv",
    )
    ap.add_argument(
        "--index-csv",
        type=Path,
        default=Path("MER/outputs/music_recognition/index.csv"),
        help="Music-recognition index (for artist/title)",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root (default: parent of MER/ if script lives in MER/)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("EmoMV/DS1_EmoMV_A/annotation_song_disjoint"),
        help="Output annotation directory (does not overwrite stock CSVs)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.10)
    args = ap.parse_args()

    repo_root = args.repo_root
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    annotation_dir = args.annotation_dir if args.annotation_dir.is_absolute() else repo_root / args.annotation_dir
    index_csv = args.index_csv if args.index_csv.is_absolute() else repo_root / args.index_csv
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    song_lookup = load_song_lookup(index_csv, repo_root)
    all_rows = load_annotation_rows(annotation_dir)

    match_rows = [r for r in all_rows if not is_mismatch(r[0])]
    mm_rows = [r for r in all_rows if is_mismatch(r[0])]

    match_items: List[dict] = []
    for i, row in enumerate(match_rows):
        stem = row[1]
        artist, title, lyrics_ok = song_lookup.get(stem, (None, None, False))
        sk = song_key(artist, title)
        match_items.append(
            {
                "idx": i,
                "row": row,
                "stem": stem,
                "video_id": video_id_from_stem(stem),
                "mood": mood_from_stem(stem),
                "song": sk,
                "artist": artist,
                "title": title,
                "lyrics_ok": lyrics_ok,
                "old_split": old_split_from_folder(row[0]),
            }
        )

    n = len(match_items)
    uf = UnionFind(n)
    by_vid: Dict[str, List[int]] = defaultdict(list)
    by_song: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for it in match_items:
        by_vid[it["video_id"]].append(it["idx"])
        if it["song"]:
            by_song[it["song"]].append(it["idx"])
    for idxs in list(by_vid.values()) + list(by_song.values()):
        for j in idxs[1:]:
            uf.union(idxs[0], j)

    buckets: Dict[int, List[int]] = defaultdict(list)
    for it in match_items:
        buckets[uf.find(it["idx"])].append(it["idx"])

    comps: List[dict] = []
    for cid, idxs in buckets.items():
        items = [match_items[j] for j in idxs]
        moods = Counter(it["mood"] for it in items)
        songs = {it["song"] for it in items if it["song"]}
        comps.append(
            {
                "id": cid,
                "n": len(idxs),
                "idxs": idxs,
                "primary": moods.most_common(1)[0][0],
                "has_song": bool(songs),
                "songs": songs,
            }
        )

    cmap = assign_match_components(comps, args.test_frac, args.val_frac, args.seed)
    assign = {i: cmap[uf.find(i)] for i in range(n)}

    # video_id -> split (for MISMATCH linking)
    video_id_to_split: Dict[str, str] = {}
    for it in match_items:
        video_id_to_split[it["video_id"]] = assign[it["idx"]]

    checks = verify(match_items, assign)
    for line in checks:
        print(line)
    if any(line.startswith("FAIL") for line in checks):
        raise SystemExit("Integrity checks failed; refusing to write CSVs.")

    mm_placed = assign_mismatch_rows(
        mm_rows, video_id_to_split, args.test_frac, args.val_frac, args.seed
    )

    out_rows: Dict[str, List[List[str]]] = {"train": [], "val": [], "test": []}
    for it in match_items:
        out_rows[assign[it["idx"]]].append(it["row"])
    for row, sp in mm_placed:
        out_rows[sp].append(row)

    # Stable order: MATCH then MISMATCH, mood order within
    def sort_key(row: List[str]):
        return (1 if is_mismatch(row[0]) else 0, row[3], row[1])

    for sp in ("train", "val", "test"):
        out_rows[sp].sort(key=sort_key)
        out_name = f"DS1_{sp.upper()}_MATCH_MISMATCH_labels.csv"
        out_path = out_dir / out_name
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerows(out_rows[sp])
        n_match = sum(1 for r in out_rows[sp] if not is_mismatch(r[0]))
        n_mm = len(out_rows[sp]) - n_match
        print(f"Wrote {out_path}  total={len(out_rows[sp])} match={n_match} mismatch={n_mm}")

    # partition map (MATCH only)
    map_path = out_dir / "partition_map.csv"
    with map_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "old_split",
                "new_split",
                "video_id",
                "artist",
                "title",
                "song_id",
                "mood",
                "lyrics_ok",
                "folder",
            ],
        )
        w.writeheader()
        for it in sorted(match_items, key=lambda x: (assign[x["idx"]], x["stem"])):
            sk = it["song"]
            w.writerow(
                {
                    "key": it["stem"],
                    "old_split": it["old_split"],
                    "new_split": assign[it["idx"]],
                    "video_id": it["video_id"],
                    "artist": it["artist"] or "",
                    "title": it["title"] or "",
                    "song_id": f"{sk[0]}::{sk[1]}" if sk else "",
                    "mood": it["mood"],
                    "lyrics_ok": int(it["lyrics_ok"]),
                    "folder": it["row"][0],
                }
            )
    print(f"Wrote {map_path}")

    # summary
    summary_lines = []
    summary_lines.append("Song/video-disjoint partition (strategy B)")
    summary_lines.append(f"seed={args.seed} test_frac={args.test_frac} val_frac={args.val_frac}")
    summary_lines.append(f"components={len(comps)} match_clips={n}")
    for sp in ("train", "val", "test"):
        items = [it for it in match_items if assign[it["idx"]] == sp]
        moods = Counter(it["mood"] for it in items)
        songs = {it["song"] for it in items if it["song"]}
        lyr = sum(1 for it in items if it["lyrics_ok"])
        summary_lines.append(
            f"{sp}: match={len(items)} songs={len(songs)} lyrics_ok={lyr} mood={dict(moods)}"
        )
    summary_lines.append("--- checks ---")
    summary_lines.extend(checks)
    mm_counts = Counter(sp for _, sp in mm_placed)
    summary_lines.append(f"mismatch placement: {dict(mm_counts)}")
    summary_path = out_dir / "partition_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
