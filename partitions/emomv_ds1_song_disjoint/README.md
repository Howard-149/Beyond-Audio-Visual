# EmoMV Dataset1 — song / video-disjoint partition

Shipped split used by the paper experiments (strategy B, `seed=42`).

| File | Role |
|------|------|
| `partition_map.csv` | MATCH clip `key` → `new_split` (+ song / video ids). Use with `MER/remap_modality_roots_for_partition.py`. |
| `DS1_{TRAIN,VAL,TEST}_MATCH_MISMATCH_labels.csv` | Same 7-column annotation schema as stock EmoMV; which file a row lives in **is** the new split. |
| `partition_summary.txt` | Counts + integrity checks |

**Do not regenerate** unless you have the AudD / lyrics index used to build song ids. Prefer these files as the canonical split.

See the root [README](../../README.md#0-song--video-disjoint-split-start-here) for how this plugs into extraction and `multimodal_classifier.py`.
