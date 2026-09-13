# Multimodal Emotion Recognition (EmoMV)

Code for multimodal emotion recognition on user-generated music videos (**EmoMV**), combining **audio**, **video (face)**, **lyrics**, and **comments**.

**Scope of this repo:** you prepare the data; we provide feature extractors, the unified embedding/CSV formats, the **song-disjoint split**, and the multimodal trainer.

```bash
python multimodal_classifier.py --mode train ...
```

This repository supports the SLT 2026 work *Beyond Audio-Visual: Unlocking the Value of Metadata for Emotion Recognition in User-Generated Content*.

## What this repo does / does not ship


| We ship                                                                                            | We do **not** ship                                         |
| -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Feature extractors (`MER/`, `VER/`, `TER/`)                                                        | Raw videos, lyrics dumps, comment scrapes                  |
| Format specs (`CSV_MANIFEST_SPECIFICATION.md`, `EMBEDDING_FORMAT.md`)                              | Download / scrape / `get_*` collectors                     |
| **Song-disjoint split** (`partitions/emomv_ds1_song_disjoint/`)                                    | Precomputed `.pt` features (gitignored)                    |
| Split helpers (`MER/make_song_disjoint_partition.py`, `MER/remap_modality_roots_for_partition.py`) | AudD / music-recognition index used to *rebuild* the split |
| `multimodal_classifier.py` + `models/`                                                             |                                                            |


Assume train/val/test media (and optional lyrics/comment text) are already on disk with labels. Then: **use the shipped split** → extract (or remap) manifests → train.

## Environments (two are enough)

`pip install -e .` only registers local packages for imports — it does **not** install third-party deps. Run it once inside each env.


| Env             | How to install                                                   | Use for                                                            |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| **VER** (conda) | `conda env create -f VER/environment.yml` → `conda activate VER` | Video extraction; **training / eval** (`multimodal_classifier.py`) |
| **MER** (pip)   | Python ≥3.10 + `pip install -r MER/requirements.txt`             | Audio extraction; lyrics/comments **text embedding** (`TER/`)      |


```bash
# --- Env 1: VER (video + train) ---
conda env create -f VER/environment.yml
conda activate VER
cd /path/to/ER && pip install -e .
python VER/extract_EmoMV.py ...
python multimodal_classifier.py --mode train ...

# --- Env 2: MER (audio + text embeddings) ---
conda create -n MER python=3.10 -y && conda activate MER
cd /path/to/ER
pip install -r MER/requirements.txt && pip install -e .
python MER/extract_EmoMV.py ...
python TER/text_embedding_extract.py ...
python TER/extract_comment_embeddings.py ...
```

Notes:

- **VER** includes `torch` / `pandas` / `wandb` plus the face stack (`dlib`, `emotiefflib`, `opencv`, …). Using it for training avoids a third env.
- **MER** is for HuggingFace audio/text (`torchaudio`, `transformers`). Face extraction will not work there.



## 0. Song / video-disjoint split (start here)

Recommended evaluation protocol for Dataset1: **no shared source video and no shared song** across train / val / test. Val and test are **one clip per song**.

This repo already ships the finished split (you do **not** need to re-run the partition builder):

```
partitions/emomv_ds1_song_disjoint/
  partition_map.csv                         # key → new_split (MATCH clips)
  DS1_{TRAIN,VAL,TEST}_MATCH_MISMATCH_labels.csv
  partition_summary.txt                     # counts + integrity checks
  README.md
```


| Split | MATCH clips | Songs | Notes                       |
| ----- | ----------- | ----- | --------------------------- |
| train | 2116        | 799   | multi-clip songs allowed    |
| val   | 263         | 263   | size-1 song components only |
| test  | 263         | 263   | size-1 song components only |


Built with `seed=42`, `test_frac=0.1`, `val_frac=0.1`. Checks: `video_cross_split=0`, `song_cross_split=0` (see `partition_summary.txt`).

How it plugs into the rest of the pipeline:


| Artifact                | Format                                                          | Used by                                                                                                                                                                       |
| ----------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Annotation CSVs         | Stock EmoMV **7-column, no header** labels                      | `MER/extract_EmoMV.py`, `VER/extract_EmoMV.py` (column 0 still names the **on-disk media folder**; only which of TRAIN/VAL/TEST files a row appears in defines the new split) |
| `partition_map.csv`     | `key,old_split,new_split,...`                                   | `MER/remap_modality_roots_for_partition.py`                                                                                                                                   |
| Remapped modality roots | `{train,val,test}_features.csv` with `key,feature_path,<label>` | `multimodal_classifier.py --*-root-dir`                                                                                                                                       |


Samples are always joined **by** `key`. `.pt` feature files can stay put; only the split membership CSVs need to change.

### Path A — you already extracted features under the original EmoMV split

Remap each modality root (writes new CSVs; does **not** copy `.pt` files):

```bash
MAP=partitions/emomv_ds1_song_disjoint/partition_map.csv

python MER/remap_modality_roots_for_partition.py \
  --partition-map "$MAP" \
  --src-root MER/EmoMV_features/mert/Dataset1 \
  --dst-root MER/EmoMV_features/mert/Dataset1_song_disjoint \
  --modality audio

python MER/remap_modality_roots_for_partition.py \
  --partition-map "$MAP" \
  --src-root VER/EmoMV_mean_features_-1_10/Dataset1 \
  --dst-root VER/EmoMV_mean_features_-1_10/Dataset1_song_disjoint \
  --modality video

python MER/remap_modality_roots_for_partition.py \
  --partition-map "$MAP" \
  --src-root TER/csvs \
  --dst-root TER/csvs_song_disjoint \
  --modality lyrics

python MER/remap_modality_roots_for_partition.py \
  --partition-map "$MAP" \
  --src-root TER/comment_features \
  --dst-root TER/comment_features_song_disjoint \
  --modality comments
```

Then point the trainer at the `*_song_disjoint` roots (same CSV schema as before):

```bash
python multimodal_classifier.py --mode train \
  --video-root-dir VER/EmoMV_mean_features_-1_10/Dataset1_song_disjoint \
  --audio-root-dir MER/EmoMV_features/mert/Dataset1_song_disjoint \
  --lyrics-root-dir TER/csvs_song_disjoint \
  --comments-root-dir TER/comment_features_song_disjoint \
  --fusion-type late_concat \
  --output-base outputs/multimodal_song_disjoint \
  --batch-size 16 --epochs 50 --allow-missing
```



### Path B — extracting from raw media with this split

1. Point extractors at the shipped annotation CSVs (or copy them next to your Dataset1 media tree).
2. Run `MER/extract_EmoMV.py` / `VER/extract_EmoMV.py` / `TER/*` so each modality writes `{train,val,test}_features.csv` under its feature root (see `[CSV_MANIFEST_SPECIFICATION.md](CSV_MANIFEST_SPECIFICATION.md)`).
3. Train with `--*-root-dir` as usual.

If you already extracted under the **original** stock split, prefer **Path A** (remap) instead of re-extracting.

### Rebuilding the partition (optional, not required)

`MER/make_song_disjoint_partition.py` can regenerate annotation CSVs + `partition_map.csv`, but it needs the AudD / lyrics **index** (not shipped). Prefer the files under `partitions/emomv_ds1_song_disjoint/`.

## Preparing your data

Bring your own media. Extractors and the trainer need consistent **keys** (clip ids) and labels. For Dataset1 experiments in this paper, use the **song-disjoint** membership above rather than the stock EmoMV train/val/test cut.

### Raw inputs (before extraction)


| Modality | Prepare                                                                                                     | Notes                                                                 |
| -------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Audio    | Per-clip audio (or extractable from video) + label                                                          | EmoMV-style annotation CSV + `--data_root` for `MER/extract_EmoMV.py` |
| Video    | Per-clip video files under a media root + labels                                                            | `VER/extract_EmoMV.py --input-base …` discovers train/val/test        |
| Lyrics   | Per-clip text (e.g. JSON/file) listed in `text_{split}.csv` with columns `key`, `lyrics_path`, `text_label` | Then `TER/text_embedding_extract.py`                                  |
| Comments | Per-clip JSON with `comments[].text` + labels                                                               | Then `TER/extract_comment_embeddings.py`                              |


EmoMV emotion labels (5-way): `exciting`, `fear`, `tense`, `sad`, `relax` (indices 0–4).

### After extraction — CSV manifests the trainer reads


| Modality          | Columns                                                              |
| ----------------- | -------------------------------------------------------------------- |
| Audio             | `key,feature_path,audio_label`                                       |
| Video             | `key,feature_path,video_label`                                       |
| Lyrics / comments | `key,feature_path,text_label` (or `lyrics_label` / `comments_label`) |


Each `--*-root-dir` is a directory containing `train_features.csv`, `val_features.csv`, and `test_features.csv` in that schema. `.pt` files use **EmbeddingSchema v1** (`utils/embedding_schema.py`). Full details:

- `[CSV_MANIFEST_SPECIFICATION.md](CSV_MANIFEST_SPECIFICATION.md)`
- `[CSV_HEADER_SPEC.py](CSV_HEADER_SPEC.py)`
- `[EMBEDDING_FORMAT.md](EMBEDDING_FORMAT.md)`


| Modality          | Extractor                                                            | Env | Example feature root                      |
| ----------------- | -------------------------------------------------------------------- | --- | ----------------------------------------- |
| Video             | `VER/extract_EmoMV.py`                                               | VER | `VER/EmoMV_mean_features_-1_10/Dataset1/` |
| Audio             | `MER/extract_EmoMV.py`                                               | MER | `MER/EmoMV_features/mert/Dataset1/`       |
| Lyrics / comments | `TER/text_embedding_extract.py`, `TER/extract_comment_embeddings.py` | MER | `TER/csvs/`, `TER/comment_features/`      |




## Quick start (classifier)

Features already extracted **and remapped** to song-disjoint (Path A); activate **VER**:

```bash
conda activate VER
pip install -e .   # once per env

python multimodal_classifier.py --mode train \
  --video-root-dir VER/EmoMV_features_-1_10/Dataset1_song_disjoint \
  --audio-root-dir MER/EmoMV_features/mert/Dataset1_song_disjoint \
  --lyrics-root-dir TER/csvs_song_disjoint \
  --comments-root-dir TER/comment_features_song_disjoint \
  --fusion-type late_concat \
  --output-base outputs/multimodal_avt \
  --batch-size 16 --epochs 50 --allow-missing
```

Useful flags:

- `--fusion-type` — `late_concat`, `cross_attention_mean`, `video_query_cross_attention`, or unimodal `audio_only` / `video_only` / `lyrics_only` / `comments_only`
- `--no-video` / `--no-audio` / … — ablations
- `--cross-attn-q/k/v` — query / key / value modalities
- `--use-wandb` — Weights & Biases

Audio-only baseline:

```bash
python multimodal_classifier.py --mode train \
  --fusion-type audio_only \
  --audio-root-dir MER/EmoMV_features/mert/Dataset1_song_disjoint \
  --video-root-dir VER/EmoMV_features_-1_10/Dataset1_song_disjoint \
  --allow-missing --output-base outputs/audio_only \
  --batch-size 16 --epochs 50
```

More examples and flags: see the Quick start section above.

```bash
python test_fusion_api.py   # fusion unit tests, no data needed
```



## Repository layout

```
multimodal_classifier.py              # ★ main train / eval entry
partitions/emomv_ds1_song_disjoint/   # ★ shipped song-disjoint split + map
MER/make_song_disjoint_partition.py   # rebuild split (needs AudD index; optional)
MER/remap_modality_roots_for_partition.py  # remap {split}_features.csv by partition_map
models/                               # encoders + fusion
data/multimodal.py                    # join modalities by key
utils/embedding_schema.py             # EmbeddingSchema v1
VER/                                  # video extraction (+ environment.yml)
MER/                                  # audio extraction (+ requirements.txt)
TER/                                  # lyrics / comment embedding extraction
CSV_MANIFEST_SPECIFICATION.md
EMBEDDING_FORMAT.md
```



## Pipeline (end-to-end)

1. **Split** — use `partitions/emomv_ds1_song_disjoint/` (do not invent a new cut for paper-comparable numbers).
2. **Audio** (MER env) — `MER/extract_EmoMV.py`, **or** remap an existing audio root (Path A).
3. **Video** (VER env) — `VER/extract_EmoMV.py`, **or** remap.
4. **Text** (MER env) — `TER/text_embedding_extract.py` / `TER/extract_comment_embeddings.py`, **or** remap.
5. **Train** (VER env) — `multimodal_classifier.py` with `--*-root-dir` pointing at song-disjoint roots.



## Citation

If you use this code, please cite the associated SLT 2026 paper:

*Beyond Audio-Visual: Unlocking the Value of Metadata for Emotion Recognition in User-Generated Content.*