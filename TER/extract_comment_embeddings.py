"""
Extract text embeddings per video from YouTube comment JSONs (sequence format for downstream).

For each video we:
1. Load up to 50 comments from the JSON (comments[].text).
2. Concatenate into one long text and chunk by model max length (512 tokens).
3. Run the same text model as TER (twitter-roberta-base) on each chunk.
4. Mean-pool over tokens *within* each chunk -> (D,) per chunk; concat chunks -> (num_chunks, D) per video.
5. Save .pt files using EmbeddingV1 schema with pooling="none", temporal_dim=num_chunks (same as TER lyrics).
6. Write feature CSVs per split: key, feature_path, text_label for use with multimodal or TER pipelines.

Output format matches other modalities: (seq_len, hidden_size) so EmoMVMultimodalDataset and
TextEncoder (GRU/transformer/attn_pool) work without changes.

File layout:
- Input:  MER/outputs/comments/DS1_TRAIN_MATCH/*.json, DS1_VAL_MATCH/*.json, DS1_TEST_MATCH/*.json
- Output: output_root/DS1_TRAIN_MATCH/<key>.pt (same relative path structure as comments_root)
- CSVs:   output_csv_dir/train_features.csv, val_features.csv, test_features.csv
          (key, feature_path, text_label). Point multimodal_classifier --comments-root-dir at output_csv_dir.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# ER repo root for utils
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.embedding_schema import EmbeddingV1, now_iso8601, validate_embedding_schema_v1

# Same model as TER for consistency
MODEL_NAME = "cardiffnlp/twitter-roberta-base-2022-154m"
MAX_LENGTH = 512
EMOMV_CLASS_LABELS = ["exciting", "fear", "tense", "sad", "relax"]
LABEL_TO_IDX = {l: i for i, l in enumerate(EMOMV_CLASS_LABELS)}

_DATA_ROOT = (
    Path(os.environ["ER_DATA_ROOT"])
    if os.environ.get("ER_DATA_ROOT")
    else PROJECT_ROOT
)

# Default paths (comment embeddings live under data_root/TER/... on cluster)
DEFAULT_COMMENTS_ROOT = _DATA_ROOT / "MER" / "outputs" / "comments"
DEFAULT_EMBEDDINGS_DIR = _DATA_ROOT / "TER" / "comment_embeddings"
DEFAULT_CSV_DIR = _DATA_ROOT / "TER" / "comment_csvs"
SPLIT_DIR_MAP = {
    "train": "DS1_TRAIN_MATCH",
    "val": "DS1_VAL_MATCH",
    "test": "DS1_TEST_MATCH",
}


def load_comments_text(comments_path: Path, max_comments: int = 50) -> str:
    """Load a comment JSON and return a single string of up to max_comments comment texts."""
    try:
        data = json.loads(comments_path.read_text())
    except Exception:
        return ""
    comments = data.get("comments") or []
    texts = []
    for c in comments[:max_comments]:
        if isinstance(c, dict) and c.get("text"):
            texts.append(c["text"].strip())
        elif isinstance(c, str):
            texts.append(c.strip())
    return "\n".join(t for t in texts if t)


def tokenize_and_chunk(
    tokenizer, text: str, max_length: int = MAX_LENGTH
) -> List[torch.Tensor]:
    """Tokenize full text and split into chunks of at most max_length tokens. Returns list of input_ids tensors (each 1, L)."""
    if not text.strip():
        return []
    # Raw content tokens (no BOS/EOS so we add them per chunk)
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
    )
    if not tokens:
        return []
    bos, eos = tokenizer.bos_token_id or 0, tokenizer.eos_token_id or 2
    content_max = max_length - 2
    chunks = []
    for i in range(0, len(tokens), content_max):
        chunk_ids = [bos] + tokens[i : i + content_max] + [eos]
        chunks.append(torch.tensor([chunk_ids], dtype=torch.long))
    return chunks


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """(B, L, D) and (B, L) -> (B, D)."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_one_video(
    tokenizer,
    model: torch.nn.Module,
    text: str,
    device: torch.device,
    max_length: int = MAX_LENGTH,
) -> Optional[torch.Tensor]:
    """Return (num_chunks, D): mean-pool within each chunk, then concat chunks (no mean over chunks)."""
    chunks = tokenize_and_chunk(tokenizer, text, max_length)
    if not chunks:
        return None
    all_vectors = []
    for chunk_ids in chunks:
        chunk_ids = chunk_ids.to(device)
        pad_len = max_length - chunk_ids.size(1)
        if pad_len > 0:
            chunk_ids = torch.nn.functional.pad(
                chunk_ids,
                (0, pad_len),
                value=tokenizer.pad_token_id or 0,
            )
        attention_mask = (chunk_ids != (tokenizer.pad_token_id or 0)).long()
        with torch.no_grad():
            out = model(input_ids=chunk_ids, attention_mask=attention_mask)
        vec = mean_pool(out.last_hidden_state, attention_mask).squeeze(0)
        all_vectors.append(vec)
    # (num_chunks, D) -> keep as sequence for downstream (seq_len, hidden_size)
    stacked = torch.stack(all_vectors, dim=0).cpu()
    return stacked


def key_to_label(key: str) -> str:
    """Infer label from key prefix, e.g. sad_ORG_SAD(5) -> sad -> 3."""
    prefix = key.split("_")[0].lower() if key else ""
    return str(LABEL_TO_IDX.get(prefix, 0))


def discover_comment_files(comments_root: Path) -> Dict[str, List[Path]]:
    """Return {split: [path, ...]} for train/val/test."""
    out = {"train": [], "val": [], "test": []}
    for split, dir_name in SPLIT_DIR_MAP.items():
        split_dir = comments_root / dir_name
        if not split_dir.is_dir():
            continue
        for p in sorted(split_dir.glob("*.json")):
            out[split].append(p)
    return out


def save_embedding(
    output_path: Path,
    embedding: torch.Tensor,
    key: str,
    split: str,
) -> None:
    """Save (seq_len, D) tensor with EmbeddingV1 schema (pooling='none', temporal_dim=seq_len)."""
    seq_len = embedding.shape[0]
    embedding_dim = embedding.shape[1]
    obj = EmbeddingV1(
        embedding=embedding,
        modality="text",
        model_name=MODEL_NAME,
        layer="last_hidden_state",
        pooling="none",
        embedding_dim=embedding_dim,
        temporal_dim=seq_len,
        schema_version="1.0",
        created_at=now_iso8601(),
        extra_metadata={
            "key": key,
            "source": "comments",
            "split": split,
        },
    ).to_dict()
    validate_embedding_schema_v1(obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(output_path))


def run_split(
    comments_root: Path,
    output_root: Path,
    split: str,
    paths: List[Path],
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    limit: int,
) -> List[Dict[str, str]]:
    rows = []
    paths = paths[:limit] if limit > 0 else paths
    for p in tqdm(paths, desc=f"Comments {split}", unit="video"):
        key = p.stem
        text = load_comments_text(p, max_comments=50)
        emb = embed_one_video(tokenizer, model, text, device)
        if emb is None:
            continue
        # Preserve relative path under output_root (e.g. DS1_TRAIN_MATCH/<key>.pt)
        rel = p.relative_to(comments_root)
        out_path = output_root / rel.with_suffix(".pt")
        save_embedding(out_path, emb, key=key, split=split)
        label = key_to_label(key)
        rows.append({
            "key": key,
            "feature_path": str(out_path.resolve()),
            "text_label": label,
        })
    return rows


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract (seq_len, D) text embeddings per video from comment JSONs: mean-pool within chunk, concat chunks."
    )
    parser.add_argument(
        "--comments_root",
        type=Path,
        default=DEFAULT_COMMENTS_ROOT,
        help="Root containing DS1_TRAIN_MATCH, DS1_VAL_MATCH, DS1_TEST_MATCH",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help="Root for .pt files; keeps same relative path as comments_root (e.g. DS1_TRAIN_MATCH/<key>.pt)",
    )
    parser.add_argument(
        "--output_csv_dir",
        type=Path,
        default=DEFAULT_CSV_DIR,
        help="Directory to write train_features.csv, val_features.csv, test_features.csv",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max videos per split (0 = no limit)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    by_split = discover_comment_files(args.comments_root)
    args.output_csv_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching comment files from {args.comments_root}...")
    for split in args.splits:
        paths = by_split.get(split, [])
        if not paths:
            print(f"No comment files for split {split}, skipping.")
            continue
        rows = run_split(
            args.comments_root,
            args.output_root,
            split,
            paths,
            tokenizer,
            model,
            device,
            args.limit,
        )
        csv_path = args.output_csv_dir / f"{split}_features.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["key", "feature_path", "text_label"])
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {csv_path}")
    print(
        "Downstream: use --text-root-dir pointing to output_csv_dir "
        "(e.g. TER/comment_features or on cluster DATA_ROOT/TER/comment_features) "
        "so multimodal_classifier finds train_features.csv, val_features.csv, test_features.csv."
    )


if __name__ == "__main__":
    main()
