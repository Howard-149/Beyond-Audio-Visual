#!/usr/bin/env python3
"""Extract lyrics (or other text) embeddings with Twitter-RoBERTa → EmbeddingSchema v1.

Writes .pt + CSV manifests for multimodal_classifier --lyrics-root-dir.
"""

import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.embedding_schema import EmbeddingV1, now_iso8601, validate_embedding_schema_v1, example_text_embedding

MODEL_NAME = "cardiffnlp/twitter-roberta-base-2022-154m"
TIMECODE_RE = re.compile(r"\[(\d{2}:\d{2}(?:\.\d{1,2})?)\]")


def strip_timecodes(text: str) -> str:
    if not text:
        return ""
    return TIMECODE_RE.sub("", text).replace("\n", " ").strip()


def load_text_csv(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def load_lyrics_text(lyrics_path: str) -> str:
    if not lyrics_path:
        return ""
    try:
        data = json.loads(Path(lyrics_path).read_text())
    except Exception:
        return ""
    text = data.get("lyrics", "")
    return strip_timecodes(text)


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def save_embedding(output_path: Path, embedding: torch.Tensor, extra_metadata: Dict[str, str]):
    obj = EmbeddingV1(
        embedding=embedding,
        modality="text",
        model_name=MODEL_NAME,
        layer="last_hidden_state",
        pooling="none",
        embedding_dim=int(embedding.shape[-1]),
        temporal_dim=embedding.shape[0],
        schema_version="1.0",
        created_at=now_iso8601(),
        extra_metadata=extra_metadata,
    ).to_dict()
    validate_embedding_schema_v1(obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(output_path))


def extract_embeddings(text_csv: Path, output_root: Path, output_csv: Path = None, batch_size: int = 8, limit: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    rows = load_text_csv(text_csv)
    texts = [load_lyrics_text(r.get("lyrics_path", "")) for r in rows]
    keys = [r.get("key", "") for r in rows]
    labels = [r.get("text_label", "") for r in rows]

    output_rows = []
    total_items = min(len(texts), limit) if limit > 0 else len(texts)
    for i in tqdm(range(0, total_items, batch_size), desc="Extracting embeddings", unit="batch"):
        batch_texts = texts[i : i + batch_size]
        batch_keys = keys[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        for k in enc:
            enc[k] = enc[k].to(device)

        with torch.no_grad():
            outputs = model(**enc)
            # pooled = mean_pool(outputs.last_hidden_state, enc["attention_mask"])  # (B, D)

        for j, key in enumerate(batch_keys):
            if not key:
                continue
            out_path = output_root / f"{key}.pt"
            save_embedding(
                out_path,
                outputs.last_hidden_state[j].detach().cpu(),
                extra_metadata={"key": key, "source_csv": str(text_csv)},
            )
            output_rows.append({
                "key": key,
                "feature_path": str(out_path),
                "text_label": batch_labels[j]
            })
    
    # Write feature paths to CSV
    if output_csv is None:
        split = text_csv.stem.replace("text_", "")
        output_csv = text_csv.parent / f"text_{split}_features.csv"
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "feature_path", "text_label"])
        writer.writeheader()
        writer.writerows(output_rows)
    
    print(f"Wrote {len(output_rows)} rows to {output_csv}")


def save_example_embedding(output_path: Path):
    obj = example_text_embedding()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(output_path))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract text embeddings from lyrics JSONs")
    parser.add_argument("--splits", nargs="+", default=["test"], help="Data splits to process (train/val/test)")
    parser.add_argument("--csv_dir", type=Path, default=Path("/path/to/ER/TER/csvs"), help="Directory containing text_{split}.csv files")
    parser.add_argument("--output_root", type=Path, default=Path("/path/to/ER/TER/EmoMV_lyrics_embeddings"), help="Output directory for .pt embeddings")
    parser.add_argument("--output_csv", type=Path, default=None, help="Output CSV with feature paths (auto-generated if not specified)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos to process in this run (0 = no limit)")
    args = parser.parse_args()

    for split in args.splits:
        text_csv = args.csv_dir / f"text_{split}.csv"
        extract_embeddings(text_csv, args.output_root, args.output_csv, args.batch_size, args.limit)


if __name__ == "__main__":
    main()
