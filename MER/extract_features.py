#!/usr/bin/env python3
"""Orchestrator: run EmoMV audio extraction for all splits listed in config.yaml.

Prefer calling `extract_EmoMV.py` directly for a single split. MTG-Jamendo support
was moved to `data_collection/mtg/` (not shipped).
"""

import yaml
import extract_EmoMV


def _load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = _load_config()
    dataset_type = config.get("dataset_type", "EmoMV")
    if dataset_type != "EmoMV":
        raise RuntimeError(
            f"Unsupported dataset_type={dataset_type!r}. "
            "Only EmoMV is supported here; MTG scripts live under data_collection/mtg/ (local)."
        )

    data_root = config.get("data_root")
    subset = config.get("subset")
    feature_dir_template = config.get(
        "feature_dir_template", "features/mert/{subset}/{split}"
    )
    model_name = config.get("model_name", "m-a-p/MERT-v1-95M")
    csv_template = config.get("csv_path_template")
    splits = config.get("splits") or [config.get("split", "train")]

    for split in splits:
        csv_path = csv_template.format(
            data_root=data_root,
            subset=subset,
            subset_num=subset[-1],
            split_upper=split.upper(),
        )
        out_dir = feature_dir_template.format(subset=subset, split=split)
        print(f"Extracting EmoMV split {split} from {csv_path} -> {out_dir}")
        extract_EmoMV.extract_from_csv(
            csv_path, out_dir, data_root, subset, split, model_name=model_name
        )


if __name__ == "__main__":
    main()
