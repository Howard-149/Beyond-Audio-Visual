#!/usr/bin/env python3
"""Extract audio embeddings for EmoMV clips (MERT or Audio Flamingo 3).

Writes EmbeddingSchema v1 .pt files and CSV manifests consumed by
multimodal_classifier.py (--audio-root-dir / --*-audio-csv).
"""

import os
import csv
import torch
import torchaudio
from transformers import AutoProcessor, AutoModel, AudioFlamingo3ForConditionalGeneration
from tqdm import tqdm
from utils.embedding_schema import EmbeddingV1, now_iso8601, validate_embedding_schema_v1


def _load_model(model_name="m-a-p/MERT-v1-95M"):
    if model_name== "m-a-p/MERT-v1-95M":
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)
    elif model_name == "nvidia/audio-flamingo-3-hf":
        processor = AutoProcessor.from_pretrained(model_name)
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
    return processor, model, device


def extract_from_csv(csv_path, out_dir, data_root, subset, split_name, model_name="m-a-p/MERT-v1-95M", target_sr=24000):
    os.makedirs(out_dir, exist_ok=True)
    processor, model, device = _load_model(model_name)
    count = 0
    if hasattr(processor, "feature_extractor"):
        target_sr = processor.feature_extractor.sampling_rate
    else:
        # Fallback for models without explicit sampling_rate in feature_extractor (rare)
        target_sr = 24000 if "MERT" in model_name else 16000
    
    print(f"Model: {model_name} | Target SR: {target_sr}")
    skipped = 0
    manifest_rows = []  # standardized rows: key, feature_path, audio_label

    # Determine total rows for a tqdm progress bar (fast single-pass count)
    with open(csv_path, "r", newline="") as f:
        total_rows = sum(1 for _ in f)

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in tqdm(reader, total=total_rows, desc=f"Extracting {subset}/{split_name}", unit="rows"):
            if len(row) < 7:
                continue
            try:
                # music label column is at index 6 (0-based)
                class_id = int(row[6])
            except Exception:
                # skip rows with non-integer label or malformed rows
                continue
            audio_path = os.path.join(data_root, row[0], row[1] + ".mp4")
            if not os.path.exists(audio_path):
                print(f"⚠️ Missing file: {audio_path}")
                continue
            file_name = row[1]
            out_path = os.path.join(out_dir, file_name + ".pt")
            if os.path.exists(out_path):
                skipped += 1
                # still add to manifest for consistency
                manifest_rows.append({
                    "key": file_name,
                    "feature_path": out_path,
                    "audio_label": str(class_id)
                })
                continue

            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)

            if model_name == "nvidia/audio-flamingo-3-hf":
                inputs = processor.feature_extractor(
                        waveform.squeeze().numpy(), 
                        sampling_rate=target_sr, 
                        return_tensors="pt"
                    )
            else:
                inputs = processor(
                    waveform.squeeze(0), 
                    sampling_rate=target_sr, 
                    return_tensors="pt", 
                    padding=True
                )
            with torch.no_grad():
                if model_name == "m-a-p/MERT-v1-95M":
                    outputs = model(inputs["input_values"].to(device))
                    hidden_states = outputs.last_hidden_state  # (1, Time, FeatureDim)
                elif model_name == "nvidia/audio-flamingo-3-hf":
                    if "input_features" in inputs:
                        feature_key = "input_features"
                    elif "input_values" in inputs:
                        feature_key = "input_values" 
                    else:
                        raise ValueError(f"Unknown input keys: {inputs.keys()}")
                    if "attention_mask" in inputs:
                        mask = inputs["attention_mask"].to(device)
                    else:
                        print("⚠️ No attention mask found, manually compute it.")
                        B = features.shape[0]
                        T = features.shape[-1] 
                        # create mask assuming no padding (all ones)
                        mask = torch.ones((B, T), dtype=torch.long, device=device)
                    features = inputs[feature_key].to(device)
                    encoder_out = model.audio_tower(features, input_features_mask=mask )
                    hidden_states = encoder_out.last_hidden_state  # (1, T, H_audio)

                    

            try:
                source_rel = os.path.relpath(audio_path, data_root)
            except Exception:
                source_rel = os.path.basename(audio_path)

            # Build schema-compliant object (sequence embedding: (T, D))
            emb = hidden_states.squeeze(0).cpu() if hidden_states.dim() == 3 else hidden_states.cpu()
            if emb.dim() == 1:
                temporal_dim = None
                embedding_dim = int(emb.shape[0])
            else:
                temporal_dim = int(emb.shape[0])
                embedding_dim = int(emb.shape[1])

            schema = EmbeddingV1(
                embedding=emb,
                modality="audio",
                model_name=model_name,
                layer="last_hidden_state",
                pooling="none",
                embedding_dim=embedding_dim,
                temporal_dim=temporal_dim,
                schema_version="1.0",
                created_at=now_iso8601(),
                sample_rate=target_sr,
                fps=None,
                extra_metadata={
                    "label": int(class_id),
                    "filename": file_name,
                    "source_rel": source_rel,
                    "source_abs": audio_path,
                    "subset": subset,
                    "split": split_name,
                },
            ).to_dict()
            
            validate_embedding_schema_v1(schema)
            torch.save(schema, out_path)
            count += 1
            manifest_rows.append({
                "key": file_name,
                "feature_path": out_path,
                "audio_label": str(class_id)
            })

    # Summary print for quick feedback
    # Write standardized manifest CSV next to out_dir
    manifest_name = f"EmoMV_{subset}_{split_name}_audio_features.csv"
    manifest_path = os.path.join(out_dir, manifest_name)
    try:
        with open(manifest_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["key", "feature_path", "audio_label"])
            w.writeheader()
            for r in manifest_rows:
                w.writerow(r)
        print(f"Saved audio manifest: {manifest_path} ({len(manifest_rows)} rows)")
    except Exception as e:
        print(f"Failed to write manifest CSV: {e}")
    print(f"Extracted {count} files to {out_dir} (skipped {skipped} already-existing files)")
    return count


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Extract MERT features for EmoMV CSV")
    p.add_argument("csv", help="path to EmoMV CSV")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--subset", default="EmoMV")
    p.add_argument("--split", default="train")
    p.add_argument("--model_name", default="m-a-p/MERT-v1-95M")
    args = p.parse_args()

    n = extract_from_csv(args.csv, args.out_dir, args.data_root, args.subset, args.split, model_name=args.model_name)
    print(f"Extracted {n} files to {args.out_dir}")
