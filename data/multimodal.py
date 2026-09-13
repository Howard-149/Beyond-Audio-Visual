"""Multimodal (audio + video + lyrics + comments) dataset for EmoMV.

Loads paired audio and video features by matching keys from separate CSV manifests.
Lyrics and comments are two separate optional text modalities.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset


def _label_col_for_modality(raw_df: Optional[pd.DataFrame], modality: str) -> Optional[str]:
    """Return label column name to use: modality_label or text_label (backward compat)."""
    if raw_df is None:
        return None
    pref = f"{modality}_label"
    if pref in raw_df.columns:
        return pref
    if "text_label" in raw_df.columns:
        return "text_label"
    return None


class EmoMVMultimodalDataset(Dataset):
    """Multimodal dataset that joins audio, video, lyrics, and comments by matching keys.

    Expected CSV columns:
    - Video CSV: key, feature_path, video_label
    - Audio CSV: key, feature_path, audio_label
    - Lyrics CSV (optional): key, feature_path, lyrics_label or text_label
    - Comments CSV (optional): key, feature_path, comments_label or text_label

    Features:
    - Video: .pt files (schema v1) -> (T_v, D_v) or (T_v, E, D_v)
    - Audio: .pt files (schema v1) -> (T_a, D_a)
    - Lyrics: .pt files (schema v1) -> (T_l, D_l)
    - Comments: .pt files (schema v1) -> (T_c, D_c)
    """

    def __init__(
        self,
        video_csv: Optional[str],
        audio_csv: Optional[str],
        lyrics_csv: Optional[str] = None,
        comments_csv: Optional[str] = None,
        labels: Optional[List[str]] = None,
        strict_label_match: bool = True,
        allow_missing_modalities: bool = False,
    ):
        vdf = pd.read_csv(video_csv, dtype=str) if video_csv else None
        adf = pd.read_csv(audio_csv, dtype=str) if audio_csv else None
        ldf = pd.read_csv(lyrics_csv, dtype=str) if lyrics_csv else None
        cdf = pd.read_csv(comments_csv, dtype=str) if comments_csv else None

        self.video_feature_shape = self._infer_feature_shape_from_df(vdf, "feature_path")
        self.audio_feature_shape = self._infer_feature_shape_from_df(adf, "feature_path")
        self.lyrics_feature_shape = self._infer_feature_shape_from_df(ldf, "feature_path")
        self.comments_feature_shape = self._infer_feature_shape_from_df(cdf, "feature_path")
        self.allow_missing_modalities = allow_missing_modalities

        for name, df in [("video", vdf), ("audio", adf)]:
            if df is None:
                continue
            if "key" not in df.columns or "feature_path" not in df.columns:
                raise RuntimeError(
                    f"{name} CSV must contain columns ['key', 'feature_path']. Found: {df.columns.tolist()}"
                )
        for name, df in [("lyrics", ldf), ("comments", cdf)]:
            if df is None:
                continue
            if "key" not in df.columns or "feature_path" not in df.columns:
                raise RuntimeError(
                    f"{name} CSV must contain columns ['key', 'feature_path']. Found: {df.columns.tolist()}"
                )

        if vdf is not None:
            vdf["key"] = vdf["key"].str.strip()
            vdf["feature_path"] = vdf["feature_path"].str.strip()
        if adf is not None:
            adf["key"] = adf["key"].str.strip()
            adf["feature_path"] = adf["feature_path"].str.strip()
        if ldf is not None:
            ldf["key"] = ldf["key"].str.strip()
            ldf["feature_path"] = ldf["feature_path"].str.strip()
        if cdf is not None:
            cdf["key"] = cdf["key"].str.strip()
            cdf["feature_path"] = cdf["feature_path"].str.strip()

        for col in ["video_label", "audio_label"]:
            if vdf is not None and col in vdf.columns:
                vdf[col] = vdf[col].str.strip()
            if adf is not None and col in adf.columns:
                adf[col] = adf[col].str.strip()
        for df, mod in [(ldf, "lyrics"), (cdf, "comments")]:
            if df is None:
                continue
            for label_col in [f"{mod}_label", "text_label"]:
                if label_col in df.columns:
                    df[label_col] = df[label_col].str.strip()
                    break

        join_how = "outer" if allow_missing_modalities else "inner"
        df = None
        for modality, raw_df in [("video", vdf), ("audio", adf), ("lyrics", ldf), ("comments", cdf)]:
            if raw_df is None:
                continue
            rename_map = {"feature_path": f"feature_path_{modality}"}
            label_col = _label_col_for_modality(raw_df, modality)
            if label_col is not None:
                rename_map[label_col] = f"{modality}_label"
            cols = ["key", "feature_path"] + ([label_col] if label_col else [])
            mod_df = raw_df[[c for c in cols if c in raw_df.columns]].copy()
            mod_df.rename(columns=rename_map, inplace=True)
            if df is None:
                df = mod_df
            else:
                df = pd.merge(df, mod_df, on="key", how=join_how)
        if df is None or df.empty:
            raise RuntimeError("No valid samples found across provided CSV manifests")

        if labels is None:
            labels = ["exciting", "fear", "tense", "sad", "relax"]
        self.labels = labels
        self.label2idx = {l: i for i, l in enumerate(labels)}
        self.has_lyrics = ldf is not None
        self.has_comments = cdf is not None

        samples: List[Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path], int]] = []
        video_required = vdf is not None
        audio_required = adf is not None
        lyrics_required = ldf is not None and not allow_missing_modalities
        comments_required = cdf is not None and not allow_missing_modalities

        for _, row in df.iterrows():
            key = row["key"]
            v_path = Path(row["feature_path_video"]) if pd.notna(row.get("feature_path_video")) else None
            a_path = Path(row["feature_path_audio"]) if pd.notna(row.get("feature_path_audio")) else None
            l_path = Path(row["feature_path_lyrics"]) if pd.notna(row.get("feature_path_lyrics")) else None
            c_path = Path(row["feature_path_comments"]) if pd.notna(row.get("feature_path_comments")) else None

            if v_path and not v_path.exists():
                print(f"Warning: missing video feature {v_path}, dropping video modality for key={key}")
                v_path = None
            if a_path and not a_path.exists():
                print(f"Warning: missing audio feature {a_path}, dropping audio modality for key={key}")
                a_path = None
            if l_path and not l_path.exists():
                print(f"Warning: missing lyrics feature {l_path}, dropping lyrics modality for key={key}")
                l_path = None
            if c_path and not c_path.exists():
                print(f"Warning: missing comments feature {c_path}, dropping comments modality for key={key}")
                c_path = None

            if v_path is None and a_path is None and l_path is None and c_path is None:
                continue

            if not allow_missing_modalities:
                if (video_required and v_path is None) or (audio_required and a_path is None):
                    continue
                if lyrics_required and l_path is None:
                    continue
                if comments_required and c_path is None:
                    continue

            labels_available = []
            for label_key in ["video_label", "audio_label", "lyrics_label", "comments_label"]:
                if label_key in row and pd.notna(row[label_key]):
                    labels_available.append(str(row[label_key]).strip())
            if not labels_available:
                print(f"Warning: missing labels for key={key}, skipping")
                continue
            if strict_label_match and len(set(labels_available)) > 1:
                continue
            lab_str = labels_available[0].lower()

            try:
                if lab_str.isdigit():
                    lab_idx = int(lab_str)
                    if not (0 <= lab_idx < len(self.labels)):
                        raise ValueError(f"Numeric label {lab_idx} out of range [0, {len(self.labels)-1}]")
                elif lab_str in self.label2idx:
                    lab_idx = self.label2idx[lab_str]
                else:
                    raise ValueError(f"Unknown label '{lab_str}'")
            except ValueError as e:
                print(f"Warning: {e} for key={key}, skipping")
                continue

            samples.append((v_path, a_path, l_path, c_path, lab_idx))

        if not samples:
            raise RuntimeError("No valid multimodal samples found after union join")

        self.samples = samples
        self.video_feature_shape = self.video_feature_shape or self._infer_feature_shape_from_samples("video")
        self.audio_feature_shape = self.audio_feature_shape or self._infer_feature_shape_from_samples("audio")
        self.lyrics_feature_shape = self.lyrics_feature_shape or self._infer_feature_shape_from_samples("lyrics")
        self.comments_feature_shape = self.comments_feature_shape or self._infer_feature_shape_from_samples("comments")
        print(f"Loaded {len(self.samples)} multimodal samples from CSVs")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        Optional[Path], Optional[Path], Optional[Path], Optional[Path],
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
        int, Optional[List[int]],
    ]:
        """Return (video_path, audio_path, lyrics_path, comments_path, video_emb, audio_emb, lyrics_emb, comments_emb, label_idx, video_indices)."""
        v_path, a_path, l_path, c_path, y = self.samples[idx]
        video_emb, v_indices = self._load_video_features(v_path)
        audio_emb = self._load_audio_features(a_path)
        lyrics_emb = self._load_sequence_features(l_path, self.lyrics_feature_shape)
        comments_emb = self._load_sequence_features(c_path, self.comments_feature_shape)
        return v_path, a_path, l_path, c_path, video_emb, audio_emb, lyrics_emb, comments_emb, y, v_indices

    def _load_video_features(self, path: Optional[Path]) -> Tuple[torch.Tensor, Optional[List[int]]]:
        v_indices = None
        if path is None:
            return self._zero_sequence(self.video_feature_shape), None
        video_data = torch.load(str(path), map_location="cpu")
        video_emb = video_data["embedding"]
        metadata = video_data.get("extra_metadata", {})
        v_indices = metadata.get("indices")
        if isinstance(v_indices, str):
            v_indices = [int(i) for i in v_indices.strip().split(",")]
        if video_emb.dim() == 3 and video_emb.shape[1] == 1:
            video_emb = video_emb.squeeze(1)
        return video_emb, v_indices

    def _load_audio_features(self, path: Optional[Path]) -> torch.Tensor:
        if path is None:
            if not self.allow_missing_modalities:
                raise RuntimeError("Audio features missing while allow_missing_modalities is False")
            return self._zero_sequence(self.audio_feature_shape)
        audio_data = torch.load(str(path), map_location="cpu")
        audio_emb = audio_data["embedding"]
        if audio_emb.dim() == 3:
            if audio_emb.shape[1] == 1:
                audio_emb = audio_emb.squeeze(1)
            elif audio_emb.shape[0] == 1:
                audio_emb = audio_emb.squeeze(0)
        elif audio_emb.dim() == 1:
            audio_emb = audio_emb.unsqueeze(0)
        return audio_emb

    def _load_sequence_features(
        self, path: Optional[Path], shape: Optional[Tuple[int, ...]]
    ) -> Optional[torch.Tensor]:
        if path is None:
            if not self.allow_missing_modalities:
                return None
            if shape is None:
                return None
            return self._zero_sequence(shape)
        data = torch.load(str(path), map_location="cpu")
        emb = data["embedding"]
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        elif emb.dim() == 3 and emb.shape[1] == 1:
            emb = emb.squeeze(1)
        return emb

    def _zero_sequence(self, shape: Optional[Tuple[int, ...]]) -> torch.Tensor:
        """Placeholder for missing modality: (1,) + shape so e.g. (768,) -> (1, 768). Real data is (seq_len, D)."""
        if shape is None:
            raise RuntimeError("Cannot create placeholder for missing modality without known shape")
        return torch.zeros((1,) + shape, dtype=torch.float32)

    def _infer_feature_shape_from_df(
        self, df: Optional[pd.DataFrame], feature_col: str
    ) -> Optional[Tuple[int, ...]]:
        if df is None or df.empty:
            return None
        for _, row in df.iterrows():
            path = row.get(feature_col)
            if pd.isna(path):
                continue
            candidate = Path(path)
            if candidate.is_file():
                s = self._get_embedding_shape(candidate)
                if s is not None:
                    return s
        return None

    def _infer_feature_shape_from_samples(self, modality: str) -> Optional[Tuple[int, ...]]:
        idx_map = {"video": 0, "audio": 1, "lyrics": 2, "comments": 3}
        col_entry = idx_map.get(modality)
        if col_entry is None:
            return None
        for sample in self.samples:
            path = sample[col_entry]
            if path is not None:
                s = self._get_embedding_shape(path)
                if s is not None:
                    return s
        return None

    def _get_embedding_shape(self, path: Path) -> Optional[Tuple[int, ...]]:
        try:
            data = torch.load(str(path), map_location="cpu")
        except Exception:
            return None
        emb = data.get("embedding")
        if emb is None:
            return None
        if emb.dim() == 3 and emb.shape[1] == 1:
            emb = emb.squeeze(1)
        return tuple(emb.shape[1:])


def collate_multimodal(batch):
    """Collate for multimodal batches with variable-length sequences.

    Batch item: (video_path, audio_path, lyrics_path, comments_path, video_emb, audio_emb, lyrics_emb, comments_emb, label_idx, video_indices)

    Returns:
        video_paths, audio_paths, lyrics_paths, comments_paths,
        video_padded, audio_padded, lyrics_padded, comments_padded,
        video_lengths, audio_lengths, lyrics_lengths, comments_lengths,
        labels, video_indices
    """
    from torch.nn.utils.rnn import pad_sequence

    video_paths = [b[0] for b in batch]
    audio_paths = [b[1] for b in batch]
    lyrics_paths = [b[2] for b in batch]
    comments_paths = [b[3] for b in batch]
    video_seqs = [b[4] for b in batch]
    audio_seqs = [b[5] for b in batch]
    lyrics_seqs = [b[6] for b in batch]
    comments_seqs = [b[7] for b in batch]
    labels = torch.tensor([b[8] for b in batch], dtype=torch.long)
    video_indices = [b[9] for b in batch]

    video_lengths = torch.tensor([v.size(0) for v in video_seqs], dtype=torch.long)
    audio_lengths = torch.tensor([a.size(0) for a in audio_seqs], dtype=torch.long)

    video_padded = pad_sequence(video_seqs, batch_first=True)
    audio_padded = pad_sequence(audio_seqs, batch_first=True)

    lyrics_padded = None
    lyrics_lengths = None
    if all(s is not None for s in lyrics_seqs):
        lyrics_lengths = torch.tensor([s.size(0) for s in lyrics_seqs], dtype=torch.long)
        lyrics_padded = pad_sequence(lyrics_seqs, batch_first=True)

    comments_padded = None
    comments_lengths = None
    if all(s is not None for s in comments_seqs):
        comments_lengths = torch.tensor([s.size(0) for s in comments_seqs], dtype=torch.long)
        comments_padded = pad_sequence(comments_seqs, batch_first=True)

    return (
        video_paths,
        audio_paths,
        lyrics_paths,
        comments_paths,
        video_padded,
        audio_padded,
        lyrics_padded,
        comments_padded,
        video_lengths,
        audio_lengths,
        lyrics_lengths,
        comments_lengths,
        labels,
        video_indices,
    )
