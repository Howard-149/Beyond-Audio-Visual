# CSV Manifest Unified Format Specification

## Overview

All feature extraction CSV manifests are now unified into the following formats:

### Audio Features
```
key,feature_path,audio_label
```

**Field Descriptions:**
- `key`: Audio file identifier (typically filename without extension)
- `feature_path`: Absolute or relative path to `.pt` feature file
- `audio_label`: Emotion label as integer or string (e.g., 'exciting', 'fear', 'sad', 'relax')

**Example:**
```csv
key,feature_path,audio_label
fear_LLT_DL(10)-2-of-10,/path/to/ER/MER/EmoMV_features/mert/Dataset1/train/fear_LLT_DL(10)-2-of-10.pt,0
relax_10,/path/to/ER/MER/EmoMV_features/mert/Dataset1/train/relax_10.pt,4
```

### Video Features
```
key,feature_path,video_label
```

**Field Descriptions:**
- `key`: Video file identifier (typically filename without extension)
- `feature_path`: Absolute or relative path to `.pt` feature file (schema v1 format)
- `video_label`: Emotion label as integer (0-4) or string (e.g., 'exciting', 'fear', 'tense','sad', 'relax')

**Example:**
```csv
key,feature_path,video_label
clip_0001,/path/to/ER/VER/EmoMV_features/Dataset1/train/clip_0001.pt,0
clip_0002,/path/to/ER/VER/EmoMV_features/Dataset1/train/clip_0002.pt,2
```

## Important Notes

### Why Distinguish `audio_label` and `video_label`?

1. **Clarity**: Immediately identify which modality the label represents
2. **Maintainability**: Reduce confusion, especially in multimodal systems
3. **Scalability**: Easy to extend with additional modalities in the future (e.g., `text_label`)

### Multimodal CSV Format

For systems combining audio and video features, use:

```
key,video_feature_path,audio_feature_path,video_label
```

**Field Descriptions:**
- `key`: Sample identifier
- `video_feature_path`: Path to video feature `.pt` file (schema v1 format)
- `audio_feature_path`: Path to audio feature `.pt` file (schema v1 format)
- `video_label`: Emotion label (from video modality)

## Converting Existing CSVs (legacy only)

New extractions already emit the unified headers. If you still have old dumps,
one-off converters live under `legacy_oneoffs/` (gitignored), e.g.
`convert_csv_headers_unified.py`, `migrate_embeddings_to_schema_v1.py`.
Prefer re-running `MER/extract_EmoMV.py` / `VER/extract_EmoMV.py` / TER extractors
when possible.

## Code Usage

### Reading in Dataset Classes
```python
import pandas as pd

# Audio features
audio_df = pd.read_csv('audio_features.csv')
for _, row in audio_df.iterrows():
    key = row['key']
    feature_path = row['feature_path']
    label = int(row['audio_label'])
    
# Video features
video_df = pd.read_csv('video_features.csv')
for _, row in video_df.iterrows():
    key = row['key']
    feature_path = row['feature_path']
    label = int(row['video_label'])
```

### Integration with DataLoaders
See the following files for implementation details:
- `VER/datasets/datasets.py` - Video DataLoader (expects `feature_path`, `video_label`)
- `MER/extract_EmoMV.py` - Audio feature extraction (generates `key`, `feature_path`, `audio_label`)
- `data/multimodal.py` - Multimodal DataLoader

## Label Mappings

### EmoMV Dataset (5 classes)
| Integer | String |
|---------|--------|
| 0 | exciting |
| 1 | fear |
| 2| tense |
| 3 | sad |
| 4 | relax |

### CREMA-D Dataset (6 classes)
| Integer | String |
|---------|--------|
| 0 | angry |
| 1 | disgust |
| 2 | fear |
| 3 | happy |
| 4 | neutral |
| 5 | sad |
