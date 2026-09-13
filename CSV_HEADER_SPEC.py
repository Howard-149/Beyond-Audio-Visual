#!/usr/bin/env python3
"""Quick reference: unified CSV manifest header specification per modality.

Documents expected fieldnames for audio / video / multimodal manifests used by
MER, VER, and multimodal_classifier.py.
"""

UNIFIED_SPEC = {
    "audio": {
        "fieldnames": ["key", "feature_path", "audio_label"],
        "example": {
            "key": "fear_LLT_DL(10)-2-of-10",
            "feature_path": "/path/to/embeddings/fear_LLT_DL(10)-2-of-10.pt",
            "audio_label": "0"  # or "fear", "exciting", etc.
        },
        "used_by": [
            "MER/extract_EmoMV.py",
            "multimodal_classifier.py (--audio-csv)"
        ]
    },
    "video": {
        "fieldnames": ["key", "feature_path", "video_label"],
        "example": {
            "key": "clip_0001",
            "feature_path": "/path/to/features/clip_0001.pt",
            "video_label": "0"  # integer 0-4 or string
        },
        "used_by": [
            "VER/extract_EmoMV.py",
            "VER/datasets/datasets.py",
            "multimodal_classifier.py (--video-csv)"
        ]
    },
    "multimodal": {
        "fieldnames": ["key", "video_feature_path", "audio_feature_path", "video_label"],
        "example": {
            "key": "sample_001",
            "video_feature_path": "/path/to/video.pt",
            "audio_feature_path": "/path/to/audio.pt",
            "video_label": "0"
        },
        "used_by": [
            "data/multimodal.py",
            "multimodal_classifier.py"
        ]
    }
}

if __name__ == "__main__":
    import json
    print(json.dumps(UNIFIED_SPEC, indent=2))
