"""Test script for flexible modality fusion API.

Tests:
1. ConcatFusion with 2 modalities (audio+video, audio+text, video+text)
2. ConcatFusion with 3 modalities (audio+video+text)
3. CrossAttentionFusion with video->audio, audio->video, video->text
"""

import torch
from models.fusion import (
    ConcatFusion,
    CrossAttentionFusion,
    VideoQueryAttentionFusion,
)


def test_concat_fusion_audio_video():
    """Test ConcatFusion with audio+video."""
    print("Testing ConcatFusion with audio+video...")
    fusion = ConcatFusion(
        audio_dim=768,
        video_dim=512,
        text_dim=0,  # Not used
        num_classes=5,
        fusion_config={"hidden_dim": 256},
    )
    
    B = 4
    modality_feats = {
        'audio': torch.randn(B, 768),
        'video': torch.randn(B, 512),
    }
    
    logits = fusion(modality_feats)
    assert logits.shape == (B, 5), f"Expected (4, 5), got {logits.shape}"
    print("✓ audio+video concat fusion passed")


def test_concat_fusion_audio_text():
    """Test ConcatFusion with audio+text."""
    print("\nTesting ConcatFusion with audio+text...")
    fusion = ConcatFusion(
        audio_dim=768,
        video_dim=0,  # Not used
        text_dim=768,
        num_classes=5,
        fusion_config={"hidden_dim": 256},
    )
    
    B = 4
    modality_feats = {
        'audio': torch.randn(B, 768),
        'text': torch.randn(B, 768),
    }
    
    logits = fusion(modality_feats)
    assert logits.shape == (B, 5), f"Expected (4, 5), got {logits.shape}"
    print("✓ audio+text concat fusion passed")


def test_concat_fusion_trimodal():
    """Test ConcatFusion with audio+video+text."""
    print("\nTesting ConcatFusion with audio+video+text...")
    fusion = ConcatFusion(
        audio_dim=768,
        video_dim=512,
        text_dim=768,
        num_classes=5,
        fusion_config={"hidden_dim": 256},
    )
    
    B = 4
    modality_feats = {
        'audio': torch.randn(B, 768),
        'video': torch.randn(B, 512),
        'text': torch.randn(B, 768),
    }
    
    logits = fusion(modality_feats)
    assert logits.shape == (B, 5), f"Expected (4, 5), got {logits.shape}"
    print("✓ audio+video+text concat fusion passed")


def test_cross_attention_video_audio():
    """Test CrossAttentionFusion with video->audio."""
    print("\nTesting CrossAttentionFusion with video->audio...")
    fusion = CrossAttentionFusion(
        audio_dim=768,
        video_dim=512,
        text_dim=0,
        num_classes=5,
        mode="cross_attention",
        q_modality="video",
        k_modality="audio",
        v_modality="audio",
        bidirectional=True,
        fusion_config={},
    )
    
    B = 2
    T_v, T_a = 10, 50
    modality_seqs = {
        'video': torch.randn(B, T_v, 512),
        'audio': torch.randn(B, T_a, 768),
    }
    modality_lengths = {
        'video': torch.tensor([8, 10]),
        'audio': torch.tensor([40, 50]),
    }
    # Mock video indices (frame numbers)
    video_indices = [
        list(range(0, 80, 10)),  # 10 frames at 10fps spacing
        list(range(0, 100, 10)),
    ]
    
    logits = fusion(
        modality_seqs=modality_seqs,
        modality_lengths=modality_lengths,
        video_indices=video_indices,
    )
    assert logits.shape == (B, 5), f"Expected (2, 5), got {logits.shape}"
    print("✓ video->audio cross-attention passed")


def test_cross_attention_audio_video():
    """Test CrossAttentionFusion with audio->video (reverse)."""
    print("\nTesting CrossAttentionFusion with audio->video...")
    fusion = CrossAttentionFusion(
        audio_dim=768,
        video_dim=512,
        text_dim=0,
        num_classes=5,
        mode="cross_attention",
        q_modality="audio",
        k_modality="video",
        v_modality="video",
        bidirectional=True,
        fusion_config={},
    )
    
    B = 2
    T_v, T_a = 10, 50
    modality_seqs = {
        'video': torch.randn(B, T_v, 512),
        'audio': torch.randn(B, T_a, 768),
    }
    modality_lengths = {
        'video': torch.tensor([8, 10]),
        'audio': torch.tensor([40, 50]),
    }
    
    # When audio is Q, we don't need video_indices (only needed when video is Q)
    logits = fusion(
        modality_seqs=modality_seqs,
        modality_lengths=modality_lengths,
        video_indices=None,
    )
    assert logits.shape == (B, 5), f"Expected (2, 5), got {logits.shape}"
    print("✓ audio->video cross-attention passed")


def test_video_query_cross_attention():
    """Test the video-query attention pipeline that aligns modalities before fusion."""
    print("\nTesting VideoQueryAttentionFusion...")
    fusion = VideoQueryAttentionFusion(
        video_dim=512,
        audio_dim=768,
        text_dim=768,
        num_classes=5,
        fusion_config={
            "hidden_dim": 256,
            "cross_attn_heads": 4,
            "dropout": 0.1,
            "text_seconds_per_emb": 0.1,
        },
    )

    B = 2
    T_v, T_a, T_t = 6, 12, 10
    video_seq = torch.randn(B, T_v, 512)
    audio_seq = torch.randn(B, T_a, 768)
    text_seq = torch.randn(B, T_t, 768)
    video_lengths = torch.tensor([6, 6])
    audio_lengths = torch.tensor([10, 12])
    text_lengths = torch.tensor([8, 10])
    video_indices = [list(range(0, 60, 10)), list(range(0, 60, 10))]

    logits = fusion(
        video_seq,
        video_lengths,
        audio_seq,
        audio_lengths,
        text_seq,
        text_lengths,
        video_indices,
    )
    assert logits.shape == (B, 5), f"Expected (2, 5), got {logits.shape}"
    print("✓ video-query attention fusion passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Flexible Modality Fusion API")
    print("=" * 60)
    
    test_concat_fusion_audio_video()
    test_concat_fusion_audio_text()
    test_concat_fusion_trimodal()
    test_cross_attention_video_audio()
    test_cross_attention_audio_video()
    test_video_query_cross_attention()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
