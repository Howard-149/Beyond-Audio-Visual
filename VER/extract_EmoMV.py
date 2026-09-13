#!/usr/bin/env python3
"""
Extract per-video facial features for EmoMV using MTCNN + EmotiEffLibRecognizer
(optional MoEDE via --model-name moede).

Writes EmbeddingSchema v1 .pt files and a manifest CSV with columns:
  key, feature_path, video_label
"""

import os
import argparse
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import torch
import dlib
import traceback
import csv
from utils.embedding_schema import EmbeddingV1, now_iso8601, validate_embedding_schema_v1
try:
    from facenet_pytorch import MTCNN
except Exception:
    print(traceback.format_exc())
    MTCNN = None
try:
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer
except Exception:
    EmotiEffLibRecognizer = None


def recognize_faces(frame, mtcnn):
    if mtcnn is None:
        raise RuntimeError('MTCNN not available')
    boxes, probs = mtcnn.detect(frame, landmarks=False)
    if probs is None or all(p is None for p in probs):
        return []
    mask = np.array(probs) > 0.9
    if not np.any(mask):
        return []
    boxes = boxes[mask]
    faces = []
    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        face = frame[y1:y2, x1:x2]
        face_size = (y2 - y1) * (x2 - x1)
        if face is None:
            continue
        if face.size == 0:
            continue
        if face.shape[-1] != 3:
            continue
        faces.append((face, face_size))
    return faces


def process_video(vid_path, mtcnn, fer, max_frames, frame_step=1, face_select='largest') -> tuple[np.ndarray|None, str|None]:
    cap = cv2.VideoCapture(vid_path)
    feats = []
    indices = []
    i = 0
    frame_count = 0
    while cap.isOpened() and (max_frames == -1 or frame_count < max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if (i % frame_step) != 0:
            i += 1
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = recognize_faces(rgb, mtcnn)
        if len(faces) > 0:
            try:
                if face_select == 'mean':
                    face_imgs = [f[0] for f in faces]
                    f_feats = fer.extract_features(face_imgs)  # (N, D=1280)
                    f_feats = f_feats.mean(axis=0)  # mean pool if multiple faces
                else:  # 'largest'  
                    largest_face = max(faces, key=lambda x: x[1])[0]  # select face with largest size
                    f_feats = fer.extract_features([largest_face])  # (1, D=1280)
            except Exception:
                traceback.print_exc()
                f_feats = None
            if f_feats is not None:
                feats.append(np.array(f_feats, dtype=np.float32))
                indices.append(i)
            frame_count += 1
        i += 1
    cap.release()
    if len(feats) == 0:
        return None, None
    idx_str = ",".join(str(x) for x in indices)
    return np.vstack(feats), idx_str # (T, D)

def moede_process_video(vid_path, fer, max_frames, frame_step, device, face_select='largest') -> tuple[np.ndarray|None, str|None]:
    from moede.data_preprocess import FaceAligner, get_landmark_dlib, preprocess_img

    cap = cv2.VideoCapture(vid_path)
    feats = []
    indices = []
    i = 0
    frame_count = 0
    
    detector = dlib.get_frontal_face_detector()
    predictor_path = Path("static_models") / "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(str(predictor_path))
    fa = FaceAligner(desiredFaceWidth=256)
    while cap.isOpened() and (max_frames == -1 or frame_count < max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if (i % frame_step) != 0:
            i += 1
            continue
        face = get_landmark_dlib(frame, detector, predictor) # get largest face
        if face is not None:
            try:
                if face_select == 'mean':
                    raise RuntimeError('MOEDE does not support mean face selection yet')
                else:  # 'largest'
                    align_img = preprocess_img(frame, face, fa)
                    align_imgs = [align_img]
                    f_feats = fer(torch.stack(align_imgs).to(device)).detach().cpu().numpy() # should be (1,expers=8,D=1280)
            except Exception:
                traceback.print_exc()
                f_feats = None
            if f_feats is not None:
                feats.append(np.array(f_feats, dtype=np.float32))
                indices.append(i)
            frame_count += 1
        i += 1
    cap.release()
    if len(feats) == 0:
        return None, None
    idx_str = ",".join(str(x) for x in indices)
    return np.vstack(feats), idx_str # (T,  D)
def get_video_files_with_labels(data_root, split, subset):
    """Load video files and their labels from annotation CSV.
    
    Returns:
        dict mapping video_stem -> (video_label as int)
    """
    n = subset[-1] # '1' or '2' or '3'
    video_labels = {}
    csv_path = os.path.join(data_root, f"{subset}/annotation/DS{n}_{split.upper()}_MATCH_MISMATCH_labels.csv")
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7: # skip invalid rows
                continue
            if row[-2] != row[-1]: #only use matched videos
                continue
            folder = row[0]
            filename = row[1] + ".mp4"
            label = int(row[-2])  # column -2 (0-indexed) contains the label as int
            full_path = os.path.join(data_root, folder, filename)
            video_stem = Path(filename).stem
            video_labels[full_path] = (video_stem, label)
    return video_labels





def main():
    p = argparse.ArgumentParser(description='Extract EmoMV per-video face features')
    p.add_argument('--input-base', required=True, help='Root with Dataset*/{train,val,test} media')
    p.add_argument('--output-dir', required=True, help='Where to write .pt features and CSVs')
    p.add_argument('--subsets', default=['Dataset1'])
    p.add_argument('--model-name', default='enet_b0_8_best_afew')
    p.add_argument('--max-frames', type=int, default=-1, help='Maximum number of frames to process per video, use -1 for all frames')
    p.add_argument('--frame-step', type=int, default=1, help='Process every N-th frame')
    p.add_argument('--limit', type=int, default=0, help='If >0, only process this many videos (for quick tests)')
    p.add_argument('--face-select', type=str, default='mean',choices=['largest', 'mean'], help='Face selection strategy: largest or mean')
    p.add_argument('--debug', action='store_true', help='Print debug info (file discovery, counts)')
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = args.input_base
    if MTCNN is None:
        raise RuntimeError('facenet-pytorch required')
    if EmotiEffLibRecognizer is None and args.model_name != 'moede':
        raise RuntimeError('EmotiEffLib required')

    mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
    if args.model_name == 'moede':
        from moede.models import EM
        fer = EM()
        for i in range(8):
            fer.experts[i].to(device)
        fer.to(device)
    else:
        fer = EmotiEffLibRecognizer(engine='onnx', model_name=args.model_name, device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    # Support common video extensions (CREMA-D is distributed as .flv in this repo)
    exts = ['mp4', 'flv', 'avi', 'mov', 'mkv']
    # discover all video files under input_base
    for subset in args.subsets:
        for split in ['train', 'val', 'test']:
            video_labels = get_video_files_with_labels(args.input_base, split, subset=subset)
            video_files = list(video_labels.keys())
            manifest = []
            
            # Extract FPS from first video for the entire dataset
            dataset_fps = None
            if len(video_files) > 0:
                cap_temp = cv2.VideoCapture(video_files[0])
                if cap_temp.isOpened():
                    dataset_fps = cap_temp.get(cv2.CAP_PROP_FPS)
                cap_temp.release()
                print(f"Using FPS={dataset_fps} for {subset}/{split}")
            for vid in tqdm(video_files):
                if args.limit > 0 and len(manifest) >= args.limit:
                    break
                stem, video_label = video_labels[vid]
                if args.model_name == 'moede':
                    feat,idx_str = moede_process_video(vid, fer, max_frames=args.max_frames, frame_step=args.frame_step, device=device, face_select=args.face_select)
                else:
                    feat,idx_str = process_video(vid, mtcnn, fer, max_frames=args.max_frames, frame_step=args.frame_step, face_select=args.face_select)
                if feat is None:
                    continue
                if args.debug:
                    print(f"Processing {vid} -> {feat.shape}")
                out_path = os.path.join(args.output_dir, f"{subset}",f"{split}", f"{stem}.pt")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                
                # Build schema-compliant video embedding
                feat_tensor = torch.tensor(feat, dtype=torch.float32)
                if feat_tensor.dim() == 2:
                    temporal_dim = int(feat_tensor.shape[0])
                    embedding_dim = int(feat_tensor.shape[1])
                elif feat_tensor.dim() == 3:
                    # Multi-expert: (T, E, D)
                    temporal_dim = int(feat_tensor.shape[0])
                    embedding_dim = int(feat_tensor.shape[2])  # D dimension
                else:
                    raise ValueError(f"Unexpected feature shape: {feat_tensor.shape}")
                
                schema = EmbeddingV1(
                    embedding=feat_tensor,
                    modality="video",
                    model_name=args.model_name,
                    layer="feature_extraction",
                    pooling="none",
                    embedding_dim=embedding_dim,
                    temporal_dim=temporal_dim,
                    schema_version="1.0",
                    created_at=now_iso8601(),
                    sample_rate=None,
                    fps=dataset_fps,
                    extra_metadata={
                        "clipName": stem,
                        "indices": idx_str,
                        "subset": subset,
                        "split": split,
                        "source_path": str(vid),
                    },
                ).to_dict()
                
                validate_embedding_schema_v1(schema)
                torch.save(schema, out_path)
                manifest.append({'key': stem, 'feature_path': out_path, 'video_label': video_label})

            if manifest:
                mf = pd.DataFrame(manifest)
                csv_path = os.path.join(args.output_dir, f"{subset}", f'{split}_features.csv')
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                mf.to_csv(csv_path, index=False)
                print(f'Wrote {len(manifest)} features to {csv_path}')
            else:
                print('No features extracted')


if __name__ == '__main__':
    main()
