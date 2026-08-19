"""
dataset.py
----------
PyTorch Dataset class for loading TVSum video features and ground-truth annotations.

This module parses the TVSum JSONL annotation files (`tvsum_train_release.jsonl` and
`tvsum_val_release.jsonl`), converts 20-annotator 1-5 ratings into normalized segment
importance scores [0.0, 1.0], and temporally aligns them with extracted visual features.
"""

import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_tvsum_annotations(annotations_dir: str = "data/annotations") -> dict:
    """
    Loads all TVSum annotations from .jsonl files in annotations_dir.

    Parameters:
        annotations_dir (str): Path to annotations directory.

    Returns:
        dict: Mapping of video_id -> annotation item dictionary.
    """
    annotations = {}
    jsonl_files = sorted(glob.glob(os.path.join(annotations_dir, "*.jsonl")))
    
    for jf in jsonl_files:
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    annotations[item["vid"]] = item

    return annotations


def get_normalized_segment_scores(labels_2d: list[list[float]]) -> np.ndarray:
    """
    Converts 2D annotator ratings [N_segments, 20] into a 1D normalized score array [N_segments].
    Each 2-second video segment is rated by 20 crowd annotators on a scale of 1 to 5.

    Parameters:
        labels_2d (list[list[float]]): List of length N_segments containing 20 scores each.

    Returns:
        np.ndarray: 1D array of shape (N_segments,) with score values normalized to [0.0, 1.0].
    """
    labels_arr = np.array(labels_2d, dtype=np.float32)  # Shape: (N_segments, 20)
    
    # Compute mean score across the 20 annotators for each segment
    mean_scores = np.mean(labels_arr, axis=1)  # Shape: (N_segments,)
    
    # Scale from 1-5 rating range to [0.0, 1.0] interval
    normalized_scores = (mean_scores - 1.0) / 4.0
    normalized_scores = np.clip(normalized_scores, 0.0, 1.0)
    
    return normalized_scores


def align_scores_with_features(scores: np.ndarray, num_features: int) -> np.ndarray:
    """
    Aligns 2-second segment scores with extracted frame feature sequences.
    
    If features are sampled at 1 FPS (num_features ≈ 2 * num_segments), each segment score
    is repeated for 2 consecutive seconds.
    If features are sampled at 0.5 FPS (num_features ≈ num_segments), alignment is 1-to-1.

    Parameters:
        scores (np.ndarray): 1D array of segment scores of length N_segments.
        num_features (int): Number of extracted frame feature vectors.

    Returns:
        np.ndarray: Aligned 1D score array of length num_features.
    """
    num_segments = len(scores)

    # Case 1: 1 FPS sampling (approx. 2 frames per 2-second segment)
    if abs(num_features - 2 * num_segments) <= 10:
        aligned = np.repeat(scores, 2)
    # Case 2: 0.5 FPS sampling (1 frame per 2-second segment)
    else:
        aligned = scores

    # Adjust exact length to match num_features
    if len(aligned) > num_features:
        aligned = aligned[:num_features]
    elif len(aligned) < num_features:
        pad_len = num_features - len(aligned)
        aligned = np.pad(aligned, (0, pad_len), mode="edge")

    return aligned


class TVSumDataset(Dataset):
    """
    PyTorch Dataset for TVSum feature vectors and target segment scores.

    Parameters:
        features_dir (str): Folder containing preprocessed .npy feature files.
        annotations_dir (str): Folder containing .jsonl annotation files.
        max_seq_len (int): Maximum sequence length for padding/cropping.
    """

    def __init__(self, features_dir: str = "data/processed", annotations_dir: str = "data/annotations", max_seq_len: int = 300):
        super(TVSumDataset, self).__init__()

        self.features_dir = features_dir
        self.annotations_dir = annotations_dir
        self.max_seq_len = max_seq_len

        self.annotations = load_tvsum_annotations(annotations_dir)
        self.feature_files = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
        self.samples = []

        for fpath in self.feature_files:
            video_id = os.path.splitext(os.path.basename(fpath))[0].replace("_features", "")
            if video_id in self.annotations:
                self.samples.append({
                    "vid": video_id,
                    "feature_path": fpath,
                    "annotation": self.annotations[video_id]
                })

        print(f"[Dataset] TVSumDataset initialized with {len(self.samples)} aligned samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Retrieves aligned feature sequence and target score tensors.

        Returns:
            features_tensor (torch.Tensor): Shape (max_seq_len, 2048).
            targets_tensor (torch.Tensor): Shape (max_seq_len, 1).
            video_id (str): Video ID.
        """
        sample = self.samples[idx]
        vid = sample["vid"]
        feature_matrix = np.load(sample["feature_path"])  # Shape: (N_frames, 2048)
        
        raw_labels = sample["annotation"]["label"]
        seg_scores = get_normalized_segment_scores(raw_labels)

        # Align scores with number of feature frames
        num_frames = feature_matrix.shape[0]
        aligned_scores = align_scores_with_features(seg_scores, num_frames)

        seq_len, num_features = feature_matrix.shape

        # Pad or trim sequence to max_seq_len
        if seq_len > self.max_seq_len:
            feature_matrix = feature_matrix[:self.max_seq_len, :]
            padded_scores = aligned_scores[:self.max_seq_len].reshape(-1, 1)
        else:
            feat_pad = np.zeros((self.max_seq_len - seq_len, num_features), dtype=np.float32)
            feature_matrix = np.vstack([feature_matrix, feat_pad])

            score_pad = np.zeros((self.max_seq_len - seq_len,), dtype=np.float32)
            padded_scores = np.concatenate([aligned_scores, score_pad]).reshape(-1, 1)

        features_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
        targets_tensor = torch.tensor(padded_scores, dtype=torch.float32)

        return features_tensor, targets_tensor, vid


def create_dataloader(features_dir: str = "data/processed", annotations_dir: str = "data/annotations", batch_size: int = 4, shuffle: bool = True) -> DataLoader:
    """Helper function to create DataLoader instance."""
    dataset = TVSumDataset(features_dir=features_dir, annotations_dir=annotations_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


if __name__ == "__main__":
    print("Testing TVSum Dataset module...")
