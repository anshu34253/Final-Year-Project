"""
dataset.py
----------
PyTorch Dataset class for loading extracted video features and TVSum annotation labels.

This module formats extracted ResNet-50 feature matrices and target importance score labels
into PyTorch Tensors for training and evaluation.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TVSumDataset(Dataset):
    """
    PyTorch Dataset for TVSum video feature vectors and target segment scores.

    Parameters:
        features_dir (str): Directory containing preprocessed .npy feature files.
        annotations_file (str): Path to annotations file (.tsv, .csv, or .h5) with ground truth scores.
        max_seq_len (int): Standardized sequence length to pad/crop feature vectors.
    """

    def __init__(self, features_dir: str, annotations_file: str = None, max_seq_len: int = 300):
        super(TVSumDataset, self).__init__()

        self.features_dir = features_dir
        self.annotations_file = annotations_file
        self.max_seq_len = max_seq_len

        # List all preprocessed .npy feature files in directory
        self.feature_files = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
        
        # Load ground truth annotations if provided
        self.annotations = None
        if annotations_file and os.path.exists(annotations_file):
            print(f"[Dataset] Loading annotations from: {annotations_file}")
            # TVSum annotations typically map video ID to frame scores
            if annotations_file.endswith(".tsv") or annotations_file.endswith(".csv"):
                self.annotations = pd.read_csv(annotations_file, sep="\t")

    def __len__(self) -> int:
        """Returns total number of feature sequence samples in dataset."""
        return len(self.feature_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves feature sequence tensor and corresponding target score tensor at index.

        Returns:
            features (torch.Tensor): Tensor of shape (max_seq_len, 2048).
            target_scores (torch.Tensor): Tensor of shape (max_seq_len, 1).
        """
        feature_path = self.feature_files[idx]
        feature_matrix = np.load(feature_path)  # Shape: (N_frames, 2048)

        seq_len, num_features = feature_matrix.shape

        # Standardize sequence length: trim if too long, zero-pad if too short
        if seq_len > self.max_seq_len:
            feature_matrix = feature_matrix[:self.max_seq_len, :]
            target_scores = np.ones((self.max_seq_len, 1), dtype=np.float32) * 0.5
        else:
            padding = np.zeros((self.max_seq_len - seq_len, num_features), dtype=np.float32)
            feature_matrix = np.vstack([feature_matrix, padding])
            
            # Dummy scores (0.5) for illustration until annotations are parsed
            scores_raw = np.ones((seq_len, 1), dtype=np.float32) * 0.5
            scores_padding = np.zeros((self.max_seq_len - seq_len, 1), dtype=np.float32)
            target_scores = np.vstack([scores_raw, scores_padding])

        # Convert NumPy arrays to PyTorch FloatTensors
        features_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
        target_scores_tensor = torch.tensor(target_scores, dtype=torch.float32)

        return features_tensor, target_scores_tensor


def create_dataloader(features_dir: str, batch_size: int = 4, shuffle: bool = True) -> DataLoader:
    """
    Helper function to instantiate a PyTorch DataLoader for mini-batching.

    Parameters:
        features_dir (str): Folder path containing .npy feature files.
        batch_size (int): Batch size per iteration.
        shuffle (bool): Whether to shuffle samples each epoch.

    Returns:
        DataLoader: Ready-to-use PyTorch DataLoader instance.
    """
    dataset = TVSumDataset(features_dir=features_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


if __name__ == "__main__":
    print("Testing dataset module...")
