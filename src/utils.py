"""
utils.py
--------
Utility helper functions for the Video Highlight Detection project.

This module provides common utilities such as setting up hardware devices (GPU/CPU),
creating necessary directories, setting random seeds for reproducibility,
and visual plotting helpers.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Parameters:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Utils] Random seed set to: {seed}")


def get_device() -> torch.device:
    """
    Checks if a GPU (CUDA) is available and returns the appropriate PyTorch device.
    
    Returns:
        torch.device: 'cuda' if GPU is available, else 'cpu'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Utils] Using device: {device}")
    return device


def ensure_directories_exist(dir_paths: list[str]) -> None:
    """
    Ensures that all specified directory paths exist. Creates them if missing.
    
    Parameters:
        dir_paths (list[str]): List of folder paths to verify/create.
    """
    for path in dir_paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"[Utils] Created directory: {path}")


def plot_importance_scores(scores: np.ndarray, save_path: str = None, title: str = "Video Segment Importance Scores") -> None:
    """
    Plots the predicted importance scores for each video segment/frame over time.
    
    Parameters:
        scores (np.ndarray): 1D array of importance scores (0.0 to 1.0).
        save_path (str, optional): If provided, saves the plot image to this path.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(scores, label="Importance Score", color="crimson", linewidth=2)
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Threshold (0.5)")
    plt.xlabel("Segment / Frame Index")
    plt.ylabel("Importance Score")
    plt.title(title)
    plt.ylim([0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Utils] Saved importance score plot to: {save_path}")
    plt.close()
