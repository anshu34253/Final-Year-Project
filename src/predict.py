"""
predict.py
----------
Inference module for predicting segment importance scores on new input videos.

This module loads trained weights from `models/best_model.pth`, evaluates feature tensors
through the LSTM model, generates per-segment importance scores, and saves output CSV files.
"""

import os
import numpy as np
import pandas as pd
import torch

from src.lstm_model import VideoHighlightLSTM
from src.utils import get_device, plot_importance_scores


def predict_importance_scores(
    features: np.ndarray,
    model_path: str = "models/best_model.pth",
    device: torch.device = None
) -> np.ndarray:
    """
    Predicts importance scores for each segment/frame in a video feature matrix.

    Parameters:
        features (np.ndarray): Array of shape (N_frames, 2048) extracted via ResNet-50.
        model_path (str): Path to saved PyTorch model checkpoint (.pth).
        device (torch.device, optional): Device to run inference on.

    Returns:
        np.ndarray: 1D array of predicted importance scores (shape: N_frames, range: [0.0, 1.0]).
    """
    if device is None:
        device = get_device()

    # Instantiate model architecture
    model = VideoHighlightLSTM(input_dim=2048, hidden_dim=256, num_layers=2)

    # Load trained model weights if available, otherwise notify and use initialized weights
    if os.path.exists(model_path):
        print(f"[Predict] Loading model checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(f"[Predict] Warning: Model file '{model_path}' not found. Using untrained initial weights.")

    model.to(device)
    model.eval()

    # Convert NumPy features to PyTorch Tensor: shape (1, sequence_length, 2048)
    feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    # Perform inference (forward pass without gradient tracking)
    with torch.no_grad():
        score_tensor = model(feature_tensor)  # Shape: (1, sequence_length, 1)

    # Remove batch and feature dimensions to return a 1D NumPy score array
    scores = score_tensor.squeeze().cpu().numpy()

    # Handle single element edge case
    if scores.ndim == 0:
        scores = np.array([scores.item()])

    print(f"[Predict] Generated importance scores for {len(scores)} segments.")
    return scores


def save_predictions(scores: np.ndarray, output_csv_path: str) -> pd.DataFrame:
    """
    Saves segment importance scores to a CSV file.

    Parameters:
        scores (np.ndarray): 1D array of scores.
        output_csv_path (str): Destination CSV path.

    Returns:
        pd.DataFrame: Formatted DataFrame containing segment indices and scores.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    df = pd.DataFrame({
        "segment_index": np.arange(len(scores)),
        "importance_score": scores
    })
    
    df.to_csv(output_csv_path, index=False)
    print(f"[Predict] Saved predictions CSV to: {output_csv_path}")
    return df


if __name__ == "__main__":
    print("Predict module ready.")
