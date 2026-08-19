"""
train.py
--------
Training module for the LSTM Video Highlight Detection model.

This module defines the training loop, loss calculation (MSE Loss between predicted
importance scores and ground truth scores), backpropagation, optimizer steps,
and model checkpoint saving to `models/best_model.pth`.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.lstm_model import VideoHighlightLSTM
from src.utils import get_device, set_seed


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 10,
    lr: float = 1e-3,
    save_path: str = "models/best_model.pth"
) -> VideoHighlightLSTM:
    """
    Executes the training loop over a specified number of epochs.

    Parameters:
        train_loader (DataLoader): PyTorch DataLoader containing training feature batches.
        val_loader (DataLoader, optional): DataLoader for validation dataset.
        epochs (int): Total training epochs.
        lr (float): Learning rate for Adam optimizer.
        save_path (str): File path where the trained model weights will be saved.

    Returns:
        VideoHighlightLSTM: The trained PyTorch model.
    """
    set_seed(42)
    device = get_device()

    # Instantiate the LSTM model architecture
    model = VideoHighlightLSTM(input_dim=2048, hidden_dim=256, num_layers=2)
    model.to(device)

    # Define Loss Function (Mean Squared Error for continuous importance scores)
    criterion = nn.MSELoss()

    # Define Optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_loss = float("inf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\n[Train] Starting training for {epochs} epochs...")
    print(f"[Train] Model saving path: {save_path}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (features, targets) in enumerate(train_loader):
            # Move data tensors to active device (GPU or CPU)
            features = features.to(device)  # Shape: (batch_size, seq_len, 2048)
            targets = targets.to(device)    # Shape: (batch_size, seq_len, 1)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass through LSTM model
            predictions = model(features)   # Shape: (batch_size, seq_len, 1)

            # Compute loss
            loss = criterion(predictions, targets)

            # Backward pass (compute gradients)
            loss.backward()

            # Gradient clipping to prevent exploding gradients in LSTM
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update model weights
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_loader))

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_train_loss:.4f}")

        # Save best model checkpoint
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model checkpoint to {save_path}")

    print("\n[Train] Model training complete!")
    return model


if __name__ == "__main__":
    print("Train module ready. Call train_model() with DataLoader to begin training.")
