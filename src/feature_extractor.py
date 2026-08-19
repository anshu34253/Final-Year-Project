"""
feature_extractor.py
--------------------
Extracts deep visual features from video frames using a pre-trained ResNet-50 model.

ResNet-50 converts raw image frames into compact 2048-dimensional feature vectors.
These visual vectors capture high-level semantic information (objects, actions, background)
which will be fed into the LSTM model to learn temporal relationships.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


class ResNet50FeatureExtractor(nn.Module):
    """
    ResNet-50 model with the final classification layer removed.
    Outputs a 2048-dimensional vector for each input image.
    """

    def __init__(self, pretrained: bool = True):
        super(ResNet50FeatureExtractor, self).__init__()

        # Load pre-trained ResNet-50 model weights from torchvision
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Replace the final fully-connected (fc) classification layer with Identity
        # This keeps the 2048-d feature map output from the average pooling layer
        resnet.fc = nn.Identity()
        self.model = resnet
        self.model.eval()  # Set model to evaluation mode (disables dropout & batchnorm update)

        # Standard ImageNet preprocessing transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean values
                std=[0.229, 0.224, 0.225]   # ImageNet standard deviation values
            )
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes preprocessed image tensor through ResNet-50.
        
        Parameters:
            x (torch.Tensor): Tensor of shape (batch_size, 3, 224, 224).
            
        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, 2048).
        """
        with torch.no_grad():
            features = self.model(x)
        return features

    def extract_from_frames(self, frames: list[np.ndarray], device: torch.device, batch_size: int = 32) -> np.ndarray:
        """
        Extracts 2048-d features from a list of RGB numpy frames in batches.

        Parameters:
            frames (list[np.ndarray]): List of frame images (RGB).
            device (torch.device): PyTorch device ('cuda' or 'cpu').
            batch_size (int): Number of frames to process in a single GPU pass.

        Returns:
            np.ndarray: Matrix of shape (N_frames, 2048) containing feature vectors.
        """
        self.model.to(device)
        all_features = []

        # Process frames in mini-batches for GPU efficiency
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Apply preprocessing transform to each frame in batch
            tensor_list = [self.transform(frame) for frame in batch_frames]
            batch_tensor = torch.stack(tensor_list).to(device)

            # Pass through ResNet-50 feature extractor
            with torch.no_grad():
                features = self.model(batch_tensor)  # Shape: (batch_size, 2048)
                
            all_features.append(features.cpu().numpy())

        if len(all_features) == 0:
            return np.empty((0, 2048))

        # Concatenate all batches into a single matrix (N_frames, 2048)
        feature_matrix = np.concatenate(all_features, axis=0)
        print(f"[FeatureExtractor] Extracted features shape: {feature_matrix.shape}")
        return feature_matrix


if __name__ == "__main__":
    # Standalone verification test
    print("Testing ResNet-50 Feature Extractor...")
    extractor = ResNet50FeatureExtractor(pretrained=True)
    dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    device = torch.device("cpu")
    feats = extractor.extract_from_frames([dummy_frame], device=device)
    print("Test output shape:", feats.shape)  # Expected: (1, 2048)
