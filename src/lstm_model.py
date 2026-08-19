"""
lstm_model.py
--------------
PyTorch LSTM network architecture for temporal video highlight prediction.

The LSTM receives a sequence of frame feature vectors (shape: [batch_size, seq_len, 2048])
and learns temporal patterns across time. It outputs an importance score (between 0.0 and 1.0)
for every single segment/frame in the video sequence.
"""

import torch
import torch.nn as nn


class VideoHighlightLSTM(nn.Module):
    """
    Bidirectional LSTM model for scoring video frames/segments.
    
    Attributes:
        input_dim (int): Dimension of feature vector per frame (default: 2048 from ResNet-50).
        hidden_dim (int): Number of hidden units in LSTM layer (default: 256).
        num_layers (int): Number of stacked LSTM layers (default: 2).
        dropout (float): Dropout probability between layers to prevent overfitting.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(VideoHighlightLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM layer
        # batch_first=True expects input shape (batch_size, seq_len, input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Fully connected regression head
        # Since the LSTM is bidirectional, output hidden dimension is hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Outputs score strictly in range [0.0, 1.0]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM model.

        Parameters:
            x (torch.Tensor): Feature sequence tensor of shape (batch_size, sequence_length, 2048).

        Returns:
            torch.Tensor: Predicted importance scores of shape (batch_size, sequence_length, 1).
        """
        # lstm_out shape: (batch_size, sequence_length, hidden_dim * 2)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Pass each time-step output through the linear scoring head
        # scores shape: (batch_size, sequence_length, 1)
        scores = self.fc(lstm_out)

        return scores


if __name__ == "__main__":
    # Simple architecture verification test
    print("Testing VideoHighlightLSTM architecture...")
    model = VideoHighlightLSTM(input_dim=2048, hidden_dim=256, num_layers=2)
    print(model)

    # Dummy input representing a batch of 2 videos, each 100 frames long with 2048 features per frame
    dummy_input = torch.randn(2, 100, 2048)
    dummy_output = model(dummy_input)

    print("Dummy input shape: ", dummy_input.shape)   # Expected: (2, 100, 2048)
    print("Dummy output shape:", dummy_output.shape)  # Expected: (2, 100, 1)
