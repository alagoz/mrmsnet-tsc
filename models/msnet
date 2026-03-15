import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class MSNet(nn.Module):
    """
    Multi-Scale Representation Network (MSNet)
    """

    def __init__(self, in_channels, n_classes, hidden=64):
        super().__init__()

        # multi-scale conv branches
        self.conv3 = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3)

        self.fusion = FusionBlock(hidden * 3, hidden)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        """
        x: (B, C, L)
        """

        b3 = F.relu(self.conv3(x))
        b5 = F.relu(self.conv5(x))
        b7 = F.relu(self.conv7(x))

        x = torch.cat([b3, b5, b7], dim=1)

        x = self.fusion(x)

        x = self.pool(x).squeeze(-1)

        return self.fc(x)
