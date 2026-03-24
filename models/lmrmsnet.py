import torch
import torch.nn as nn
import torch.nn.functional as F


class LMRMSNet(nn.Module):
    """
    Lightweight Multi-Scale Network (LS-Net)
    """

    def __init__(self, in_channels, n_classes, hidden=32, exit_threshold=0.8):
        super().__init__()

        self.exit_threshold = exit_threshold

        self.conv3 = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # early classifier
        self.early_fc = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

        # main pathway
        self.conv_main = nn.Sequential(
            nn.Conv1d(hidden * 2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        self.main_fc = nn.Linear(128, n_classes)

    def forward(self, x, inference=False):

        b3 = F.relu(self.conv3(x))
        b5 = F.relu(self.conv5(x))

        x = torch.cat([b3, b5], dim=1)

        pooled = self.pool(x).squeeze(-1)

        early_logits = self.early_fc(pooled)

        if inference:
            probs = torch.softmax(early_logits, dim=1)
            conf, _ = probs.max(dim=1)

            if conf.mean() > self.exit_threshold:
                return early_logits

        x = self.conv_main(x)

        x = self.pool(x).squeeze(-1)

        return self.main_fc(x)
