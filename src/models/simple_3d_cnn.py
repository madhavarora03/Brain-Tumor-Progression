import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN3D(nn.Module):
    def __init__(self, in_channels=12, num_classes=4):
        super(CNN3D, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # Output shape: (B, 128, 1, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),               # (B, 128)
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def test():
    model = CNN3D(in_channels=12, num_classes=4)
    dummy = torch.randn(1, 12, 128, 128, 128)  # (B, C, D, H, W)
    out = model(dummy)
    print("Output shape:", out.shape)


if __name__ == "__main__":
    test()
