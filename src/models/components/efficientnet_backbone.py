import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetB0(nn.Module):
    """
    Wrapper around torchvision's EfficientNet-B0 with a custom classification head.

    This module loads a pretrained EfficientNet-B0 and replaces its final fully connected layer
    to support a specified number of output classes. Optionally, it can freeze the backbone
    and train only the classification head.
    """

    def __init__(self, num_classes: int = 2, weights=EfficientNet_B0_Weights.DEFAULT, freeze_backbone: bool = False):
        """
        Initialize the EfficientNet-B0 model.

        :param num_classes: Number of output classes (e.g., 2 for binary classification).
        :param weights: Pretrained weights to use (e.g., EfficientNet_B0_Weights.DEFAULT or None).
        :param freeze_backbone: If True, freezes all backbone layers and trains only the classifier.
        """
        super().__init__()
        self.model = efficientnet_b0(weights=weights)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input tensor.
        :return: Output logits.
        """
        return self.model(x)
