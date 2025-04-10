import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18(nn.Module):
    """
    Wrapper around torchvision's ResNet18 with a custom classification head.

    This module loads a pretrained ResNet18 and replaces its final fully connected layer
    to support a specified number of output classes.
    """

    def __init__(self, num_classes: int = 2, weights=ResNet18_Weights.DEFAULT):
        """
        Initialize the ResNet18 model.

        :param num_classes: Number of output classes (e.g., 2 for binary classification).
        :param weights: Pretrained weights to use (e.g., ResNet18_Weights.DEFAULT or None).
        """
        super().__init__()
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input tensor.
        :return: Output logits.
        """
        return self.model(x)
