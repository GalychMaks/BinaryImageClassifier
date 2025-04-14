import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18(nn.Module):
    """
    Wrapper around torchvision's ResNet18 with a custom classification head.

    This module loads a pretrained ResNet18 and replaces its final fully connected layer
    to support a specified number of output classes. Optionally, it can freeze the backbone
    and train only the classification head.
    """

    def __init__(self, num_classes: int = 2, weights=ResNet18_Weights.DEFAULT, freeze_backbone: bool = False):
        """
        Initialize the ResNet18 model.

        :param num_classes: Number of output classes (e.g., 2 for binary classification).
        :param weights: Pretrained weights to use (e.g., ResNet18_Weights.DEFAULT or None).
        :param freeze_backbone: If True, freezes all backbone layers and trains only the classifier.
        """
        super().__init__()
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if freeze_backbone:
            for param in self.model.layer1.parameters():
                param.requires_grad = False
            for param in self.model.layer2.parameters():
                param.requires_grad = False
            for param in self.model.layer3.parameters():
                param.requires_grad = False
            for param in self.model.layer4.parameters():
                param.requires_grad = False
            for param in self.model.conv1.parameters():
                param.requires_grad = False
            for param in self.model.bn1.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: Input tensor.
        :return: Output logits.
        """
        return self.model(x)
