from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import F1Score


class ArtifactImageLitModule(LightningModule):
    """
    A LightningModule for binary classification of generated images with and without artifacts.

    This module wraps any backbone model (e.g., ResNet18) and supports training, validation,
    and testing with micro F1 tracking.
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        compile: bool = False,
    ) -> None:
        """
        Initialize the classification module.

        :param net: Neural network model to use for classification.
        :param optimizer: Optimizer instance.
        :param scheduler: Optional learning rate scheduler.
        :param compile: Whether to compile the model with `torch.compile()`.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = nn.CrossEntropyLoss()

        self.train_f1 = F1Score(task="binary", average="micro")
        self.val_f1 = F1Score(task="binary", average="micro")
        self.test_f1 = F1Score(task="binary", average="micro")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_f1_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        :param x: Batch of input images.
        :return: Model logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """
        Reset validation metrics at the start of training.
        """
        self.val_loss.reset()
        self.val_f1.reset()
        self.val_f1_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shared step for training, validation, and testing.

        :param batch: A tuple of (input images, target labels).
        :return: A tuple of (loss, predictions, targets).
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Run a training step on a single batch.

        :param batch: A batch of training data.
        :param batch_idx: Index of the batch.
        :return: Training loss.
        """
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_f1(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Run a validation step on a single batch.

        :param batch: A batch of validation data.
        :param batch_idx: Index of the batch.
        """
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_f1(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        At the end of a validation epoch, update the best F1 score.
        """
        current_f1 = self.val_f1.compute()
        self.val_f1_best(current_f1)
        self.log("val/f1_best", self.val_f1_best.compute(), prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Run a test step on a single batch.

        :param batch: A batch of test data.
        :param batch_idx: Index of the batch.
        """
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_f1(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """
        Setup the module before training or evaluation.

        :param stage: One of "fit", "validate", "test", or "predict".
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and optional learning rate scheduler.

        :return: Dictionary with optimizer and optional scheduler.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
