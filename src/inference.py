from pathlib import Path
from typing import Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, task_wrapper  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def predict_image(cfg: DictConfig) -> Tuple[str, float]:
    """
    Run prediction on a single image.

    :param cfg: Config containing model configuration, checkpoint path, and image path.
    :returns: Tuple containing the predicted class name and confidence score.
    """
    assert cfg.image_path, "Image path must be provided!"
    assert cfg.ckpt_path, "Checkpoint path must be provided!"

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Open and transform the image
    image = Image.open(cfg.image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Instantiate model using the target class from the config and load the checkpoint
    model_class = hydra.utils.get_class(cfg.model._target_)
    model = model_class.load_from_checkpoint(cfg.ckpt_path)
    model.eval()

    # Ensure the input tensor is on the same device as the model
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        # Adjust your logic for determining class_name if necessary
        class_name = "artifact" if predicted_class == 0 else "clean"

    return class_name, confidence


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for single image prediction.

    :param cfg: Configuration composed by Hydra.
    """
    assert cfg.image_path, "Image path must be specified!"
    assert cfg.ckpt_path, "Checkpoint path must be specified!"

    image_path = Path(cfg.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    class_name, confidence = predict_image(cfg)

    log.info(f"Image: {image_path.name}")
    log.info(f"Prediction: {class_name}")
    log.info(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
