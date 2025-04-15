import os
import tempfile

import gradio as gr
from hydra import main
from omegaconf import DictConfig

from src.inference import predict_image


@main(version_base="1.3", config_path="configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    """
    Launch a Gradio web interface for image classification inference.

    Uses Hydra to initialize the configuration, allowing full interpolation support
    (e.g., `${hydra:runtime.output_dir}`), and dynamically updates the configuration
    with the uploaded image path and model checkpoint.

    The interface provides a wrapper around the `predict_image` function, which
    handles image preprocessing, model inference, and result formatting.

    :param cfg: The composed Hydra configuration from `inference.yaml`.
    """

    def gradio_predict(pil_image):
        """
        Predict the class of the uploaded image and return the result as a string.

        This function:
          - Saves the uploaded PIL image to a temporary file.
          - Updates the Hydra config with the image and checkpoint paths.
          - Runs prediction using the model defined in the config.
          - Deletes the temporary image file after prediction.

        :param pil_image: The image uploaded by the user.
        :return: A formatted prediction string containing the class and confidence.
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_image.save(tmp.name)
            image_path = tmp.name

        cfg.image_path = image_path

        predicted_class, confidence = predict_image(cfg)

        os.remove(image_path)

        return f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}"

    interface = gr.Interface(
        fn=gradio_predict,
        inputs=gr.Image(type="pil", label="Upload an image"),
        outputs="text",
        title="Binary Image Classifier",
        description="Upload an image and run inference using Hydra configuration.",
    )

    interface.launch()


if __name__ == "__main__":
    main()
