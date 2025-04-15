# Binary Image Classification with PyTorch Lightning

<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

This project implements a binary image classification system designed to detect visual artifacts in AI-generated images. Built with PyTorch and PyTorch Lightning, it provides a robust framework for training and evaluating models that can distinguish between clean and artifact-containing AI-generated images.

### 💡 Key Features

- **PyTorch Lightning**: Enables clean, maintainable, and scalable code with a modern architecture

- **Hydra**: Manages experiments and configurations with flexible setup

- **Training, Validation, and Testing Workflows**: Provided through a comprehensive training pipeline

- **Micro F1 Score Tracking**: Accurately monitors performance on imbalanced data during training

- **Support for Multiple Architectures and Configs**: Facilitates easy experimentation

- **Gradio**: Provides a simple web interface for testing and demonstrating the model

### 📂 Expected Data Format

The project expects the input dataset to be organized in the following directory structure:

```bash
data/
├── train/
└── test/
```

Each subdirectory (train/, test/) should contain image files named using the following pattern:

```bash
image_<frame_index>_<label>.png
```

Where:

- <frame_index> is a unique numeric ID or frame number

- <label\> is either:

  - 0 – for images with artifacts

  - 1 – for artifact-free images

**Example Filenames:**

- image_00045_0.png

- 0 image_01234_1.png

No additional metadata or folder-level class separation is required — the class label is parsed directly from the filename.

## 🔧 Installation

```bash
# clone project
git clone https://github.com/GalychMaks/BinaryImageClassifier.git
cd BinaryImageClassifier

# [OPTIONAL] create virtual environment
python -m venv .venv
source .venv/bin/activate

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## 🚀 How to run

### 🧠 Training

Train model with default configuration:

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/):

```bash
python src/train.py experiment=example
```

You can override any parameter from command line like this:

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

### 📈 Evaluation

Evaluate trained model:

```bash
python src/eval.py ckpt_path=path/to/checkpoint.ckpt
```

### 🔍 Inference

Run inference on a single image from command line:

```bash
python src/inference.py \
    image_path=path/to/image.png \
    ckpt_path=path/to/checkpoint.ckpt
```

### 🌐 Gradio Interface

Run the Gradio web interface for model testing:

```bash
python app.py
```

This will start a local web server where you can upload images and test the model interactively.

## 📊 Results

I conducted a series of experiments to evaluate the performance of different model architectures and training strategies on a classification task. All training was done on the **It-Jim Trainee Program 2025 Dataset — Task 1 (Deep Learning)**.

Below are the summarized results across training, validation, and test sets. The best scores for each metric are **bolded**.

### 🔬 Experiment Configurations

| Experiment | Model        | Backbone Frozen | Accelerator | Tags                               |
|------------|--------------|------------------|-------------|------------------------------------|
| v1         | ResNet18     | ❌               | GPU         | `["resnet18", "full-finetune"]`    |
| v2         | EfficientNet | ❌               | CPU         | `["efficientnet", "full-finetune"]`|
| v3         | ResNet18     | ✅               | GPU         | `["resnet18", "freeze-backbone"]`  |
| v4         | EfficientNet | ✅               | CPU         | `["efficientnet", "freeze-backbone"]`|

### 📈 Performance Metrics

| Experiment | Train Loss | Val Loss | Test Loss | Train F1 | Val F1 | Test F1 |
|------------|------------|----------|-----------|----------|--------|---------|
| v1         | 0.174      | 0.182    | **0.153** | 0.970    | 0.962  | 0.970   |
| v2         | **0.123**  | **0.094**| 0.164     | **0.977**| **0.985**| **0.981** |
| v3         | 0.253      | 0.186    | 0.205     | 0.953    | 0.965  | 0.952   |
| v4         | 0.286      | 0.215    | 0.240     | 0.946    | 0.967  | 0.949   |

### 📝 Observations

- **Best Overall:** v2 (EfficientNet + full fine-tuning) achieves the best overall performance with the highest F1 scores and lowest train/val losses.
- **Model Architecture:** EfficientNet outperforms ResNet18 in both full fine-tuning and frozen settings.
- **Backbone Freezing:** Freezing the backbone (v3, v4) consistently reduces performance, regardless of model.

## 📚 References

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [PyTorch Lightning](https://www.pytorchlightning.ai/) - Deep learning framework for training and scaling
- [Hydra](https://hydra.cc/) - Framework for elegantly configuring complex applications
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) - Project template for PyTorch Lightning and Hydra
- [Gradio](https://www.gradio.app/) - Web interface for machine learning models
- [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18) - A lightweight residual neural network architecture commonly used for image classification
- [EfficientNetB0](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.efficientnet_b0) - A scalable and efficient convolutional neural network architecture designed for high accuracy with fewer parameters
