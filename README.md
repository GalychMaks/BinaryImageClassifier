# Binary Image Classification with PyTorch Lightning

<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

This project implements a binary image classification system designed to detect visual artifacts in AI-generated images. Built with PyTorch and PyTorch Lightning, it provides a robust framework for training and evaluating models that can distinguish between clean and artifact-containing AI-generated images.

### Key Features

- **PyTorch Lightning**: Enables clean, maintainable, and scalable code with a modern architecture

- **Hydra**: Manages experiments and configurations with flexible setup

- **Training, Validation, and Testing Workflows**: Provided through a comprehensive training pipeline

- **Micro F1 Score Tracking**: Accurately monitors performance on imbalanced data during training

- **Support for Multiple Architectures and Configs**: Facilitates easy experimentation

- **Gradio**: Provides a simple web interface for testing and demonstrating the model

## Installation

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

## How to run

### Training

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

### Evaluation

Evaluate trained model:

```bash
python src/eval.py ckpt_path=path/to/checkpoint.ckpt
```

### Inference

Run inference on a single image from command line:

```bash
python src/inference.py \
    image_path=path/to/image.png \
    ckpt_path=path/to/checkpoint.ckpt
```

### Gradio Interface

Run the Gradio web interface for model testing:

```bash
python app.py
```

This will start a local web server where you can upload images and test the model interactively.
