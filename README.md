# Conditional TIL Image Generation with cGAN

This repository contains the implementation of a conditional Generative Adversarial Network (cGAN) designed to generate Tumor-Infiltrating Lymphocyte (TIL) images. This project aims to augment datasets for researchers working on TIL analysis, potentially aiding in the advancement of immunotherapy and cancer research.

## Overview

Tumor-Infiltrating Lymphocytes (TILs) are a type of white blood cell present within tumors. They are an important part of the immune response to cancer. The ability to generate synthetic, yet realistic, TIL images can significantly contribute to the research by providing augmented data for training machine learning models, thus potentially leading to more accurate predictions and insights.

This cGAN model learns to generate TIL images conditioned on specific features, allowing for the generation of diverse and targeted datasets that can be used to improve the robustness of TIL analysis tools.

## Features

- Conditional image generation of TIL samples.
- Training on custom datasets with diverse TIL characteristics.
- Evaluation metrics including Inception Score (IS) and Frechet Inception Distance (FID) for generated image quality assessment.
- Visualization tools for intermediate layers and model weights to understand model behavior.
- UMAP visualization for latent space exploration.

## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision 0.9+
- scikit-learn 0.24+
- matplotlib 3.3+
- seaborn 0.11+
- umap-learn 0.5+

For a complete list of requirements, please refer to `requirements.txt`.

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/MuteJester/TILcGAN.git
