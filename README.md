# Fine-Tuning Open-Source LLM Using LoRA with MLflow and PEFT

## Overview

This repository demonstrates fine-tuning a large language model (LLM), specifically the `Qwen2.5-7B` model, to create a Python coding assistant using **LoRA**, **PEFT**, and **MLflow**. The tutorial focuses on reducing GPU memory requirements and leveraging MLflow to track the training lifecycle, including metrics, model artifacts, and prompt templates.

## Features
- Fine-tune the [Qwen2.5-7B model](https://huggingface.co/Qwen/Qwen2.5-7B) with LoRA and PEFT.
- Log and manage experiments with **MLflow**.
- Use a dataset with 18.6k Python code generation tasks from [HuggingFace](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca).
- Optimize GPU resource usage for training using **LoRA** adapters.
- Save and load to and from MLFlow fine-tuned models with prompt templates and inference parameters.


## Contents

### Dataset Preparation
- Load the `iamtarun/python_code_instructions_18k_alpaca` dataset with 18.6k Python coding tasks.
- Split the dataset into training and testing subsets.

### Model Setup
- Load the `Qwen2.5-7B` model from HuggingFace.
- Configure the tokenizer and setup LoRA adapter using PEFT.

### Fine-Tuning Workflow
- Leverage the `SFTTrainer` for training with PEFT configuration.
- Log all experiments, parameters and metrics to MLflow for reproducibility.

### Model Saving and Deployment
- Save the fine-tuned model with associated prompt template and inference parameters to MLflow.
- Demonstrate how to reload the model for inference with MLflow’s API.

## Installation

### Hardware Requirements
This code is tested on a single NVIDIA H100 GPU with 80GB of VRAM.

### Python Dependencies
Install the required Python packages:
```bash
pip install mlflow==2.15.1 transformers peft accelerate trl datasets
```

Alternatively, use `poetry`:
- Install [poetry](https://python-poetry.org/docs/)
- Run the following to install the necessary dependencies:
    ```bash
    poetry install --no-root
    ```


If using a managed MLflow deployment, set the following environment variables:

```bash
export MLFLOW_TRACKING_SERVER_CERT_PATH=path/to/tracking/server/certificate
export MLFLOW_TRACKING_URI=tracking/server/uri
export MLFLOW_TRACKING_USERNAME=username
export MLFLOW_TRACKING_PASSWORD=password
```
