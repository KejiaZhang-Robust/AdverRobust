<img src="images/README/AdverRobust.png" alt="AdverRobust Logo" style="width: 300px; float: left; margin-right: 20px;">

<br>

# **AdverRobust: An Adversarial Training Framework for Adversarial Robustness in Deep Learning Models**

## Introduction

**AdverRobust** is a PyTorch-based repository designed to provide a comprehensive framework for adversarial training and robustness evaluation in visual tasks. With state-of-the-art adversarial training algorithms and robustness assessment tools, AdverRobust aims to facilitate research and development in adversarial robustness for deep learning models. This repository is highly modular, making it easy to integrate with existing projects and customize for various tasks and model architectures.

## Adversarial Training Frameworks

AdverRobust supports several state-of-the-art adversarial training frameworks, each providing unique strategies to enhance model robustness. Below are the methods available in this repository, along with their references:

- **PGD-AT (Projected Gradient Descent - Adversarial Training)**
  A classic adversarial training method that uses iterative gradient-based attack to generate adversarial examples.
  _Reference_: [Madry et al., 2018, ICLR](https://arxiv.org/abs/1706.06083)
- **TRADES (TRadeoff-inspired Adversarial Defense via Surrogate-loss minimization)**
  Balances robustness and accuracy by minimizing KL-divergence between logits generated from adversarial and natural exmaples.
  _Reference_: [Zhang et al., 2019, ICML](https://arxiv.org/abs/1901.08573)
- **MART (Misclassification Aware Adversarial Training)**
  An extend of TRADES, MART focuses on misclassified examples to further enhance model robustness by adjusting the loss function.
  _Reference_: [Wang et al., 2020, ICLR](https://openreview.net/forum?id=rklOg6EFwS)
- **AWP (Adversarial Weight Perturbation)**
  Regularizes the training process by minimizing the change in loss relative to the weight (weight loss landscape).
  _Reference_: [Wu et al., 2020, NeurIPS](https://arxiv.org/abs/2004.05884)
- **FSR (Feature Separation and Recalibration - Adversarial Training)**
  Recalibrates the non-robust activations to restore discriminative cues that help the model make correct predictions under adversarial attack.
  _Reference_: [Kim et al., 2023, CVPR](https://arxiv.org/abs/2303.13846)
- **FPCM (Frequency Preference Control Module - Adversarial Training)**
  Introduces a module adaptively reconfigures the low- and highfrequency components of intermediate feature representations.
  _Reference_: [Bu et al., 2023, ICCV](https://arxiv.org/abs/2307.09763)

## Installation

To get started with AdverRobust, you need to set up your environment with the necessary dependencies. Follow the steps below to install the required packages:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/KejiaZhang-Robust/AdverRobust
   cd AdverRobust
   ```

2. **Requirements**
   A suitable [conda](https://conda.io/) environment named `at_robust` can be created
   and activated with:

   ```
   conda env create -f environment.yaml
   conda activate at_robust
   ```

## Usage

To use the AdverRobust framework for adversarial training or robustness evaluation, follow these steps:

### 1. Edit `configs_train.yml` File

The `configs_train.yml` file contains all the configuration parameters needed to customize your adversarial training and evaluation process. You need to edit this file to select suitable parameters for your specific task. Below is an explanation of the key sections and parameters within the file:

```yaml
# General Operation Parameters
Operation:
  # Name of the experiment or model configuration
  Prefix: "WRN34_10"
  # A description or any notes related to the training process
  record_words: ""
  # Whether to resume training from a checkpoint
  Resume: False

# Training Configuration
Train:
  # Model of training: Choose from [ResNet-18, WRN34-10, etc]
  Train_Net: "ResNet-18"
  # Method of training: Choose from [PGD-AT, TRADES, MART, FSR, FPCM, Natural]
  Train_Method: "AT"
  # Dataset to use: Options are [CIFAR10, CIFAR100, TinyImageNet, Imagenette]
  Data: "CIFAR10"
  # Number of training epochs
  Epoch: 110
  # Initial learning rate
  Lr: 0.1
  # Factor for label smoothing (if applicable)
  Factor: 0
  # Parameters for adversarial training: epsilon clipping, FGSM steps, PGD steps
  clip_eps: 8
  fgsm_step: 2
  pgd_train: 10
  # Epochs at which to change the learning rate
  lr_change_iter: [100, 105]

# Dataset Configuration
DATA:
  # Number of output classes for the dataset
  num_class: 10
  # Mean and standard deviation for normalizing the dataset
  mean: !!python/tuple [0.4914, 0.4822, 0.4465]
  std: !!python/tuple [0.2471, 0.2435, 0.2616]

# Adversarial Attack Parameters
ADV:
  # FGSM (Fast Gradient Sign Method) parameters during training
  clip_eps: 8
  fgsm_step: 2
  # PGD (Projected Gradient Descent) attack parameters used during validation
  pgd_attack_test: 10
  pgd_attack_1: 20
  pgd_attack_2: 100
```

#### Explanation of Key Sections

- **Operation**: This section defines the basic operational parameters for your training process, such as the model name (`Prefix`), optional notes (`record_words`), and whether to resume from a previous checkpoint (`Resume`).
- **Train**: Specifies the core training parameters, including:

  - `Train_Net`: Choose the model from Adversarial Training Frameworks.
  - `Train_Method`: Choose the training method from Adversarial Training Frameworks.
  - `Data`: Select the dataset for training (e.g., CIFAR10, CIFAR100, TinyImageNet, Imagenette).
  - `Epoch`: The total number of epochs for training.
  - `Lr`: The initial learning rate, with a special note for setting a different rate for certain datasets like Imagenette.
  - `clip_eps`, `fgsm_step`, `pgd_train`: Parameters for configuring adversarial attacks used during training, such as FGSM steps and PGD steps.
  - `lr_change_iter`: Epochs where the learning rate will change to allow for dynamic learning rate schedules.

- **DATA**: Contains dataset-related configurations such as the number of classes (`num_class`) and normalization parameters (`mean`, `std`) used for standardizing the dataset.
- **ADV**: Configures adversarial attack parameters used during training and testing phases:

  - `clip_eps`, `fgsm_step`: Parameters defining the epsilon clipping and FGSM steps during training.
  - `pgd_attack_test`, `pgd_attack_1`, `pgd_attack_2`: Parameters for PGD attack configurations during evaluation, specifying the number of attack iterations.

By properly configuring the `configs_train.yml` file, you can tailor the adversarial training process to fit various models, datasets, and adversarial attack methods. This flexibility is crucial for developing robust deep learning models in adversarial environments.

### 2. Run the Training Script

After configuring the `configs_train.yml` file with your desired parameters, run the `train_model.py` script to start the adversarial training process:

```bash
python train_model.py
```

This command will initiate the training process using the parameters specified in the `configs_train.yml` file. Make sure you have all the dependencies installed and the appropriate environment set up as mentioned in the **Installation** section.

By following these steps, you can effectively utilize the AdverRobust framework for adversarial training and robustness evaluation of deep learning models.

Here's a revised version for the **Testing Robust Performance** section:

### 3. Test the Robust Performance

To evaluate the robustness of a trained model using AdverRobust, you can perform standard robustness testing or transfer robustness testing. Our framework supports multiple adversarial attack methods, including FGSM, PGD, CW, and AutoAttack. Below are the steps for each type of testing:

#### 3.1. Standard Robustness Testing

To test the robustness of a trained model against adversarial attacks, run the `test_robust.py` script. This script loads the trained model and evaluates it using adversarial attacks defined in the `configs_test.yml` configuration file:

```bash
python test_robust.py
```

The `configs_test.yml` file contains testing parameters that are largely consistent with those in `configs_train.yml`, such as dataset settings, adversarial attack parameters, and other hyperparameters. You can modify `configs_test.yml` to customize your testing setup.

#### 3.2. Transfer Robustness Testing

To evaluate the transferability of the model's robustness to different tasks or datasets, use the `test_transfer.py` script:

```bash
python test_transfer.py
```

This script uses the `configs_test_transfer.yml` configuration file, which also follows a structure similar to `configs_train.yml`. It allows you to fine-tune the parameters for evaluating the robustness transfer of your model.

#### 3.3. Configuration Files Overview

Both `configs_test.yml` and `configs_test_transfer.yml` are designed to provide flexibility in testing by sharing a similar format with `configs_train.yml`. This allows for consistent configuration management across training and testing phases. Adjust these configuration files according to your testing needs to comprehensively assess the robustness and transferability of your models using AdverRobust.
