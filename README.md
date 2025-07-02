<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white" alt="Python">
    </a>
    <a href="https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
    </a>
    <a href="https://arxiv.org/abs/2406.15958">
        <img src="https://img.shields.io/badge/arXiv-2406.15958-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv Paper">
    </a>
    <a href="https://github.com/weights-and-biases/wandb">
        <img src="https://img.shields.io/badge/Weights%20%26%20Biases-Enabled-FFBE00?logo=wandb&logoColor=black" alt="Weights & Biases">
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-green.svg?logo=license&logoColor=white" alt="License: MIT">
    </a>
</p>

# Bone-Fracture-Classification

This repository contains a deep learning-based system for bone fracture classification. The primary objective of this project is to accurately classify whether a given X-ray image of a bone exhibits a fracture. The system is developed using PyTorch, a powerful open-source machine learning library, and leverages a pre-trained EfficientNet-B6 model, a state-of-the-art convolutional neural network (CNN), to achieve high classification accuracy.

The automatic detection of fractures in radiological images is a critical task in clinical practice. It can assist radiologists and orthopedic surgeons in diagnosing fractures more quickly and accurately, leading to better patient outcomes. This project aims to provide a robust and efficient solution by employing transfer learning, a technique where a model developed for a task is reused as the starting point for a model on a second task.

## The FracAtlas Dataset

The performance of any deep learning model is heavily dependent on the quality and size of the dataset used for training. This project utilizes the **FracAtlas dataset**, a large and comprehensive collection of bone X-ray images. The dataset is specifically curated for fracture detection tasks and contains a diverse set of images, covering various types of bones and fracture patterns.

The dataset is categorized into two main classes:
*   **Fractured**: X-ray images that clearly show one or more fractures.
*   **Non-fractured**: X-ray images of healthy, unbroken bones.

To ensure a robust and unbiased evaluation of the model, the dataset is partitioned into three distinct subsets:
*   **Training set (80%)**: This is the largest portion of the dataset and is used to train the neural network. The model learns to identify the features and patterns associated with fractures from these images.
*   **Validation set (10%)**: This subset is used to tune the model's hyperparameters and to monitor its performance during the training process. It helps in preventing overfitting, a phenomenon where the model performs well on the training data but poorly on unseen data.
*   **Test set (10%)**: This portion of the dataset is held out until the very end of the development process. It is used to provide a final, unbiased evaluation of the model's performance on completely unseen data, simulating a real-world scenario.

## Data Preprocessing and Loading

Before the images can be fed into the neural network, they must undergo a series of preprocessing steps. This is a crucial stage that ensures the data is in a suitable format for the model and helps in improving its performance. The following transformations are applied to each image:

*   **Resize**: All images are resized to a uniform dimension of 224x224 pixels. This is a standard input size for many pre-trained models, including EfficientNet, and it ensures that all input tensors have the same shape.
*   **Convert to PyTorch Tensor**: The images, which are initially in PIL (Python Imaging Library) format, are converted into PyTorch tensors. Tensors are the fundamental data structure used in PyTorch for all computations.
*   **Normalization**: The pixel values of the images are normalized. This is a common practice in deep learning that helps in stabilizing and speeding up the training process. The images are normalized using the mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. These specific values are the pre-computed mean and standard deviation of the ImageNet dataset, on which the EfficientNet model was originally trained. Using the same normalization values is essential for effective transfer learning.

The preprocessed data is then loaded in batches using PyTorch's `DataLoader`. A batch size of 16 is used for training, validation, and testing. Batching allows for more efficient use of memory and computational resources during training.

## Model Architecture and Transfer Learning

The core of this classification system is a pre-trained **EfficientNet-B6** model. EfficientNet is a family of convolutional neural networks known for their high accuracy and computational efficiency. The `efficientnet-pytorch` library is used to easily load and integrate the model into the PyTorch framework.

This project employs **transfer learning**, a powerful technique in deep learning where a model pre-trained on a large dataset (like ImageNet) is adapted for a new, specific task. The rationale behind transfer learning is that the features learned by the model on the large dataset (e.g., edges, textures, shapes) are often general enough to be useful for other tasks.

The implementation of transfer learning in this project involves the following steps:
1.  **Loading the Pre-trained Model**: An EfficientNet-B6 model, with weights pre-trained on the ImageNet dataset, is loaded.
2.  **Freezing the Convolutional Layers**: The weights of all the convolutional layers in the pre-trained model are frozen. This means that their values will not be updated during the training process. This is done to retain the general-purpose features learned from the ImageNet dataset.
3.  **Replacing the Final Classifier**: The final fully connected layer (the classifier) of the pre-trained model is replaced with a new linear layer. This new layer is designed for our specific task and has 2 output units, corresponding to the two classes in our dataset (fractured and non-fractured). Only the weights of this new layer will be trained on the FracAtlas dataset.

## Loss Function

The loss function is a critical component of the training process that measures how well the model's predictions match the actual labels. For this multi-class classification problem, the **Cross-Entropy Loss** (`nn.CrossEntropyLoss`) is used. This loss function is a standard and effective choice for classification tasks. It combines the `LogSoftmax` and `NLLLoss` (Negative Log Likelihood Loss) in a single class, making it numerically stable and efficient.

## Training the Model

The model is trained for 20 epochs. An epoch is one complete pass through the entire training dataset. The training process is driven by the **Adam optimizer**, a popular and effective optimization algorithm. The learning rate, which controls how much the model's weights are updated in response to the estimated error, is set to 0.001.

To monitor the training process and save the best performing model, the validation set is used. After each epoch, the model's performance is evaluated on the validation set, and the model with the highest validation accuracy is saved.

The entire training process, including the training loss and validation accuracy at each epoch, is logged using **`wandb`** (Weights & Biases). `wandb` is a powerful tool for experiment tracking, visualization, and collaboration in machine learning projects.

## Inference

After the training process is complete, the best saved model (the one with the highest validation accuracy) is loaded. This model is then used for inference on the test set. The test set, being completely unseen by the model during training and validation, provides the most reliable measure of the model's generalization performance. The final test accuracy is calculated to assess how well the model is expected to perform in a real-world application.
