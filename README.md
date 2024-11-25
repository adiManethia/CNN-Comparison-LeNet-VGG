# CNN-Comparison-LeNet-VGG
Comparison of LeNet, VGG16, and VGG19 Architectures in Computer Vision📊

## Introduction

If you're just starting out in the world of deep learning and computer vision, understanding the foundational architectures is key. This repository provides an exploration of three basic and important CNN architectures: LeNet, VGG16, and VGG19. These models have played a crucial role in advancing computer vision research and practical applications.

In this repository, you'll explore how these architectures differ in terms of complexity, depth, and performance. We will also implement these models from scratch using PyTorch, providing a solid foundation for understanding how they work.

## Why these Architectures are important for beginners?
**1. LeNet Architecture**📝
LeNet, developed by Yann LeCun [original paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf), is one of the pioneering CNN architectures. Originally designed for handwritten digit recognition (MNIST), LeNet serves as the perfect starting point for CNNs.

Key Features of LeNet:
 * Shallow Architecture: Composed of 5 layers — 3 convolutional layers and 2 fully connected 
   layers.
 * Simplified Model: Its simplicity makes it an excellent choice for beginners to understand 
   the principles of CNNs.
   
Dataset Used:
 * MNIST dataset: A collection of 60,000 28x28 grayscale images of handwritten digits (0-9) 
   used for classification tasks.
   
```python
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
**2. VGG16 Architecture**💻
VGG16, introduced by the Visual Geometry Group at Oxford [original paper](https://arxiv.org/abs/1409.1556), marked a significant step forward with its use of small 3x3 filters and deep architectures. VGG16 proved that increasing the depth of a network could improve accuracy, making it one of the most popular networks for image classification tasks.

Key Features of VGG16:
 * Depth: 16 layers with 13 convolutional layers.
 * Small 3x3 Filters: The architecture uses small 3x3 filters, making it flexible and easy to 
   scale.
 * Uniform Architecture: The network structure follows a repetitive pattern of convolution 
   layers followed by max-pooling layers.

Dataset Used:
 * FashionMNIST dataset: A collection of 60,000 28x28 grayscale images of clothing items, 
   categorized into 10 classes (e.g., t-shirts, dresses, shoes).
```python
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more layers as required...
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```
**3. VGG19 Architecture**⚡
VGG19 is a deeper version of VGG16. The only difference is the number of layers: VGG19 has 19 layers (16 convolutional and 3 fully connected layers). This increased depth provides more capacity to the network and improves its ability to learn more complex features.

Key Features of VGG19:
 * Deeper Architecture: 19 layers vs. 16 in VGG16, offering more capacity.
 * Similar Structure: Like VGG16, VGG19 uses small 3x3 filters and is easy to scale for more 
   complex tasks.
```python
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more layers as required...
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```
## 🔍 Key Differences Summary:![lg](https://github.com/user-attachments/assets/d0caec25-536a-4ed0-b409-4c8bf8c4f2be)

## How to Clone and Run the Notebooks
**Clone the Repository**
Run the following command in your terminal or command prompy:
```bash
git clone https://github.com/adiManethia/CNN-Comparison-LeNet-VGG.git
```
**Run the Notebooks**
1. Navigate to the repository directory:
   ```bash
   cd CNN-Comparison-LeNet-VGG
   ```
2. Open the jupyter Notebooks:
   ```bash
   jupyter notebook
   ```
3. Select the appropriate notebook (`LENET_5.ipynb`, `VGG16_19`) to run and start training process.

## Conclusion🎯
* LeNet is a simple yet effective architecture, ideal for small datasets like MNIST. It is an excellent starting point for beginners who are new to convolutional neural networks (CNNs). With only a few layers, it’s easy to understand and computationally efficient, but it may not perform well on more complex datasets like FashionMNIST.
* VGG16 and VGG19 are deeper and more complex networks, designed to capture finer details in images by stacking multiple convolutional layers. These architectures are powerful for handling more complex datasets like FashionMNIST. VGG19 is slightly faster than VGG16 despite having more layers. This is because VGG19's layer design and architecture allow for more efficient feature extraction compared to VGG16, especially in cases where more complex patterns need to be learned.

Overall, VGG16 and VGG19 are more powerful than LeNet, but they come at the cost of higher computational power and longer training times. VGG19 offers an improvement in speed over VGG16, making it a better choice when working with large datasets or when model efficiency is key.

Each of these architectures serves a different purpose in the world of computer vision. LeNet is best for beginners and small datasets, VGG16 is great for more complex tasks with a good balance of performance and computational cost, and VGG19 provides a faster, deeper alternative for tackling even more sophisticated image recognition challenges.
