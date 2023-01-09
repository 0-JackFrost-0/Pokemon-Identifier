import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# from torchvision import datasets, transforms
# from torchvision.transforms import ToTensor, Lambda


class NeuralNetwork1(nn.Module):
    def __init__(self):
        super(NeuralNetwork1, self).__init__()
        self.flatten = nn.Flatten()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear = nn.Sequential(
            nn.Linear(173056, 1200),
            nn.Linear(1200, 500),
            nn.Linear(500, 4*7*7),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(43264, 150),
            # nn.Linear(43264, 1024),
            # nn.Linear(1024, 150)
        )
        

    def forward(self, x):
        # x = self.flatten(x)
        # print(x.size())
        x = x.permute(0,2,1,3)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # x = self.linear(x)
        x = self.linear_layers(x)
        return x

model = resnet50(weights="IMAGENET1K_V1")


num_ftrs = model_ft.fc.in_features
# Initialize the Weight Transforms
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

# Apply it to the input image
img_transformed = preprocess(img)

class NeuralNetwork2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        
            # nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            # nn.ReLU(),
            # nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            # nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            # nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            # nn.ReLU(),
            # nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            # nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.lin_layers = nn.Sequential(
            
            nn.Linear(43264,150),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # # nn.ReLU(),
            # nn.Linear(512,150)
        )
    
    def forward(self, x):
        x = x.permute(0,2,1,3)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.lin_layers(x)
        return x



if __name__ == "__main__":
    model = NeuralNetwork2()
    print(model)