#!/usr/bin/env python3
##
## EPITECH PROJECT, 2023
## model
## File description:
## model
##
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d (1, 1, kernel_size = 5)
        self.maxpool1 = torch.nn.MaxPool2d (kernel_size = 2, stride = 2)
        self.conv2 = torch.nn.Conv2d (1, 1, kernel_size = 5)
        self.maxpool2 = torch.nn.MaxPool2d (kernel_size = 2, stride = 2)
        self.linear1 = torch.nn.Linear(16, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.maxpool2(x)

        x = x.reshape(-1, 4 * 4)

        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.linear3(x)
        return x

model = MyModel()
