#!/usr/bin/env python3
##
## EPITECH PROJECT, 2023
## train.py
## File description:
## function tha train the ia
##

import torchvision
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def train_ia(model):
    train_set = torchvision.datasets.MNIST("../data/", train=True, download = True, transform = torchvision.transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size = 64, shuffle = True)

    EPOCHS = 3

    loss_fonct = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(EPOCHS):
        for batch in train_loader:
            images, labels = batch
            output = model.forward(images.reshape(images.shape[0], 1, 28, 28))
            loss = loss_fonct(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss.item())
    return model

def test_ia(model):
    test_set = torchvision.datasets.MNIST("../data/", train=False, download = True, transform = torchvision.transforms.ToTensor())
    total, correct = 0, 0
    for image, label in test_set:
        output = model.forward(image.reshape(image.shape[0], 1, 28, 28))
        if (output.argmax(dim=1).item() == label):
            correct += 1
        total += 1

    accuracy = correct / total * 100
    print(f'Your accuracy is {accuracy}% !')
    return model
