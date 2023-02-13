#!/usr/bin/env python3
##
## EPITECH PROJECT, 2023
## main.py
## File description:
## main
##

import sys
import torch
from train import *
sys.path.append('./model/')
from model import MyModel

PATH = "../data/model/model.pt"

def make_model():
    model = MyModel()
    model = train_ia(model)
    test_ia(model)
    torch.save(model.state_dict(), PATH)

make_model()
