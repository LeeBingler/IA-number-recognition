#!/usr/bin/env python3
##
## EPITECH PROJECT, 2023
## app.py
## File description:
## launch the app
##
import sys
import gradio as gr
import torch
from train import *
sys.path.append('./model/')
from model import MyModel

PATH = "../data/model/model.pt"

def app(input):
    model = MyModel()
    model.load_state_dict(torch.load(PATH))
    model.eval()
    input = torch.from_numpy(input).to("cuda")
    input = input.type(torch.FloatTensor)
    input = input.reshape(1, 1, 28, 28)
    output = model.forward(input)
    return output.argmax(dim=1).item()


gr.Interface(fn = app, inputs = "sketchpad", outputs = "text").launch()
