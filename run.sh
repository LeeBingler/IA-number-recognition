#!/usr/bin/env bash
##
## EPITECH PROJECT, 2023
## run.sh
## File description:
## file that make the IA work
##

if [ -f "./data/model.pt" ]
then
    cd src/
    python app.py
else
    cd src/
    python src/make_model.py
    python src/app.py
fi
