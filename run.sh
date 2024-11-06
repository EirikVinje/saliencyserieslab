#!/bin/bash -e

python saliencyserieslab/amee_train_classifier.py --model rocket --dataset ECG5000 --save --id 1 
python saliencyserieslab/amee_train_classifier.py --model weasel --dataset ECG5000 --save --id 1 
python saliencyserieslab/amee_train_classifier.py --model mrseql --dataset ECG5000 --save --id 1

python saliencyserieslab/amee_train_classifier.py --model rocket --dataset Planes --save --id 1 
python saliencyserieslab/amee_train_classifier.py --model weasel --dataset Planes --save --id 1 
python saliencyserieslab/amee_train_classifier.py --model mrseql --dataset Planes --save --id 1

python saliencyserieslab/amee_train_classifier.py --model rocket --dataset ECG200 --save --id 1 
python saliencyserieslab/amee_train_classifier.py --model weasel --dataset ECG200 --save --id 1 
python saliencyserieslab/amee_train_classifier.py --model mrseql --dataset ECG200 --save --id 1

python saliencyserieslab/amee_train_classifier.py --model rocket --dataset SwedishLeaf --save --id 1 
python saliencyserieslab/amee_train_classifier.py --model weasel --dataset SwedishLeaf --save --id 1 
python saliencyserieslab/amee_train_classifier.py --model mrseql --dataset SwedishLeaf --save --id 1

git add models/*
git commit -m "add trained models"
git push