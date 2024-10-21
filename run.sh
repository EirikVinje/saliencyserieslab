#!/bin/bash -e

python -m saliencyserieslab.train_sktime_classifier --model inception --id 123
python -m saliencyserieslab.train_sktime_classifier --model resnet --id 123
python -m saliencyserieslab.train_sktime_classifier --model rocket --id 123

python -m saliencyserieslab.generate_explanations --model inception --explainer lime --id 123
python -m saliencyserieslab.generate_explanations --model inception --explainer kernelshap --id 123

python -m saliencyserieslab.generate_explanations --model rocket --explainer lime --id 123
python -m saliencyserieslab.generate_explanations --model rocket --explainer kernelshap --id 123

python -m saliencyserieslab.generate_explanations --model resnet --explainer lime --id 123
python -m saliencyserieslab.generate_explanations --model resnet --explainer kernelshap --id 123