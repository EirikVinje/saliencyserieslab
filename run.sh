#!/bin/bash -e

python -m saliencyserieslab.train_sktime_classifier --model inception --id 123
python -m saliencyserieslab.train_sktime_classifier --model resnet --id 123
python -m saliencyserieslab.train_sktime_classifier --model rocket --id 123

python -m saliencyserieslab.generate_explanations --id 123

python -m saliencyserieslab.recommender --expdir exp_123