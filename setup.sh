#!/bin/bash -e

if [ "$(basename "$PWD")" != "saliencyserieslab" ]; then
    echo "Error: You are not in the project root directory (saliencyserieslab)."
    echo "Please navigate to the correct directory and run this script again."
    exit 1
fi

DATA_PATH="$(pwd)/data/insectsound"
if [ ! -d "$DATA_PATH" ]; then
    cd data
    echo "Downloading the InsectSound dataset..."
    wget https://www.timeseriesclassification.com/aeon-toolkit/InsectSound.zip
    unzip InsectSound.zip
    mkdir -p data/insectsound
    mv InsectSound/* data/insectsound
    rm -rf InsectSound.zip
    rm -rf InsectSound
    cd ..
fi

LOG_PATH="$(pwd)/log"
if [ ! -d "$LOG_PATH" ]; then
    echo "Setting up environment..."
    mkdir results
    mkdir models
    mkdir plots
    mkdir data
    mkdir log
fi

pip install -r requirements.txt