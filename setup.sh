#!/bin/bash -e

if [ "$(basename "$PWD")" != "saliencyserieslab" ]; then
    echo "Error: You are not in the project root directory (saliencyserieslab)."
    echo "Please navigate to the correct directory and run this script again."
    exit 1
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

DATA_PATH="$(pwd)/data/insectsound"
if [ ! -d "$DATA_PATH" ]; then
    echo "Downloading the InsectSound dataset..."
    cd data
    wget https://www.timeseriesclassification.com/aeon-toolkit/InsectSound.zip
    unzip InsectSound.zip
    mkdir insectsound
    mv InsectSound/* insectsound/
    rm -rf InsectSound.zip
    rm -rf InsectSound
    cd ..
fi

DATA_PATH="$(pwd)/data/ecg"
if [ ! -d "$DATA_PATH" ]; then
    echo "Downloading the ECG dataset..."
    cd data
    mkdir -p ecg
    cd ..
fi

pip install -r requirements.txt



