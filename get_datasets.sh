#!/bin/bash -e

if ! command -v unzip >/dev/null 2>&1; then
    echo "unzip is not installed. You can install it by running 'sudo apt install unzip'."
    exit 1
fi

if [ "$(basename "$PWD")" != "saliencyserieslab" ]; then
    echo "Error: You are not in the project root directory (saliencyserieslab)."
    echo "Please navigate to the correct directory and run this script again."
    exit 1
fi

LOG_PATH="$(pwd)/log"
if [ ! -d "$LOG_PATH" ]; then
    echo "Setting up environment..."
    mkdir results
    mkdir plots
    mkdir data
    mkdir log
fi

DATA_PATH="$(pwd)/data/insectsound"
if [ ! -d "$DATA_PATH" ]; then
    rm -rf data/InsectSound.zip
    echo "Downloading the InsectSound dataset..."
    wget -P data https://www.timeseriesclassification.com/aeon-toolkit/InsectSound.zip
    unzip -d data data/InsectSound.zip
    rm -rf data/InsectSound.zip
    mkdir data/insectsound
    mv data/InsectSound/* data/insectsound/
    rm -rf data/InsectSound
fi

DATA_PATH="$(pwd)/data/ecg"
if [ ! -d "$DATA_PATH" ]; then
    echo "Downloading the ECG dataset..."
    mkdir data/ecg
fi

PKL_PATH="$(pwd)/data/insectsound/insectsound_train.pkl"
if [ ! -f "$PKL_PATH" ]; then
    echo "Formatting the InsectSound dataset..."
    python utils/format_insectsound.py --n_classes 10
    # rm -rf data/insectsound/*.arff
fi

# pip install -r requirements.txt



