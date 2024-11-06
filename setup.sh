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

DATA_PATH="$(pwd)/data/ECG200"
if [ ! -d "$DATA_PATH" ]; then
    rm -rf data/InsectSound.zip
    echo "Downloading ECG200 dataset..."
    wget -P data http://www.timeseriesclassification.com/aeon-toolkit/ECG200.zip
    mkdir -p data/ECG200
    unzip -d data/ECG200 data/ECG200.zip
    python utils/format_dataset.py --dataset ECG200
    rm -rf data/ECG200.zip
    rm -rf data/ECG200/*.ts
    rm -rf data/ECG200/*.txt
fi


