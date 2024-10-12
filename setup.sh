#!/bin/bash -e

if [ "$(basename "$PWD")" != "saliencyserieslab" ]; then
    echo "Error: You are not in the project root directory (saliencyserieslab)."
    echo "Please navigate to the correct directory and run this script again."
    exit 1
fi

# check if the path "data/insectsound" exists and exit if it does
if [ -d "data/insectsound" ]; then
    echo "setup is already done"
    exit 1
fi

echo "Setting up environment..."
mkdir results
mkdir models
mkdir plots
mkdir data
mkdir log

echo "Downloading the InsectSound dataset..."

cd data
wget https://www.timeseriesclassification.com/aeon-toolkit/InsectSound.zip
unzip InsectSound.zip
mkdir insectsound
mv InsectSound/* insectsound
rm -rf InsectSound.zip
rm -rf InsectSound

echo "Installing dependencies..."

pip install -r requirements.txt