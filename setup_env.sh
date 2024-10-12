    #!/bin/bash -e

    if [ "$(basename "$PWD")" != "project_ikt463" ]; then
        echo "Error: You are not in the project root directory (project_ikt463)."
        echo "Please navigate to the correct directory and run this script again."
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
    unzip -j InsectSound.zip