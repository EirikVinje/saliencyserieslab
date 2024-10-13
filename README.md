This repository holds the code for the IKT463 (AI in Society) project.

# Setup

#### Aeon installation
```bash
conda create --name aeon python==3.10
pip install tensorflow[and-cuda]
pip install -U aeon
```


# Datasets

## InsectSound

### Download

```bash

cd ~/projects/project_ikt463
mkdir data
cd data
wget https://www.timeseriesclassification.com/aeon-toolkit/InsectSound.zip
unzip -j InsectSound.zip

```



### class names

- Aedes_female: 0
- Aedes_male: 1
- Fruit_flies: 2
- House_flies: 3
- Quinx_female: 4
- Quinx_male: 5
- Stigma_female: 6
- Stigma_male: 7
- Tarsalis_female: 8
- Tarsalis_male: 9

## ECG

- download link : https://www.timeseriesclassification.com/description.php?Dataset=ECG
