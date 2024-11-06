This repository holds the code for the IKT463 (AI in Society) project.

# Setup
1. Head to (UCR ARCHIVE)[https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/] and download the archive.
2. Fetch ECG200

#### Conda environment installation
```bash
conda create --name ssl python==3.10
pip install -U sktime[all_extras]
pip install tensorflow[and-cuda]
pip install dash
```
