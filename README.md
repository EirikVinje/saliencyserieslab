This repository holds the code for the IKT463 project.

### Evaluating post-hoc XAI methods for Time-Series Classification: Faithfulness and Interpretability Assessment Using the AMEE Frame

#### Installation
```bash
pip install e.
```

#### Train Models
- Specify models to train and dataset to train on.
```bash
./train.sh
```

#### Generate Explanations
- Specify explainer to use and model to explain
```bash
./generate.sh
```

#### Run recommender
- Specify explainers to rank based on explanations
```bash
python saliencyserieslab/recommender.py
```
