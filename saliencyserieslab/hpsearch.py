import argparse
import logging
import pickle
import gc
import os

from sklearn.metrics import accuracy_score
import optuna

from load_sktime_classifier import SktimeClassifier
from train_sktime_classifier import train
from generate_config import generate_hp_config

logger = logging.getLogger('saliencyserieslab')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def objective(trail):
    
    hpconfig = generate_hp_config(MODEL_NAME, trail)

    model = SktimeClassifier(hpconfig)
    
    model.fit(traindata['x'], traindata['y'])

    accuracy = model.evaluate(evaldata['x'], evaldata['y'])

    return accuracy
    
    
if  __name__ == "__main__":
    
    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    parser = argparse.ArgumentParser()

    parser.add_argument('--trainpath', type=str, default='./data/insectsound/insectsound_train_n10.pkl', help='Path to train data')
    parser.add_argument('--testpath', type=str, default='./data/insectsound/insectsound_test_n10.pkl', help='Path to eval data')
    parser.add_argument('--model', type=str, default="inception", help='sktime model [inception, rocket, resnet]')
    parser.add_argument('--trial', type=int, default=10, help='number of trials')
    args = parser.parse_args()

    trainpath = args.trainpath
    testpath = args.testpath
    MODEL_NAME = args.model
    
    logger.info("Running {}".format(__file__))

    with open(trainpath, 'rb') as f:
        traindata = pickle.load(f)
    logger.info(f'Loaded train data from {trainpath}')

    with open(testpath, 'rb') as f:
        evaldata = pickle.load(f)
    logger.info(f'Loaded eval data from {testpath}')

    model = SktimeClassifier()
    model.load_model(MODEL_NAME)

    study = optuna.create_study(direction='maximize', storage="sqlite:///optuna.db", study_name=f"{MODEL_NAME}_study")
    
    study.optimize(objective, n_trials=args.trial)