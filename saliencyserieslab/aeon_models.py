import argparse
import logging
import pickle
import json
import os

from sklearn.metrics import accuracy_score, classification_report

from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.deep_learning import ResNetClassifier

logger = logging.getLogger('src')


def resnet(traindata, evaldata):

    model = ResNetClassifier(verbose=True, n_epochs=25, batch_size=128, random_state=42)
    logger.info("Running {}".format(model.__class__.__name__))

    train_x, train_y, labels = traindata['x'], traindata['y'], traindata['labels']
    
    model.n_classes_ = labels

    model.fit(train_x, train_y)

    eval_x, eval_y = evaldata['x'], evaldata['y']

    predictions = model.predict(eval_x)

    accuracy = accuracy_score(eval_y, predictions)
    print(f'Accuracy: {accuracy}')
    print()
    
    report = classification_report(eval_y, predictions, target_names=labels, output_dict=True)

    for label, item in list(report.items())[:10]:
        print(f'{label} : precision : {item["precision"]}')

def rocket(traindata, evaldata):

    model = RocketClassifier()
    logger.info("Running {}".format(model.__class__.__name__))

    train_x, train_y, labels = traindata['x'], traindata['y'], traindata['labels']
    
    model.n_classes_ = labels

    model.fit(train_x, train_y)

    eval_x, eval_y = evaldata['x'], evaldata['y']

    predictions = model.predict(eval_x)

    accuracy = accuracy_score(eval_y, predictions)
    print(f'Accuracy: {accuracy}')
    print()
    print()

    report = classification_report(eval_y, predictions, target_names=labels, output_dict=True)

    print(report)


if __name__ == '__main__':
    
    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./modelconfigs/modelconfig.json', help='Path to config file')
    args = parser.parse_args()
    config_path = args.config

    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Running {}".format(__file__))

    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f'Loaded config from {config_path}')

    with open(config['trainpath'], 'rb') as f:
        traindata = pickle.load(f)
    logger.info(f'Loaded train data from {config["trainpath"]}')

    with open(config['testpath'], 'rb') as f:
        evaldata = pickle.load(f)
    logger.info(f'Loaded eval data from {config["testpath"]}')

    # rocket(traindata, evaldata)
    resnet(traindata, evaldata)