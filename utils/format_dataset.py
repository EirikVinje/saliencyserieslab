from typing import List
import argparse 
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.io.arff import loadarff
import numpy as np


def format_dataset(
        inpath : str,
        outpath : str, 
        ):

    rawdata = loadarff(open(inpath, 'r'))
    rawdata = [list(d) for d in rawdata[0]]

    data_x = [d[:-1] for d in rawdata]
    data_x = np.array(data_x, dtype=np.float32)

    data_y = [d[-1] for d in rawdata]
    data_y = np.array(list((map(lambda x: x.decode('utf-8'), data_y))))
    
    x_formated = []
    y_formated = []
    
    class_names = np.unique(data_y)
    
    for i, cn in enumerate(class_names):
        
        is_class = np.where(data_y == cn)[0]
        
        c_x = data_x[is_class, :]
        c_y = np.full(c_x.shape[0], i)
        
        x_formated.append(c_x)
        y_formated.append(c_y)
        
    x_formated = np.vstack(x_formated)
    y_formated = np.concatenate(y_formated)

    scaler = StandardScaler()
    x_formated = scaler.fit_transform(x_formated, y_formated)


    with open(outpath, 'wb') as f:
        pickle.dump({'x': x_formated, 'y': y_formated, 'labels': class_names}, f)


if __name__ == "__main__":

    if not os.path.isfile('./setup.py'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ECG200", help='ECG200 or InsectSound')
    args = parser.parse_args()
    
    dataset = args.dataset

    datasetdir = os.path.join('./data', dataset)

    if not os.path.isdir(datasetdir):
        raise RuntimeError("Dataset {} not found in {}".format(dataset, datasetdir))

    train_in = os.path.join(datasetdir, "{}_TRAIN.arff".format(dataset))
    test_in = os.path.join(datasetdir, "{}_TEST.arff".format(dataset))

    print("formatting {} dataset".format(dataset))
    print("train : {}".format(train_in))
    print("test : {}".format(test_in))

    train_out = os.path.join(datasetdir, "{}_train.pkl".format(dataset.lower()))
    test_out = os.path.join(datasetdir, "{}_test.pkl".format(dataset.lower()))

    print("formatting train...")
    format_dataset(train_in, train_out)
    print("formatting test...")
    format_dataset(test_in, test_out)




