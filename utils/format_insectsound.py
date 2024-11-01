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
        train_outpath : str, 
        test_outpath : str, 
        train_size : float = 0.8,
        test_size : float = 0.2,
        n_classes : int = 10, 
        ):

    class_names = [
                'Aedes_female',
                'Aedes_male',
                'Fruit_flies',
                'House_flies',
                'Quinx_female',
                'Quinx_male',
                'Stigma_female',
                'Stigma_male',
                'Tarsalis_female',
                'Tarsalis_male'
                ]
    
    class_names = class_names[:n_classes]
    
    print("Generating train-test split : ({},{}) with {} classes".format(train_size, test_size, n_classes))

    rawdata = loadarff(open(inpath, 'r'))
    rawdata = [list(d) for d in rawdata[0]]

    data_x = [d[:-1] for d in rawdata]
    data_x = np.array(data_x, dtype=np.float32)

    data_y = [d[-1] for d in rawdata]
    data_y = np.array(list((map(lambda x: x.decode('utf-8'), data_y))))
    
    x_formated = []
    y_formated = []

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

    train_x, test_x, train_y, test_y = train_test_split(x_formated, y_formated, test_size=test_size, train_size=train_size, random_state=42)

    with open(train_outpath, 'wb') as f:
        pickle.dump({'x': train_x, 'y': train_y, 'labels': class_names}, f)

    with open(test_outpath, 'wb') as f:
        pickle.dump({'x': test_x, 'y': test_y, 'labels': class_names}, f)


if __name__ == "__main__":

    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes to include in the dataset, defualt and max is 10')
    parser.add_argument('--trainsize', type=float, default=0.7, help='size of the testset, default is 0.2')
    parser.add_argument('--testsize', type=float, default=0.3, help='size of the testset, default is 0.2')
    args = parser.parse_args()

    n_classes = args.n_classes
    train_size = args.trainsize
    test_size = args.testsize
    
    in_path = "./data/insectsound/InsectSound.arff"
    train_out = "./data/insectsound/insectsound_train_n{}.pkl".format(n_classes)
    test_out = "./data/insectsound/insectsound_test_n{}.pkl".format(n_classes)

    format_dataset(in_path, train_out, test_out, train_size, test_size, n_classes)




