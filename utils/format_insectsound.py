from typing import List
import pickle
import os

from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.io.arff import loadarff


def format_dataset(inpath : str, outpath : str):

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
    
    rawdata = loadarff(open(inpath, 'r'))
    rawdata = [list(d) for d in rawdata[0]]

    data_x = [d[:-1] for d in rawdata]
    data_x = np.array(data_x, dtype=np.float32)

    data_y = [d[-1] for d in rawdata]
    data_y = list(map(lambda x: x.decode('utf-8'), data_y))
    data_y = np.array([class_names.index(y) for y in data_y])

    data = {"x": data_x, "y": data_y, "labels": class_names}
    
    with open(outpath, 'wb') as f:
        pickle.dump(data, f)
    


if __name__ == "__main__":

    if not os.path.isfile('./setup.sh'):
        raise RuntimeError('Please run this script in the root directory of the repository')

    in_trainpath = "./data/insectsound/InsectSound_TRAIN.arff"
    out_trainpath = "./data/insectsound/insectsound_train.pkl"

    in_testpath = "./data/insectsound/InsectSound_TEST.arff"
    out_testpath = "./data/insectsound/insectsound_test.pkl"

    format_dataset(in_trainpath, out_trainpath)
    format_dataset(in_testpath, out_testpath)