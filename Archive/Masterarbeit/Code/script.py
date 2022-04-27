import pandas as pd
from natsort import natsorted
from glob import glob
import os

if __name__ == "__main__":
    images = pd.DataFrame(natsorted(glob(os.path.join("/export/data/compvis/kinetics/train/", "*/*"))), columns=['path'])
    labels = images['path'].apply(lambda x: x.split('/')[6]).unique().tolist()
    images['label'] = images['path'].apply(lambda x: x.split('/')[6]).apply(lambda x: labels.index(x))
    images.to_csv('train.csv', index=False, header=False, sep=' ')