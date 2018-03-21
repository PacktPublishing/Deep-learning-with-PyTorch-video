import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset


label_idx = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}


class IrisDataset(Dataset):

    def __init__(self, data):
        self.data = data
           
    def __getitem__(self, index):
        item = self.data.iloc[index].values
        return (item[0:4].astype(np.float32), item[4].astype(np.int))

    def __len__(self):
        return self.data.shape[0]


def get_datasets(iris_file, train_ratio=0.80):

    labels = {'class': label_idx}
    data = pd.read_csv(iris_file)
    data.replace(labels, inplace=True)

    train_df = data.sample(frac=train_ratio, random_state=3)
    test_df = data.loc[~data.index.isin(train_df.index), :]

    return IrisDataset(train_df), IrisDataset(test_df)

