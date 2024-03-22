import json
import os
import pickle
import time
import warnings
from math import ceil, sqrt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')


# String splitter
def string_split(str_for_split):
    str_no_space = str_for_split.replace(' ', '')
    str_split = str_no_space.split(',')
    value_list = [eval(x) for x in str_split]

    return value_list


# Dot dictionary
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Metrics
def RSE(pred, true):
    """
    Calculates relative quared error.
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    """
    Calculates correlation coefficient.
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """
    Calculates mean absolute error.
    """
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """
    Calculates mean squared error.
    """
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """
    Calculates root mean suared error.
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """
    Calculates mean absolute percentage error.
    """
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    """
    Calculates mean squared percentage error.
    """
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    """
    Wraps up metric functions, calculates and returns all.
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


# Standardscaling
class StandardScaler():
    """
    Straightforward StandardScaler that can handle Pytorch tensors.
    Methods included are '.fit()', '.transform()', '.inverse_transform().'
    """
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


class Dataset_ETT_hour(Dataset):
    def __init__(
        self,
        root_path,
        data_path="ETTh1.csv",
        flag="train",
        size=None,
        data_split = [0.7, 0.1, 0.2],
        scale=True,
        inverse=False,
        scale_statistic=None,
    ):
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train":0, "val":1, "test":2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.data_split:
            if (self.data_split[0] > 1):
                train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
            else:
                train_num = int(len(df_raw)*self.data_split[0]);
                test_num = int(len(df_raw)*self.data_split[2])
                val_num = len(df_raw) - train_num - test_num;
            border1s = [0, train_num - self.seq_len, train_num + val_num - self.seq_len]
            border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        else:
            border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
            border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean = self.scale_statistic["mean"], std = self.scale_statistic["std"])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        if self.inverse:
        ## Then data y is defined from unscaled values
            self.data_y = df_data[border1:border2]
        else:
            # Include scaled data as y
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_WIND_hour(Dataset):
    """
    PyTorch dataloader class for an hourly dataset that is an adaptation of the original
    Dataset_MTS in Crossformer combined with hourly dataset class from Informer.

    New ARG:
        data_split: a list of either [0.7, 0.1, 0.2] ratios or absolute numbers
                    [12*30*24, 4*30*24, 4*30*24].I f not given, the function just
                    defaults to the way Informer and Autoformer hourly data loaders make
                    the split.
    """

    def __init__(
        self,
        root_path,
        data_path="DEWINDh_small.csv",
        flag="train",
        size=None,
        data_split = [0.7, 0.1, 0.2],
        scale=True,
        inverse=False,
        scale_statistic=None,
    ):
        # NOTE Renamed in_len and out_len to seq_len and pred_len to use same notation
        if size == None:
           self.seq_len = 24*4*4
           self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.inverse = inverse  # NOTE added this
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        # If the data_split argument is given
        if self.data_split:
            # Check whether data split is given in sequence lengths
            if (self.data_split[0] > 1):
                train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
            # Or as ratios
            else:
                train_num = int(len(df_raw)*self.data_split[0]);
                test_num = int(len(df_raw)*self.data_split[2])
                val_num = len(df_raw) - train_num - test_num;
            # Define borders
            border1s = [0, train_num - self.seq_len, train_num + val_num - self.seq_len]
            border2s = [train_num, train_num+val_num, train_num + val_num + test_num]
        else:
            border1s = [0, 18 * 30 * 24 - self.seq_len, 18 * 30 * 24 + 3 * 30 * 24 - self.seq_len]
            border2s = [18 * 30 * 24, 18 * 30 * 24 + 3 * 30 * 24, 18 * 30 * 24 + 6 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        # Crossformer scaling has an additional functionality here where you can specify
        # what mean and std you want to scale to
        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean = self.scale_statistic['mean'], std = self.scale_statistic['std'])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        # NOTE added inverse possibility
        if self.inverse:
            ## Then data y is defined from unscaled values
            self.data_y = df_data[border1:border2]
        else:
            # Include scaled data as y
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_SYNTH_hour(Dataset):
    """
    PyTorch dataloader class for an hourly dataset that is an adaptation of the original
    Dataset_MTS in Crossformer combined with hourly dataset class from Informer.

    New ARG:
        data_split: a list of either [0.7, 0.1, 0.2] ratios or absolute numbers
                    [12*30*24, 4*30*24, 4*30*24]. If not given, the function just
                    defaults to the way Informer and Autoformer hourly data loaders
                    make the split.
    """

    def __init__(
            self,
            root_path,
            data_path="SYNTHh1.csv",
            flag="train",
            size=None,
            data_split = [0.7, 0.1, 0.2],
            scale=True,
            inverse=False,
            scale_statistic=None,
        ):
        if size == None:
           self.seq_len = 24*4*4
           self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train":0, "val":1, "test":2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.data_split:
            if (self.data_split[0] > 1):
                train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
            else:
                train_num = int(len(df_raw)*self.data_split[0]);
                test_num = int(len(df_raw)*self.data_split[2])
                val_num = len(df_raw) - train_num - test_num;
            # Define borders
            border1s = [0, train_num - self.seq_len, train_num + val_num - self.seq_len]
            border2s = [train_num, train_num+val_num, train_num + val_num + test_num]
        else:
            border1s = [0, 18 * 30 * 24 - self.seq_len, 18 * 30 * 24 + 3 * 30 * 24 - self.seq_len]
            border2s = [18 * 30 * 24, 18 * 30 * 24 + 3 * 30 * 24, 18 * 30 * 24 + 6 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean = self.scale_statistic["mean"], std = self.scale_statistic["std"])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data[border1:border2]
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_SYNTH_additive(Dataset):
    """
    PyTorch dataloader class for an hourly dataset that is an adaptation of the original
    Dataset_MTS in Crossformer combined with hourly dataset class from Informer.

    New ARG:
        data_split: a list of either [0.7, 0.1, 0.2] ratios or absolute numbers
                    [12*30*24, 4*30*24, 4*30*24]. If not given, the function just
                    defaults to the way Informer and Autoformer hourly data loaders make
                    the split.
    """

    def __init__(
        self,
        root_path,
        data_path="SYNTH_additive.csv",
        flag="train",
        size=None,
        data_split = [0.7, 0.1, 0.2],
        scale=True,
        inverse=False,
        scale_statistic=None,
    ):

        if size == None:
           self.seq_len = 24*4*4
           self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train":0, "val":1, "test":2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.data_split:
            if (self.data_split[0] > 1):
                train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
            else:
                train_num = int(len(df_raw)*self.data_split[0]);
                test_num = int(len(df_raw)*self.data_split[2])
                val_num = len(df_raw) - train_num - test_num;
            border1s = [0, train_num - self.seq_len, train_num + val_num - self.seq_len]
            border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        else:
            border1s = [0, 18 * 30 * 24 - self.seq_len, 18 * 30 * 24 + 3 * 30 * 24 - self.seq_len]
            border2s = [18 * 30 * 24, 18 * 30 * 24 + 3 * 30 * 24, 18 * 30 * 24 + 6 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean = self.scale_statistic['mean'], std = self.scale_statistic['std'])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        #### B.A. - Added self.inverse
        if self.inverse:
            ## Then data y is defined from unscaled values
            self.data_y = df_data[border1:border2]
        else:
            # Include scaled data as y
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_SYNTH_additive_reversal(Dataset):
    """
    PyTorch dataloader class for an hourly dataset that is an adaptation of the original
    Dataset_MTS in Crossformer combined with hourly dataset class from Informer.
    """

    def __init__(
            self,
            root_path,
            data_path="SYNTH_additive_reversals.csv",
            flag="train",
            size=None,
            data_split = [0.7, 0.1, 0.2],
            scale=True,
            inverse=False,
            scale_statistic=None
        ):

        if size == None:
           self.seq_len = 24*4*4
           self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train":0, "val":1, "test":2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.data_split:
            if (self.data_split[0] > 1):
                train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
            else:
                train_num = int(len(df_raw)*self.data_split[0]);
                test_num = int(len(df_raw)*self.data_split[2])
                val_num = len(df_raw) - train_num - test_num;
            border1s = [0, train_num - self.seq_len, train_num + val_num - self.seq_len]
            border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        else:
            border1s = [0, 18 * 30 * 24 - self.seq_len, 18 * 30 * 24 + 3 * 30 * 24 - self.seq_len]
            border2s = [18 * 30 * 24, 18 * 30 * 24 + 3 * 30 * 24, 18 * 30 * 24 + 6 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean = self.scale_statistic["mean"], std = self.scale_statistic["std"])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data[border1:border2]
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_SYNTH_multiplicative(Dataset):
    """
    PyTorch dataloader class for an hourly dataset that is an adaptation of the original
    Dataset_MTS in Crossformer combined with hourly dataset class from Informer.
    """

    def __init__(
        self,
        root_path,
        data_path="SYNTH_multiplicative.csv",
        flag="train",
        size=None,
        data_split = [0.7, 0.1, 0.2],
        scale=True,
        inverse=False,
        scale_statistic=None
    ):

        if size == None:
           self.seq_len = 24*4*4
           self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train":0, "val":1, "test":2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.data_split:
            if (self.data_split[0] > 1):
                train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
            else:
                train_num = int(len(df_raw)*self.data_split[0]);
                test_num = int(len(df_raw)*self.data_split[2])
                val_num = len(df_raw) - train_num - test_num;
            border1s = [0, train_num - self.seq_len, train_num + val_num - self.seq_len]
            border2s = [train_num, train_num+val_num, train_num + val_num + test_num]
        else:
            border1s = [0, 18 * 30 * 24 - self.seq_len, 18 * 30 * 24 + 3 * 30 * 24 - self.seq_len]
            border2s = [18 * 30 * 24, 18 * 30 * 24 + 3 * 30 * 24, 18 * 30 * 24 + 6 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean = self.scale_statistic["mean"], std = self.scale_statistic["std"])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]

        if self.inverse:
            self.data_y = df_data[border1:border2]
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_SYNTH_multiplicative_reversal(Dataset):
    """
    PyTorch dataloader class for an hourly dataset that is an adaptation of the original
    Dataset_MTS in Crossformer combined with hourly dataset class from Informer.
    """

    def __init__(
        self,
        root_path,
        data_path="SYNTH_multiplicative_reversals.csv",
        flag="train",
        size=None,
        data_split = [0.7, 0.1, 0.2],
        scale=True,
        inverse=False,
        scale_statistic=None
    ):

        if size == None:
           self.seq_len = 24*4*4
           self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train":0, "val":1, "test":2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.data_split:
            if (self.data_split[0] > 1):
                train_num = self.data_split[0]; val_num = self.data_split[1]; test_num = self.data_split[2];
            else:
                train_num = int(len(df_raw)*self.data_split[0]);
                test_num = int(len(df_raw)*self.data_split[2])
                val_num = len(df_raw) - train_num - test_num;
            border1s = [0, train_num - self.seq_len, train_num + val_num - self.seq_len]
            border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        else:
            border1s = [0, 18 * 30 * 24 - self.seq_len, 18 * 30 * 24 + 3 * 30 * 24 - self.seq_len]
            border2s = [18 * 30 * 24, 18 * 30 * 24 + 3 * 30 * 24, 18 * 30 * 24 + 6 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean = self.scale_statistic["mean"], std = self.scale_statistic["std"])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]

        if self.inverse:
            self.data_y = df_data[border1:border2]
        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# Experiment class =====================================================================
class Exp_Basic(object):
    """
    Parent class for fitting and testing. The actual model training class will inherit
    from this class.
    """

    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            # if not self.args.use_multi_gpu else self.args.devices
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

# MODEL COMPONENTS =====================================================================
# Data Embedding
class DSW_embedding(nn.Module):
    """
    Crossformer's Dimension-Segment-Wise 2D embedding module.
    """

    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)

        return x_embed


# Attention Mechanisms
class FullAttention(nn.Module):
    """
    The Vanilla Attention operation
    """
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    """
    The Multi-head Self-Attention (MSA) Layer
    """
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = out.view(B, L, -1)

        return self.out_projection(out)


class TwoStageAttentionLayer(nn.Module):
    """
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    """
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff = None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))

    def forward(self, x):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and
        # distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b = batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat = batch)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b = batch)

        return final_out


# Encoder
class SegMerging(nn.Module):
    """
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment
    to get representation of a coarser scale we set win_size = 2 in our paper
    """

    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim = -2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)

        return x


class scale_block(nn.Module):
    """
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper
    """

    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, \
                    seg_num = 10, factor=10):
        super(scale_block, self).__init__()

        if (win_size > 1):
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, \
                                                        d_ff, dropout))

    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)

        return x


class Encoder(nn.Module):
    """
    The Encoder of Crossformer.
    """

    def __init__(self, e_blocks, win_size, d_model, n_heads, d_ff, block_depth, dropout,
                in_seg_num = 10, factor=10):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(scale_block(1, d_model, n_heads, d_ff, block_depth, dropout,\
                                            in_seg_num, factor))
        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(win_size, d_model, n_heads, d_ff, block_depth, dropout,\
                                            ceil(in_seg_num/win_size**i), factor))

    def forward(self, x):
        encode_x = []
        encode_x.append(x)

        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x


# Decoder Layer
class DecoderLayer(nn.Module):
    """
    The decoder layer of Crossformer, each layer will make a prediction at its scale
    """

    def __init__(
        self,
        seg_len,
        d_model,
        n_heads,
        d_ff=None,
        dropout=0.1,
        out_seg_num = 10,
        factor = 10,
    ):
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayer(out_seg_num, factor, d_model, n_heads, \
                                d_ff, dropout)
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.GELU(),
                                nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        """
        x: the output of last decoder layer
        cross: the output of the corresponding encoder layer
        """
        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')

        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        tmp = self.cross_attention(x, cross, cross,)
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x+y)

        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b = batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')

        return dec_output, layer_predict


class Decoder(nn.Module):
    """
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    """

    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout,\
                router=False, out_seg_num = 10, factor=10):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(DecoderLayer(seg_len, d_model, n_heads, d_ff, dropout, \
                                        out_seg_num, factor))

    def forward(self, x, cross):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x,  cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1

        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d = ts_d)

        return final_predict


# Enitre crossformer
class Crossformer(nn.Module):

    def __init__(self, data_dim, seq_len, pred_len, seg_len, win_size = 2,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3,
                dropout=0.0, baseline = False, device=torch.device('cuda:0')):
        super(Crossformer, self).__init__()
        self.data_dim = data_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.seg_len = seg_len
        self.merge_win = win_size

        self.baseline = baseline

        self.device = device

        # The padding operation to handle invisible sgemnet length
        # Again switched in/out convention to seq/pred for consistency
        self.pad_seq_len = ceil(1.0 * seq_len / seg_len) * seg_len
        self.pad_pred_len = ceil(1.0 * pred_len / seg_len) * seg_len
        self.seq_len_add = self.pad_seq_len - self.seq_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_seq_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_seq_len // seg_len), factor = factor)

        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_pred_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, \
                                    out_seg_num = (self.pad_pred_len // seg_len), factor = factor)

    def forward(self, x_seq):
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if (self.seq_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.seq_len_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)

        return base + predict_y[:, :self.pred_len, :]


# Learning Rate Decay
def adjust_learning_rate(optimizer, epoch, args):
    """
    Helper function for learning rate decay
    """
    if args.lradj=='type1':
        lr_adjust = {2: args.learning_rate * 0.5 ** 1, 4: args.learning_rate * 0.5 ** 2,
                     6: args.learning_rate * 0.5 ** 3, 8: args.learning_rate * 0.5 ** 4,
                     10: args.learning_rate * 0.5 ** 5}
    elif args.lradj=='type2':
        lr_adjust = {5: args.learning_rate * 0.5 ** 1, 10: args.learning_rate * 0.5 ** 2,
                     15: args.learning_rate * 0.5 ** 3, 20: args.learning_rate * 0.5 ** 4,
                     25: args.learning_rate * 0.5 ** 5}
    else:
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


# Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss


# Running Experiments
class Exp_crossformer(Exp_Basic):
    """
    Exp_Crossformer is the main class for the Crossformer architecture, which wraps up
    every component above.
    """
    def __init__(self, args):
        super(Exp_crossformer, self).__init__(args)
    # in/out convention changed to seq_len, pred_len
    def _build_model(self):
        model = Crossformer(
            self.args.data_dim,
            self.args.seq_len,
            self.args.pred_len,
            self.args.seg_len,
            self.args.win_size,
            self.args.factor,
            self.args.d_model,
            self.args.d_ff,
            self.args.n_heads,
            self.args.e_layers,
            self.args.dropout,
            self.args.baseline,
            self.device
        ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args


        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'SYNTHh1': Dataset_SYNTH_hour,
            'SYNTHh2': Dataset_SYNTH_hour,
            'SYNTH_additive': Dataset_SYNTH_additive,
            'SYNTH_additive_reversal': Dataset_SYNTH_additive_reversal,
            'SYNTH_multiplicative': Dataset_SYNTH_multiplicative,
            'SYNTH_multiplicative_reversal': Dataset_SYNTH_multiplicative_reversal,
            'DEWINDh_small': Dataset_WIND_hour,
        }

        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size;
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size;

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            inverse=args.inverse,
            data_split = args.data_split,
        )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        """
        Sets up Adam optimizer with specified learning rate.
        """
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        Sets up MSE loss.
        """
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        scale_statistic = {'mean': train_data.scaler.mean, 'std': train_data.scaler.std}
        with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
            pickle.dump(scale_statistic, f)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)

        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'checkpoint.pth')

        return self.model

    def test(self, setting, save_pred=True, inverse=False):
        """
        Main method for testing model following training.
        We added a line to be able to call the first batch of data from the test set for
        possible trouble shooting.
        We also added returns to the testing function.
        """
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []
        metrics_all = []
        instance_num = 0

        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                # NOTE for possible debug
                first_batch_test = {
                    'batch_x': batch_x,
                    'batch_y': batch_y
                }

                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, inverse)
                ### Changed if save preds logic to info/auto logic
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)


        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        # Explicitly define all metrics
        all_metrics = np.array([mae, mse, rmse, mape, mspe])

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)

        # Added returns
        return preds, trues, mse, mae, mape, all_metrics, first_batch_test

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse=False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)

        return outputs, batch_y

    # eval method
    def eval(self, setting, save_pred=False, inverse=False):
        # evaluate a saved model
        args = self.args
        # NOTE added data_dict
        data_dict = {
            "ETTh1": Dataset_ETT_hour,
            "SYNTHh1": Dataset_SYNTH_hour,
            "SYNTHh2": Dataset_SYNTH_hour,
            "SYNTH_additive": Dataset_SYNTH_additive,
            "SYNTH_additive_reversal": Dataset_SYNTH_additive_reversal,
            "SYNTH_multiplicative": Dataset_SYNTH_multiplicative,
            "SYNTH_multiplicative_reversal": Dataset_SYNTH_multiplicative_reversal,
            "DEWINDh_small": Dataset_WIND_hour
        }

        # NOTE added the Data variable
        Data = data_dict[self.args.data]  #
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag="test",
            size=[args.seq_len, args.pred_len],
            data_split=args.data_split,
            scale=True,
            scale_statistic=args.scale_statistic,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

        self.model.eval()

        preds = []
        trues = []
        metrics_all = []
        instance_num = 0

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
                pred, true = self._process_one_batch(
                    data_set, batch_x, batch_y, inverse
                )
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = (
                    np.array(
                        metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
                    )
                    * batch_size
                )
                metrics_all.append(batch_metric)
                if save_pred:
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis=0)
        metrics_mean = metrics_all.sum(axis=0) / instance_num

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print("mse:{}, mae:{}".format(mse, mae))

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        if save_pred:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            np.save(folder_path + "pred.npy", preds)
            np.save(folder_path + "true.npy", trues)

        return mae, mse, rmse, mape, mspe


# Our Simple User Interface ============================================================
class CrossformerTS:
    """
    Our custom wrapper class (in progress) to provide an user-friendly interface to
    fitting and testing the Crossformer model.
    For simplicity of use, methods included align with naming used by Keras.
    """

    def __init__(self, model="crossformer"):
        if model != "crossformer":
            raise ValueError("Model not supported. Please use 'crossformer'.")
        # Initialize dot dictionary
        self.args = dotdict()
        self.args.checkpoints = "./checkpoints/"
        self.args.seq_len = 168
        # Segment length
        self.args.seg_len = 6
        # Window size for segment merge
        self.args.win_size = 2
        # Number of routers in cross-dimensions tage of TSA
        self.args.factor = 10
        self.args.data_dim = 1
        # args.data_split = '0.7, 0.1, 0.2' # NOTE activate for ratio-based split
        self.args.d_model = 256
        self.args.d_ff = 512
        self.args.n_heads = 4
        self.args.e_layers = 3
        self.args.dropout = 0.2
        self.args.baseline = False
        self.args.num_workers = 0
        self.args.lradj = "type1"
        self.args.itr = 3
        self.args.save_pred = True
        self.args.use_gpu = True
        self.args.use_multi_gpu = False
        self.args.gpu = 0
        self.args.devices = "0,1,2,3"
        self.args.inverse = False

        self.args.use_gpu = (
            True if torch.cuda.is_available() and self.args.use_gpu else False
        )
        if self.args.use_gpu and self.args.use_multi_gpu:
            self.args.devices = self.args.devices.replace(" ", "")
            device_ids = self.args.devices.split(",")
            self.args.device_ids = [int(id_) for id_ in device_ids]
            self.args.gpu = self.args.device_ids[0]
            print(self.args.gpu)

        self.data_parser = {
            "ETTh1": {
                "data": "ETTh1.csv",
                "data_dim": 7,
                "split": [12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24],
            },
            "SYNTHh1": {
                "data": "SYNTHh1.csv",
                "data_dim": 1,
                "split": [18 * 30 * 24, 3 * 30 * 24, 3 * 30 * 24],
            },
            "SYNTH_additive": {
                "data": "SYNTH_additive.csv",
                "data_dim": 1,
                "split": [18 * 30 * 24, 3 * 30 * 24, 3 * 30 * 24],
            },
            "SYNTH_additive_reversal": {
                "data": "SYNTH_additive_reversal.csv",
                "data_dim": 1,
                "split": [18 * 30 * 24, 3 * 30 * 24, 3 * 30 * 24],
            },
            "SYNTH_multiplicative": {
                "data": "SYNTH_multiplicative.csv",
                "data_dim": 1,
                "split": [18 * 30 * 24, 3 * 30 * 24, 3 * 30 * 24],
            },
            "SYNTH_multiplicative_reversal": {
                "data": "SYNTH_multiplicative_reversal.csv",
                "data_dim": 1,
                "split": [18 * 30 * 24, 3 * 30 * 24, 3 * 30 * 24],
            },
            "DEWINDh_small": {
                "data": "DEWINDh_small.csv",
                "data_dim": 1,
                "split": [18 * 30 * 24, 3 * 30 * 24, 3 * 30 * 24],
            },
        }

    def compile(self, learning_rate=1e-4, loss="mse", early_stopping_patience=3):
        """
        Compiles the crossformer model for training.
        Args:
            learning_rate (float): Learning rate to be used. Default: '1e-4'.
            loss (str): Loss function to be used. Default: 'mse'.
            early_stopping_patience (int): Amount of epochs to beak training loop after
                                           no validation performance improvement.
                                           Default: 3.
        """
        if loss != "mse":
            raise ValueError("Loss function not supported. Please use 'mse'.")
        self.args.learning_rate = learning_rate
        self.args.loss = loss
        self.args.patience = early_stopping_patience

    def fit(
        self,
        data="SYNTHh1",
        data_root_path="./SYNTHDataset/",
        batch_size=32,
        epochs=20,
        pred_len=24,
        seq_len=168,
        iter=1,
    ):
        """
        Fits the crossformer model.
        Args:
            data (str): Name of the dataset used. For now, only 'SYNTHh1', 'SYNTHh2',
                        'DEWINDh_large' and 'DEWINDh_small' are supported.
                        Default: 'SYNTHh1'.
            data_root_path (str): Root folder for given dataset.
                                  Default: './SYNTHDataset'.
            batch_size (int): Batch size. Default: 32.
            epochs (int): Number of epochs for training the model. Default: 8.
            pred_len (int): Prediction window length. Default: 24.
        """
        # temporary line
        possible_datasets = [
            "SYNTHh1",
            "SYNTHh2",
            "SYNTH_additive",
            "SYNTH_additive_reversal",
            "SYNTH_multiplicative",
            "SYNTH_multiplicative_reversal",
            "DEWINDh_large",
            "DEWINDh_small",
            "ETTh1",
        ]
        if data not in possible_datasets:
            raise ValueError(
                "Dataset not supported. Please use one of the following: "
                "'SYNTHh1', 'SYNTHh2', 'SYNTH_additive', 'SYNTH_additive_reversal', "
                "'SYNTH_multiplicative', 'SYNTH_multiplicative_reversal', "
                "'DEWINDh_large', 'DEWINDh_small' ,'ETTh1'."
            )
        # temporary line
        possible_predlens = [24, 48, 96, 168, 336, 720]
        if pred_len not in possible_predlens:
            raise ValueError(
                "Prediction length outside current experiment scope. Please use either "
                "24, 48, 96, 168, 336, 720."
            )
        self.args.data = data
        self.args.root_path = data_root_path
        self.args.data_path = f"{self.args.data}.csv"
        self.args.train_epochs = epochs
        self.args.batch_size = batch_size
        self.args.pred_len = pred_len
        self.args.seq_len = seq_len
        self.args.iter = iter

        if self.args.data in self.data_parser.keys():
            self.data_info = self.data_parser[self.args.data]
            self.args.data_path = self.data_info["data"]
            self.args.data_dim = self.data_info["data_dim"]
            self.args.data_split = self.data_info["split"]
        else:
            self.args.data_split = string_split(self.args.data_split)

        print("Beginning to fit the model with the following arguments:")
        print(f"{self.args}")
        print("=" * 150)

        Experiment_Model = Exp_crossformer
        self.setting = "Crossformer_{}_il{}_pl{}_sl{}_win{}_fa{}_dm{}_nh{}_el{}_iter{}".format(
            self.args.data,
            self.args.seq_len,
            self.args.pred_len,
            self.args.seg_len,
            self.args.win_size,
            self.args.factor,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.iter,
        )
        # Initialize model class
        self.experiment_model = Experiment_Model(self.args)
        # Train
        print(
            ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(self.setting)
        )
        self.model = self.experiment_model.train(self.setting)

    def predict(self):
        """
        Makes predictions on pre-defined test set. Does not require any arguments.
        Returns:
            preds: A 3D array of predictions of the following shape
                   (number of windows, number of time points per window, number of targets.)
                   As self variables, trues, mse, mae, all_metrics, and first_batch_test
                   can also be called.
        """
        # Predict
        (
            self.preds,
            self.trues,
            self.mse,
            self.mae,
            self.mape,
            self.all_metrics,
            self.first_batch_test,
        ) = self.experiment_model.test(self.setting)
        # Clear memory
        torch.cuda.empty_cache()

        return self.preds
