# A script file to easily and convinently run and use the Logsparse model. Includes a wrapper class for the model that can be imported and easily used for training and prediction.

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List
import os
import time
import pickle
import math
from math import sqrt
import copy

from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings


warnings.filterwarnings('ignore')

###########################################################################################################################
# Dot dictionary ##########################################################################################################
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

###########################################################################################################################
# Metrics #################################################################################################################
    
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true, return_mean=True):
    _logits = np.abs(pred - true)
    if return_mean:
        return np.mean(_logits)
    else:
        return _logits


def MSE(pred, true, return_mean=True):
    _logits = (pred - true) ** 2
    if return_mean:
        return np.mean(_logits)
    else:
        return _logits


def RMSE(pred, true, return_mean=True):
    return np.sqrt(MSE(pred, true, return_mean=return_mean))


def MAPE(pred, true, eps=1e-07, return_mean=True):
    _logits = np.abs((pred - true) / (true + eps))
    if return_mean:
        return np.mean(_logits)
    else:
        return _logits


def MSPE(pred, true, eps=1e-07, return_mean=True):
    _logits = np.square((pred - true) / (true + eps))
    if return_mean:
        return np.mean(_logits)
    else:
        return _logits


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


###########################################################################################################################
# Masking #################################################################################################################

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class LogSparseMask():
    def __init__(self, B, Q_L, K_L, win_len=0, res_len=None, device="cpu"):
        mask_shape = [B, 1, Q_L, K_L]
        with torch.no_grad():
            if res_len is None:
                L_ls = K_L - 1 - win_len
                n_reps = 1
            else:
                n_reps = np.ceil(K_L / res_len)
                L_ls = res_len

            l_floor = int(np.log2(L_ls))
            indxs = np.array([int(2 ** (l_floor - i)) for i in range(l_floor + 1)])
            reps_array = np.expand_dims(np.arange(n_reps, dtype='int')*L_ls, 1)
            indxs = (indxs + reps_array + win_len).flatten()
            indxs = indxs[indxs < (K_L - 1)]
            my_mask = np.ones(K_L, dtype='int')
            my_mask[indxs] = 0
            my_mask[:(win_len + 1)] = 0
            my_mask = np.concatenate([np.flip(my_mask[1:]), my_mask])
            my_mask = np.array([my_mask[(K_L - i):(K_L * 2 - i)] for i in range(1, Q_L + 1)], dtype='bool')
            self._mask = torch.from_numpy(my_mask).to(device).unsqueeze(0).unsqueeze(1).expand(mask_shape)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L_Q, index, scores, L_K, top_keys=False, device="cpu"):
        _mask = torch.ones(L_Q, L_K, dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L_Q, L_K)
        if top_keys:
            indicator = _mask_ex[torch.arange(B)[:, None, None, None],
                        torch.arange(H)[None, :, None, None],
                        torch.arange(L_Q)[None, None, :, None],
                        index[:, :, None,:]].to(device)
        else:
            indicator = _mask_ex[torch.arange(B)[:, None, None],
                        torch.arange(H)[None, :, None],
                        index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


###########################################################################################################################
# Time Features #############################################################################################################
    
class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)

###########################################################################################################################
# Timestamps ##############################################################################################################

def time_features(dates, freq='h'):
    """
    > `time_features` takes in a `dates` dataframe with a 'dates' column and extracts the date down to `freq` where freq can be any of the following:
     > * Q - [month]
    > * M - [month]
    > * W - [Day of month, week of year]
    > * D - [Day of week, day of month, day of year]
    > * B - [Day of week, day of month, day of year]
    > * H - [Hour of day, day of week, day of month, day of year]
    > * T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    > * S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]
    """
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])

###########################################################################################################################
# Data loaders #################################################################################################

class Dataset_ETT_hour(Dataset):
    '''
    PyTorch dataloader class for the wind dataset, which constitutes amendments of the original Logsparse dataloader for the Wind dataset.
    Class comprehensively handles the train-val-test split, scaling, time-feature encoding. Further included (but unused at this point) method allows for reverse scaling. 
    '''
    def __init__(self, root_path, flag='train', size=None, features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse = False, timeenc=0, freq='h', **_):

        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        self.seq_len = size[0]          # S (notation used in paper)
        self.label_len = size[1]        # L
        self.pred_len = size[2]         # P

        self.total_seq_len = self.seq_len + self.pred_len
        assert self.label_len <= self.seq_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target        
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        _path = os.path.join(self.root_path, self.data_path).replace('\\', '/')  # replace in case windows
        df_raw = pd.read_csv(_path)

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Construct the time array
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 10)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError("Pass timeenc as either 0 or 1")

        self.data_stamp = data_stamp    # The full time dataset
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

        # Assumes inputs of shape [nodes, seq, feats] or [seq, feats]
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_WIND_hour(Dataset):
    '''
    PyTorch dataloader class for the wind dataset, which constitutes amendments of the original Logsparse dataloader for the Wind dataset.
    Class comprehensively handles the train-val-test split, scaling, time-feature encoding. Further included (but unused at this point) method allows for reverse scaling. 
    '''
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='DEWINDh_small.csv',
                 target='TARGET', scale=True, inverse = False, timeenc=0, freq='h', **_):

        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        self.seq_len = size[0]          # S (notation used in paper)
        self.label_len = size[1]        # L
        self.pred_len = size[2]         # P

        self.total_seq_len = self.seq_len + self.pred_len
        assert self.label_len <= self.seq_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target        
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        _path = os.path.join(self.root_path, self.data_path).replace('\\', '/')  # replace in case windows
        df_raw = pd.read_csv(_path)

        border1s = [0, 18 * 30 * 24 - self.seq_len, 18 * 30 * 24 + 3 * 30 * 24 - self.seq_len]
        border2s = [18 * 30 * 24, 18 * 30 * 24 + 3 * 30 * 24, 18 * 30 * 24 + 6 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
             #### Added error raise as dataset univariate
            raise ValueError("M and MS invalid settings. Use univariate setting 'S' for the given dataset.")
            #cols_data = df_raw.columns[df_raw.columns.get_level_values(0) != 'time']
            #df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Construct the time array
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 10)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError("Pass timeenc as either 0 or 1")

        self.data_stamp = data_stamp    # The full time dataset
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

        # Assumes inputs of shape [nodes, seq, feats] or [seq, feats]
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_SYNTH_hour(Dataset):
     def __init__(self, root_path, flag='train', size=None, features='S', data_path='SYNTHh1.csv',
                 target='TARGET', scale=True, inverse = False, timeenc=0, freq='h', **_):

        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        self.seq_len = size[0]          # S (notation used in paper)
        self.label_len = size[1]        # L
        self.pred_len = size[2]         # P

        self.total_seq_len = self.seq_len + self.pred_len
        assert self.label_len <= self.seq_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target        
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

     def __read_data__(self):
        self.scaler = StandardScaler()
        _path = os.path.join(self.root_path, self.data_path).replace('\\', '/')  # replace in case windows
        df_raw = pd.read_csv(_path)

        # Fit scaler based on training data only
        
        border1s = [0, 18 * 30 * 24 - self.seq_len, 18 * 30 * 24 + 3 * 30 * 24 - self.seq_len]
        border2s = [18 * 30 * 24, 18 * 30 * 24 + 3 * 30 * 24, 18 * 30 * 24 + 6 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
             #### Added error raise as dataset univariate
            raise ValueError("M and MS invalid settings. Use univariate setting 'S' for the given dataset.")
            #cols_data = df_raw.columns[df_raw.columns.get_level_values(0) != 'time']
            #df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Construct the time array
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 10)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError("Pass timeenc as either 0 or 1")

        self.data_stamp = data_stamp    # The full time dataset
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

     def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

     def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

        # Assumes inputs of shape [nodes, seq, feats] or [seq, feats]
     def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)    
     
class Dataset_SYNTH_additive(Dataset):
     def __init__(self, root_path, flag='train', size=None, features='S', data_path='SYNTH_additive.csv',
                 target='TARGET', scale=True, inverse = False, timeenc=0, freq='h', **_):

        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        self.seq_len = size[0]          # S (notation used in paper)
        self.label_len = size[1]        # L
        self.pred_len = size[2]         # P

        self.total_seq_len = self.seq_len + self.pred_len
        assert self.label_len <= self.seq_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target        
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

     def __read_data__(self):
        self.scaler = StandardScaler()
        _path = os.path.join(self.root_path, self.data_path).replace('\\', '/')  # replace in case windows
        df_raw = pd.read_csv(_path)

        # Fit scaler based on training data only
        if self.flag != 'train':
            train_data = pd.read_csv(_path.replace(self.flag, 'train'))

        
        border1s = [0, 18 * 30 * 24 - self.seq_len, 18 * 30 * 24 + 3 * 30 * 24 - self.seq_len]
        border2s = [18 * 30 * 24, 18 * 30 * 24 + 3 * 30 * 24, 18 * 30 * 24 + 6 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
             #### Added error raise as dataset univariate
            raise ValueError("M and MS invalid settings. Use univariate setting 'S' for the given dataset.")
            #cols_data = df_raw.columns[df_raw.columns.get_level_values(0) != 'time']
            #df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Construct the time array
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 10)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError("Pass timeenc as either 0 or 1")

        self.data_stamp = data_stamp    # The full time dataset
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

     def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

     def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

        # Assumes inputs of shape [nodes, seq, feats] or [seq, feats]
     def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)    
     
class Dataset_SYNTH_additive_reversal(Dataset):
     def __init__(self, root_path, flag='train', size=None, features='S', data_path='SYNTH_additive_reversals.csv',
                 target='TARGET', scale=True, inverse = False, timeenc=0, freq='h', **_):

        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        self.seq_len = size[0]          # S (notation used in paper)
        self.label_len = size[1]        # L
        self.pred_len = size[2]         # P

        self.total_seq_len = self.seq_len + self.pred_len
        assert self.label_len <= self.seq_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target        
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

     def __read_data__(self):
        self.scaler = StandardScaler()
        _path = os.path.join(self.root_path, self.flag, self.data_path).replace('\\', '/')  # replace in case windows
        df_raw = pd.read_csv(_path)
        
        border1s = [0, 18 * 30 * 24 - self.seq_len, 18 * 30 * 24 + 3 * 30 * 24 - self.seq_len]
        border2s = [18 * 30 * 24, 18 * 30 * 24 + 3 * 30 * 24, 18 * 30 * 24 + 6 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
             #### Added error raise as dataset univariate
            raise ValueError("M and MS invalid settings. Use univariate setting 'S' for the given dataset.")
            #cols_data = df_raw.columns[df_raw.columns.get_level_values(0) != 'time']
            #df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Construct the time array
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 10)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError("Pass timeenc as either 0 or 1")

        self.data_stamp = data_stamp    # The full time dataset
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

     def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

     def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

        # Assumes inputs of shape [nodes, seq, feats] or [seq, feats]
     def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)    
     
class Dataset_SYNTH_multiplicative(Dataset):
     def __init__(self, root_path, flag='train', size=None, features='S', data_path='SYNTH_multiplicative.csv',
                 target='TARGET', scale=True, inverse = False, timeenc=0, freq='h', **_):

        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        self.seq_len = size[0]          # S (notation used in paper)
        self.label_len = size[1]        # L
        self.pred_len = size[2]         # P

        self.total_seq_len = self.seq_len + self.pred_len
        assert self.label_len <= self.seq_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target        
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

     def __read_data__(self):
        self.scaler = StandardScaler()
        _path = os.path.join(self.root_path, self.flag, self.data_path).replace('\\', '/')  # replace in case windows
        df_raw = pd.read_csv(_path)

        
        border1s = [0, 18 * 30 * 24 - self.seq_len, 18 * 30 * 24 + 3 * 30 * 24 - self.seq_len]
        border2s = [18 * 30 * 24, 18 * 30 * 24 + 3 * 30 * 24, 18 * 30 * 24 + 6 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
             #### Added error raise as dataset univariate
            raise ValueError("M and MS invalid settings. Use univariate setting 'S' for the given dataset.")
            #cols_data = df_raw.columns[df_raw.columns.get_level_values(0) != 'time']
            #df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Construct the time array
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 10)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError("Pass timeenc as either 0 or 1")

        self.data_stamp = data_stamp    # The full time dataset
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

     def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

     def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

        # Assumes inputs of shape [nodes, seq, feats] or [seq, feats]
     def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)    
     
class Dataset_SYNTH_multiplicative_reversal(Dataset):
     def __init__(self, root_path, flag='train', size=None, features='S', data_path='SYNTH_multiplicative_reversals.csv',
                 target='TARGET', scale=True, inverse = False, timeenc=0, freq='h', **_):

        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        self.seq_len = size[0]          # S (notation used in paper)
        self.label_len = size[1]        # L
        self.pred_len = size[2]         # P

        self.total_seq_len = self.seq_len + self.pred_len
        assert self.label_len <= self.seq_len

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target        
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

     def __read_data__(self):
        self.scaler = StandardScaler()
        _path = os.path.join(self.root_path, self.flag, self.data_path).replace('\\', '/')  # replace in case windows
        df_raw = pd.read_csv(_path)
        
        border1s = [0, 18 * 30 * 24 - self.seq_len, 18 * 30 * 24 + 3 * 30 * 24 - self.seq_len]
        border2s = [18 * 30 * 24, 18 * 30 * 24 + 3 * 30 * 24, 18 * 30 * 24 + 6 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
             #### Added error raise as dataset univariate
            raise ValueError("M and MS invalid settings. Use univariate setting 'S' for the given dataset.")
            #cols_data = df_raw.columns[df_raw.columns.get_level_values(0) != 'time']
            #df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Construct the time array
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 10)
            data_stamp = df_stamp.drop(['time'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            raise ValueError("Pass timeenc as either 0 or 1")

        self.data_stamp = data_stamp    # The full time dataset
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

     def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

     def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

        # Assumes inputs of shape [nodes, seq, feats] or [seq, feats]
     def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)    
     
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'SYNTHh1': Dataset_SYNTH_hour,
    'SYNTHh2': Dataset_SYNTH_hour,
    'SYNTH_additive': Dataset_SYNTH_additive,
    'SYNTH_additive_reversal': Dataset_SYNTH_additive_reversal,
    'SYNTH_multiplicative': Dataset_SYNTH_multiplicative,
    'SYNTH_multiplicative_reversal' : Dataset_SYNTH_multiplicative_reversal,
    'DEWINDh_large': Dataset_WIND_hour,
    'DEWINDh_small': Dataset_WIND_hour
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )

    print(flag, len(data_set))
    data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)    

    return data_set, data_loader

###########################################################################################################################
# Experiment class ########################################################################################################
    '''
    Parent class for fitting and testing. The actual model training class will inherit from this class. 
    '''
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

###########################################################################################################################
### MODEL COMPONENTS ######################################################################################################
# Data Embedding ##########################################################################################################
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3, spatial=False):
        super(TokenEmbedding, self).__init__()
        assert torch.__version__ >= '1.5.0'
        padding = kernel_size - 1
        self.spatial = spatial
        self.d_model = d_model
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=kernel_size, padding=padding, padding_mode='zeros', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        B, L, d = x.shape[:3]
        if self.spatial:
            x = x.permute(0, 3, 1, 2).reshape(-1, L, d)
            x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)[:, :L, :]
            x = x.reshape(B, -1, L, self.d_model).permute(0, 2, 3, 1)
        else:
            x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)[:, :L, :]
        return x
    
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()
    
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3, '10min': 5}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)
    
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 6
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't' or freq == '10min':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x
    
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, kernel_size=3,
                 spatial=False, temp_embed=True, d_pos=None, pos_embed=True):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, kernel_size=kernel_size, spatial=spatial)
        self.d_model = d_model
        if d_pos is None:
            self.d_pos = d_model
        else:
            self.d_pos = d_pos
        self.position_embedding = PositionalEmbedding(d_model=self.d_pos) if pos_embed else None
        if temp_embed:
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                        freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, n_node=None):
        val_embed = self.value_embedding(x)
        temp_embed = self.temporal_embedding(x_mark) if self.temporal_embedding is not None else None
        pos_embed = self.position_embedding(x) if self.position_embedding is not None else None
        if self.d_pos != self.d_model and pos_embed is not None:
            pos_embed = pos_embed.repeat_interleave(2, dim=-1)
        if temp_embed is not None:
            if not (len(val_embed.shape) == len(temp_embed.shape)):  # == len(pos_embed.shape)
                temp_embed = torch.unsqueeze(temp_embed, -1)
                pos_embed = torch.unsqueeze(pos_embed, -1) if pos_embed is not None else None
        if n_node is not None and temp_embed is not None:
            temp_embed = torch.repeat_interleave(temp_embed, n_node, 0)
        if pos_embed is not None:
            x = val_embed + temp_embed + pos_embed if temp_embed is not None else val_embed + pos_embed
        else:
            x = val_embed + temp_embed if temp_embed is not None else val_embed
        return self.dropout(x)
    
###########################################################################################################################
# Attention Mechanisms ####################################################################################################

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False,
                 sparse_flag=False, win_len=0, res_len=None, fft_flag=False, **_):
        super(FullAttention, self).__init__()
        self.sparse_flag = sparse_flag
        self.win_len = win_len
        self.res_len = res_len
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.fft_flag = fft_flag

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape[:4]

        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

        if self.sparse_flag:
            sparse_mask = LogSparseMask(B, L, S, self.win_len, self.res_len, device=queries.device)
            if self.mask_flag:
                attn_mask._mask = attn_mask._mask.logical_or(sparse_mask._mask)
            else:
                attn_mask = sparse_mask

        if self.sparse_flag or self.mask_flag:
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        if self.fft_flag:
            V = torch.einsum("bhls,bshdc->blhdc", A, values)
        else:
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        
# Convolutional Attention from the LogSparse Transformer
class LogSparseAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, qk_ker, d_keys=None, d_values=None, v_conv=False, **_):
        super(LogSparseAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.qk_ker = qk_ker
        self.query_projection = nn.Conv1d(d_model, d_keys * n_heads, self.qk_ker)
        self.key_projection = nn.Conv1d(d_model, d_keys * n_heads, self.qk_ker)
        self.v_conv = v_conv
        if v_conv:
            self.value_projection = nn.Conv1d(d_model, d_values * n_heads, self.qk_ker)
        else:
            self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, **_):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = nn.functional.pad(queries.permute(0, 2, 1), pad=(self.qk_ker - 1, 0))
        queries = self.query_projection(queries).permute(0, 2, 1).view(B, L, H, -1)

        keys = nn.functional.pad(keys.permute(0, 2, 1), pad=(self.qk_ker - 1, 0))
        keys = self.key_projection(keys).permute(0, 2, 1).view(B, S, H, -1)

        if self.v_conv:
            values = nn.functional.pad(values.permute(0, 2, 1), pad=(self.qk_ker - 1, 0))
            values = self.value_projection(values).permute(0, 2, 1).view(B, S, H, -1)
        else:
            values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, output_size=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model) if output_size is None else nn.Linear(d_values * n_heads, output_size)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, **_):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

###########################################################################################################################
# Encoder-Decoder #########################################################################################################
    
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, **kwargs):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            **kwargs
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, **kwargs):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask, **kwargs)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, **kwargs)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, **kwargs):
        x, a_sa = self.self_attention(x, x, x, attn_mask=x_mask, **kwargs)
        x = x + self.dropout(x)
        x = self.norm1(x)

        x, a_ca = self.cross_attention(x, cross, cross, attn_mask=cross_mask, **kwargs)
        x = x + self.dropout(x)

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y), a_sa, a_ca


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None, **_):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, **kwargs):
        attn = []
        for layer in self.layers:
            x, a_sa, a_ca = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, **kwargs)
            attn.append(a_sa)
            attn.append(a_ca)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, attn
    
###########################################################################################################################
### ENTIRE MODEL FRAMEWORKS ###############################################################################################
# LogSparse #######################################################################################################  
    
class Logsparse(nn.Module):

    def __init__(self, configs):
        super(Logsparse, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.total_length = configs.pred_len + configs.seq_len

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, kernel_size=configs.kernel_size)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, kernel_size=configs.kernel_size)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    LogSparseAttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, sparse_flag=configs.sparse_flag,
                                      win_len=configs.win_len, res_len=configs.res_len),
                        configs.d_model, configs.n_heads, configs.qk_ker, v_conv=configs.v_conv),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    LogSparseAttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, sparse_flag=configs.sparse_flag,
                                      win_len=configs.win_len, res_len=configs.res_len),
                        configs.d_model, configs.n_heads, configs.qk_ker, v_conv=configs.v_conv),
                    LogSparseAttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, sparse_flag=configs.sparse_flag,
                                      win_len=configs.win_len, res_len=configs.res_len),
                        configs.d_model, configs.n_heads, configs.qk_ker, v_conv=configs.v_conv),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, **_):
        attns = []
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, a = self.encoder(enc_out, attn_mask=enc_self_mask)
        attns.append(a)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out, a = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        attns.append(a)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
###########################################################################################################################
### Training Helpers ######################################################################################################
# Learning Rate Decay  ####################################################################################################
        
def adjust_learning_rate(optimizer, epoch, args, warmup=0):
    epoch = epoch - 1
    if args.lradj == 'type1':
        if epoch < warmup:
            lr_adjust = {epoch: epoch*(args.learning_rate - args.learning_rate / 100)/warmup + args.learning_rate / 100}
        else:
            lr_adjust = {epoch: args.learning_rate * (args.lr_decay_rate ** ((epoch - warmup) // 1))}   # 0.5
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        period = 5
        decay_rate1 = args.lr_decay_rate + (1 - args.lr_decay_rate) / 2
        decay_rate2 = args.lr_decay_rate
        lr_start = args.learning_rate * decay_rate1**((epoch + period) // period)/decay_rate1
        lr_end = args.learning_rate * decay_rate2 ** ((epoch + period * 2) // period) / decay_rate2
        lr_adjust = {epoch: (0.5 + 0.5*np.cos(np.pi/period*(epoch % period)))*(lr_start - lr_end) + lr_end}
    elif args.lradj == 'type4':
        epoch += 1
        lr_adjust = {epoch: args.learning_rate*min(epoch**-0.5, epoch*50**-1.5)}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if args.lradj != 'type4':
            print('Updating learning rate to {}'.format(lr))

###########################################################################################################################
# Early Stopping ##########################################################################################################
            
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint=True, model_setup=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint = checkpoint
        self.model_setup = model_setup
        self.val_losses = []

    def __call__(self, val_loss, model, path, epoch):
        score = -val_loss
        self.val_losses.append(val_loss)
        if self.best_score is None:
            self.best_score = score
            if self.checkpoint:
                self.save_checkpoint(val_loss, model, path, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.checkpoint:
                self.save_checkpoint(val_loss, model, path, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, epoch):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

        # if not os.path.exists(path + '/' + 'model_setup.pickle'):
        pickle.dump(self.model_setup.__dict__, open(path + '/' + 'model_setup.pickle', 'wb'))
        pickle.dump({'epoch': epoch, 'val_loss_min': val_loss, 'val_losses': self.val_losses},
                    open(path + '/' + 'epoch_loss.pickle', 'wb'))
        with open(path + '/' + 'model_setup.txt', 'w') as f:
            f.write('Epoch: ' + str(epoch + 1) + '\nValLoss: ' + str(val_loss))
            f.write('\n\n__________________________________________________________\n\n')
            for key, value in self.model_setup.__dict__.items():
                f.write('%s \t%s\n' % (key, value))

###########################################################################################################################
# Running Experiments #####################################################################################################
                
class Exp_Logsparse(Exp_Basic):
    def __init__(self, args):
        super(Exp_Logsparse, self).__init__(args)

    def _build_model(self):
        model_dict = {
            #'Transformer': Transformer,
            'Logsparse': Logsparse,
            #'LSTM': LSTM,
        }

        model = model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def MAPE(self, pred, tar, eps=1e-07):
        loss = torch.mean(torch.abs(pred - tar) / (tar + eps))
        return loss

    def vali(self, setting, vali_data, vali_loader, criterion, epoch=0, plot_res=1, save_path=None):
        total_loss = []
        total_mse = []
        total_mape = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                # Non-Graph Data (i.e. just temporal)
                dec_inp = batch_y

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_zeros = dec_inp[:, (self.args.label_len-1):self.args.label_len, :]         # Select last value
                dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                 # Repeat for pred_len
                dec_inp = torch.cat([dec_inp[:, :self.args.label_len, :], dec_zeros], dim=1)   # Add Placeholders
                dec_inp = dec_inp.float().to(self.device)

                dec_inp = dec_inp[:, :, -self.args.dec_in:]
                batch_x = batch_x[:, :, -self.args.enc_in:]

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if 'M' in self.args.features:
                    f_dim = -self.args.c_out
                else:
                    f_dim = 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')     # self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self.vali_losses = []  # Store validation losses

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and self.args.checkpoint_flag:
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                       checkpoint=self.args.checkpoint_flag, model_setup=self.args)

        if self.args.checkpoint_flag:
            load_path = os.path.normpath(os.path.join(path, 'checkpoint.pth'))
            if os.path.exists(load_path) and self.load_check(path=os.path.normpath(os.path.join(path, 'model_setup.pickle'))):
                self.model.load_state_dict(torch.load(load_path))
                epoch_info = pickle.load(
                    open(os.path.normpath(os.path.join('./checkpoints/' + setting, 'epoch_loss.pickle')), 'rb'))
                start_epoch = epoch_info['epoch']
                early_stopping.val_losses = epoch_info['val_losses']
                early_stopping.val_loss_min = epoch_info['val_loss_min']
                self.vali_losses = epoch_info['val_losses']
                del epoch_info
            else:
                start_epoch = 0
                print('Could not load best model')
        else:
            start_epoch = 0

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        teacher_forcing_ratio = 0.8     # For LSTM Enc-Dec training (not used for others).
        total_num_iter = 0
        time_now = time.time()
        for epoch in range(start_epoch, self.args.train_epochs):

            # Reduce the tearcher forcing ration every epoch
            if self.args.model == 'LSTM':
                teacher_forcing_ratio -= 0.08
                teacher_forcing_ratio = max(0., teacher_forcing_ratio)
                print('teacher_forcing_ratio: ', teacher_forcing_ratio)

            # type4 lr scheduling is updated more frequently
            if self.args.lradj != 'type4':
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            train_loss = []

            self.model.train()
            epoch_time = time.time()
            num_iters = len(train_loader)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if self.args.lradj == 'type4':
                    adjust_learning_rate(model_optim, total_num_iter + 1, self.args)
                    total_num_iter += 1

                dec_inp = batch_y

                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_zeros = dec_inp[:, (self.args.label_len-1):self.args.label_len, :]         # Select last value
                dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                 # Repeat for pred_len
                dec_inp = torch.cat([dec_inp[:, :self.args.label_len, :], dec_zeros], dim=1)   # Add Placeholders
                dec_inp = dec_inp.float().to(self.device)

                dec_inp = dec_inp[:, :, -self.args.dec_in:]
                batch_x = batch_x[:, :, -self.args.enc_in:]

                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         teacher_forcing_ratio=teacher_forcing_ratio, batch_y=batch_y)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         teacher_forcing_ratio=teacher_forcing_ratio, batch_y=batch_y)

                if 'M' in self.args.features:
                    f_dim = -self.args.c_out
                else:
                    f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0 and self.args.verbose == 1:
                    print("\titers: {0}/{1}, epoch: {2} | loss: {3:.7f}".format(i + 1, num_iters, epoch + 1, np.average(train_loss)))

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(setting, vali_data, vali_loader, criterion, epoch=epoch, save_path=path)
            test_flag = False
            if test_flag:
                test_loss = self.vali(setting, test_data, test_loader, criterion, epoch=epoch, save_path=path)

            # Plot the losses:
            if self.args.plot_flag and self.args.checkpoint_flag:
                loss_save_dir = path + '/pic/train_loss.png'
                loss_save_dir_pkl = path + '/train_loss.pickle'
                if os.path.exists(loss_save_dir_pkl):
                    fig_progress = pickle.load(open(loss_save_dir_pkl, 'rb'))

                if 'fig_progress' not in locals():
                    fig_progress = PlotLossesSame(epoch + 1,
                                                  Training=train_loss,
                                                  Validation=vali_loss)
                else:
                    fig_progress.on_epoch_end(Training=train_loss,
                                              Validation=vali_loss)

                if not os.path.exists(os.path.dirname(loss_save_dir)):
                    os.makedirs(os.path.dirname(loss_save_dir))
                fig_progress.fig.savefig(loss_save_dir)
                pickle.dump(fig_progress, open(loss_save_dir_pkl, 'wb'))    # To load figure that we can append to

            if test_flag:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path, epoch)
            self.vali_losses += [vali_loss]       # Append validation loss
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # After Training, load the best model.
        if self.args.checkpoint_flag:
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def load_check(self, path, ignore_vars=None, ignore_paths=False):
        # Function to check that the checkpointed and current settings are compatible.
        if ignore_vars is None:
            ignore_vars = [
                'is_training',
                'train_epochs',
                'plot_flag',
                'root_path',
                'data_path',
                'data_path',
                'checkpoints',
                'checkpoint_flag',
                'output_attention',
                'do_predict',
                'des',
                'verbose',
                'itr',
                'patience',
                'des',
                'gpu',
                'use_gpu',
                'use_multi-gpu',
                'devices',
            ]
        if ignore_paths:
            ignore_vars += [
                'model_id',
            ]

        setting2 = pickle.load(open(path, 'rb'))
        for key, val in self.args.__dict__.items():
            if key in ignore_vars:
                continue
            if val != setting2[key]:
                print(val, ' is not equal to ', setting2[key], ' for ', key)
                return False

        return True

    def test(self, setting, test=1, base_dir='', save_dir=None, ignore_paths=False, save_flag=True):
        test_data, test_loader = self._get_data(flag='test')
        if save_dir is None:
            save_dir = base_dir
        if test:
            print('loading model')
            if len(base_dir) == 0:
                load_path = os.path.normpath(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            else:
                load_path = os.path.normpath(os.path.join(base_dir + 'checkpoints/' + setting, 'checkpoint.pth'))
            load_check_flag = self.load_check(path=os.path.normpath(os.path.join(os.path.dirname(load_path),
                                                                                 'model_setup.pickle')),
                                              ignore_paths=ignore_paths)
            if os.path.exists(load_path) and load_check_flag:
                self.model.load_state_dict(torch.load(load_path))
            else:
                print('Could not load best model')

        preds = []
        trues = []
        station_ids = []
        if save_flag:
            if len(save_dir) == 0:
                folder_path = './results/' + setting + '_iter_' + str(self.iter) + '/'
            else:
                folder_path = save_dir + 'results/' + setting + '_iter_' + str(self.iter) +'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                dec_inp = batch_y

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_zeros = dec_inp[:, (self.args.label_len-1):self.args.label_len, :]
                dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()
                dec_inp = torch.cat([dec_inp[:, :self.args.label_len, :], dec_zeros], dim=1)
                dec_inp = dec_inp.float().to(self.device)

                dec_inp = dec_inp[:, :, -self.args.dec_in:]
                batch_x = batch_x[:, :, -self.args.enc_in:]

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if 'M' in self.args.features:
                    f_dim = -self.args.c_out
                else:
                    f_dim = 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if self.args.data == 'WindGraph':
                    station_ids.append(batch_x.station_names)
                if i % 20 == 0:
                    if self.args.data == 'WindGraph':
                        input = batch_x.nodes.detach().cpu().numpy()
                    else:
                        input = batch_x.detach().cpu().numpy()

                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    #if save_flag:
                        #visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))
                # Added line - saves the first batch of the test set.
                if i == 0:
                    first_batch = {
                    'batch_x' : batch_x ,
                    'batch_y' : batch_y ,
                    'batch_x_mark' : batch_x_mark ,
                    'batch_y_mark' : batch_y_mark ,
                    }    

        preds = np.vstack(preds)
        trues = np.vstack(trues)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # save results
        if save_flag:
            if len(save_dir) == 0:
                folder_path = './results/' + setting + '/'
            else:
                folder_path = save_dir + 'results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        if self.args.inverse:
            preds_un = test_data.inverse_transform(preds)
            trues_un = test_data.inverse_transform(trues)
            mae_un, mse_un, rmse_un, mape_un, mspe_un = metric(preds_un, trues_un)
            losses = {
            'mae_sc': mae,
            'mse_sc': mse,
            'rmse_sc': rmse,
            'mape_sc': mape,
            'mspe_sc': mspe,
            '': '\n\n',
            'mae_un': mae_un,
            'mse_un': mse_un,
            'mape_un': mape_un,
            'rmse_un': rmse_un,
            'mape_un': mape_un,
            'mspe_un': mspe_un,
            }
        else:
            losses = {
            'mae_sc': mae,
            'mse_sc': mse,
            'mape_sc': mape,
            'rmse_sc': rmse,
            'mape_sc': mape,
            'mspe_sc': mspe,
            }

        if not save_flag:
            return losses

        with open(folder_path + "results_loss.txt", 'w') as f:
            for key, value in losses.items():
                f.write('%s:%s\n' % (key, value))

        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        if self.args.inverse:
            np.save(folder_path + 'pred_un.npy', preds_un)
            np.save(folder_path + 'true_un.npy', trues_un)
            np.save(folder_path + 'metrics_un.npy', np.array([mae_un, mse_un, rmse_un, mape_un, mspe_un]))

        with open(folder_path + 'metrics.txt', 'w') as f:
            f.write('mse: ' + str(mse) + '\n')
            f.write('mae: ' + str(mae) + '\n')
            f.write('rmse: ' + str(rmse) + '\n')
            f.write('mape: ' + str(mape) + '\n')
            f.write('mspe: ' + str(mspe) + '\n')

        return preds , trues ,mae ,mape, mse , losses, first_batch
    
###########################################################################################################################
# Our Simple User Interface ###############################################################################################
class LogsparseTS():
    '''
    Our custom wrapper class to provide an user-friendly interface for fitting and testing the of the Logsparse model. 
    For simplicity of use, methods included align with naming used by Keras.  
    '''

    def __init__(self, model='Logsparse'):
        # Currently only Logsparse is supported
        if model != 'Logsparse':
            raise ValueError("Model not supported. Please use 'Logsparse'.")
        # Initialize dot dictionary
        self.args = dotdict()
        # basic config
        self.args.model = model
        self.args.plot_flat = 0
        self.args.verbose = 1
        self.args.is_training = 1
        self.args.inverse = False
        # data loader
        self.args.data = 'Synth1'
        self.args.root_path = './SYNTHDataset'
        self.args.data_path ='SYNTHh1.csv' 
        self.args.target = 'TARGET'
        self.args.freq = 'h'
        self.args.checkpoints = './checkpoints/'
        self.args.checkpoint_flag = 1
        # forecasting task
        self.args.features = 'S' #univariate
        self.args.seq_len = 168
        self.args.label_len = 48
        self.args.pred_len = 24
        self.args.enc_in = 1
        self.args.dec_in =1
        self.args.c_out = 1
        # model define
        self.args.d_model = 512
        self.args.n_heads = 8
        self.args.e_layers = 2
        self.args.d_layers = 1
        self.args.d_ff = 2048
        self.args.factor = 3
        self.args.distil = True
        self.args.dropout = 0.05
        self.args.embed = 'timeF'
        self.args.activation = 'gelu'
        self.args.output_attention = False
        self.args.win_len = 6
        self.args.res_len = None
        self.args.qk_ker = 4
        self.args.v_conv = 0
        self.args.sparse_flag = 1
        self.args.top_keys = 0
        self.args.kernel_size = 3
        self.args.train_strat_lstm = 'recursive'
        self.args.model_id= self.args.model + '_' + str(self.args.data) + '_' + str(self.args.pred_len)
        # Optimization
        self.args.num_workers = 0
        self.args.itr = 3
        self.args.train_epochs = 10
        self.args.batch_size = 32
        self.args.patience= 5
        self.args.learning_rate = 0.001
        self.args.lr_decay_rate = 0.8
        self.args.des = 'test'
        self.args.loss = 'mse'
        self.args.lradj = 'type1'
        # GPU
        self.args.use_gpu = True 
        self.args.gpu = 0
        self.args.use_multi_gpu = False
        self.args.devices = '0,1,2,3'

        self.args.use_gpu = True if torch.cuda.is_available() and self.args.use_gpu else False
        if self.args.use_gpu and self.args.use_multi_gpu: 
            self.args.devices = self.args.devices.replace(' ', '')
            device_ids = self.args.devices.split(',')
            self.args.device_ids = [int(id_) for id_ in device_ids]
            self.args.gpu = self.args.device_ids[0]
        


    def compile(self, learning_rate=1e-3, loss='mse', early_stopping_patience=5):
        '''
        Compiles the Logsparse model for training.
        Args:
            learning_rate (float): Learning rate to be used. Default: '1e-3'.
            loss (str): Loss function to be used. Default: 'mse'.
            early_stopping_patience (int): Amount of epochs to beak training loop after no validation performance improvement. Default: 5.
        '''
        if loss != 'mse':
            raise ValueError("Loss function not supported. Please use 'mse'.")
        self.args.learning_rate = learning_rate
        self.args.loss = loss
        self.args.patience = early_stopping_patience

    def fit(self, data='SYNTHh1', data_root_path='./SYNTHDataset/', batch_size=32, epochs=10, pred_len=24,
            seq_len = 168 , features = 'S' , target = 'TARGET', enc_in = 1, dec_in = 1, c_out = 1):
        '''
        Fits the Logsparse model.
        Args:
            data (str): Name of the dataset used. For now, only 'SYNTHh1', 'SYNTHh2', 'DEWINDh_large' and 'DEWINDh_small' are supported. Default: 'SYNTHh1'. 
            data_root_path (str): Root folder for given dataset. Default: './SYNTHDataset'.
            batch_size (int): Batch size. Default: 32.
            epochs (int): Number of epochs for training the model. Default: 10.
            pred_len (int): Prediction window length. Default: 24. Recommended: 24, 48, 168, 336, 720.
        '''
        # temporary line
        possible_datasets = ['SYNTHh1', 'SYNTHh2', 'SYNTH_additive' , 'SYNTH_additive_reveral' , 'SYNTH_multiplicative', 'SYNTH_multiplicative_reversal' , 'DEWINDh_large', 'DEWINDh_small' , 'ETTh1']
        if data not in possible_datasets:
            raise ValueError("Dataset not supported. Please use one of the following: 'SYNTHh1', 'SYNTHh2', SYNTH_additive', 'SYNTH_additive_reveral' 'SYNTH_multiplicative', 'SYNTH_multiplicative_reversal' , 'DEWINDh_large', 'DEWINDh_small' ,'ETTh1'.")
        # temporary line
        possible_predlens = [24, 48, 168, 336, 720]
        #if pred_len not in possible_predlens:
            #raise ValueError('Prediction length outside current experiment scope. Please use either 24, 48, 168, 336, 720.')
        self.args.data = data
        self.args.root_path = data_root_path
        self.args.data_path = f'{self.args.data}.csv'
        self.args.train_epochs = epochs
        self.args.batch_size = batch_size
        self.args.seq_len = seq_len
        self.args.pred_len = pred_len
        self.args.features = features
        self.args.target = target
        self.args.enc_in = enc_in
        self.args.dec_in = dec_in
        self.args.c_out = c_out
        
       #self.args.detail_freq = self.args.freq
       #self.args.freq = self.args.freq[-1:]
        
        print('Beginning to fit the model with the following arguments:')
        print(f'{self.args}')
        print('='*150)
        # Set up model variable
        Experiment_Model = Exp_Logsparse
        # Set up training settings
        self.setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(self.args.model, self.args.data, self.args.features, 
                self.args.seq_len, self.args.label_len, self.args.pred_len,
                self.args.d_model, self.args.n_heads, self.args.d_layers, self.args.d_ff, self.args.factor, self.args.embed, self.args.distil, self.args.des, self.iter)
        # Initialize Model Class
        self.experiment_model = Experiment_Model(self.args)
        # Train
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(self.setting))
        self.model = self.experiment_model.train(self.setting)

    def predict(self):
        '''
        Makes predictions on pre-defined test set. Does not require any arguments.
        Returns:
            preds: A 3D array of predictions of the following shape (number of windows, number of time points per window, number of targets.)
            As self variables, trues, mse, mae, all_metrics, and first_batch_test can also be called. 
        '''
        #if not self.model:
            #raise ValueError('No model trained. Make sure to run .fit() first.')
        # Predict
        self.preds, self.trues, self.mse, self.mae, self.losses, self.first_batch_test = self.experiment_model.test(self.setting)
        # Clear memory
        torch.cuda.empty_cache()
        return self.preds

