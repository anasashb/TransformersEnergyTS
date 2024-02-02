# A script file to easily and convinently run and use the Pyaformer model. Includes a wrapper class for the model that can be imported and easily used for training and prediction.

# Imports
import argparse
from typing import List
from typing import Union
from functools import lru_cache
from tqdm import tqdm
import time
import os.path
import sys
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.functional import align_tensors
from torch.nn.modules.linear import Linear
from torch.nn.modules import loss
import torch.optim as optim

import pynvml
#Only works on a computer with GPU
#pynvml.nvmlInit()

###########################################################################################################################
# Dot dictionary ##########################################################################################################
class dotdict(dict):
    '''
    Simple class to allow storing model arguments in a dot dictionary.
    '''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

###########################################################################################################################
# Metrics #################################################################################################################
    
def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae,mse,rmse,mape,mspe

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

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

class TopkMSELoss(torch.nn.Module):
    def __init__(self, topk) -> None:
        super().__init__()
        self.topk = topk
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, output, label):
        losses = self.criterion(output, label).mean(2).mean(1)
        losses = torch.topk(losses, self.topk)[0]

        return losses

class SingleStepLoss(torch.nn.Module):
    """ Compute top-k log-likelihood and mse. """

    def __init__(self, ignore_zero):
        super().__init__()
        self.ignore_zero = ignore_zero

    def forward(self, mu, sigma, labels, topk=0):
        if self.ignore_zero:
            indexes = (labels != 0)
        else:
            indexes = (labels >= 0)

        distribution = torch.distributions.normal.Normal(mu[indexes], sigma[indexes])
        likelihood = -distribution.log_prob(labels[indexes])

        diff = labels[indexes] - mu[indexes]
        se = diff * diff

        if 0 < topk < len(likelihood):
            likelihood = torch.topk(likelihood, topk)[0]
            se = torch.topk(se, topk)[0]

        return likelihood, se

def AE_loss(mu, labels, ignore_zero):
    if ignore_zero:
        indexes = (labels != 0)
    else:
        indexes = (labels >= 0)

    ae = torch.abs(labels[indexes] - mu[indexes])
    return ae

class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

###########################################################################################################################
# Time Features #############################################################################################################
    
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

def time_features(dates, timeenc=1, freq='h'):
    if timeenc==0:
        dates['month'] = dates.date.apply(lambda row:row.month,1)
        dates['day'] = dates.date.apply(lambda row:row.day,1)
        dates['weekday'] = dates.date.apply(lambda row:row.weekday(),1)
        dates['hour'] = dates.date.apply(lambda row:row.hour,1)
        dates['minute'] = dates.date.apply(lambda row:row.minute,1)
        dates['minute'] = dates.minute.map(lambda x:x//15)
        freq_map = {
            'y':[],'m':['month'],'w':['month'],'d':['month','day','weekday'],
            'b':['month','day','weekday'],'h':['month','day','weekday','hour'],
            't':['month','day','weekday','hour','minute'],
        }
        return dates[freq_map[freq.lower()]].values
    if timeenc==1:
        dates = pd.to_datetime(dates.date.values)
        return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)]).transpose(1,0)
    
"""Long range dataloader for synthetic dataset"""
class Dataset_SYNTH_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='SYNTHh1.csv', 
                 dataset='SYNTHh1', timeenc = 0, scale=True,  inverse=False):
        if size == None:
            self.seq_len = 24*4*1
            self.pred_len = 24*4
        self.seq_len = size[0]
        self.pred_len = size[1]
       # init
        assert flag in ['train', 'test']
        type_map = {'train':0, 'test':1}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        preprocess_path = os.path.join(self.root_path, self.data_path)
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.window_stride = 1 
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis = 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def fit(self, data):
        mean = data.mean()
        std = data.std()
        return mean, std

    def inverse_transform(self, output, seq_y, mean, std):
        output = output *  (mean.unsqueeze(1).unsqueeze(1) + 1)
        seq_y = seq_y * (mean.unsqueeze(1).unsqueeze(1) + 1)
        return output, seq_y   

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark , self.scaler.mean, self.scaler.std
       
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1
    
"""Long range dataloader for synthetic dataset"""
class Dataset_SYNTH_additive(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='SYNTH_additive.csv', 
                 dataset='SYNTH_additive', timeenc = 0, scale=True,  inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*1
            self.pred_len = 24*4
        self.seq_len = size[0]
        self.pred_len = size[1]
       # init
        assert flag in ['train', 'test']
        type_map = {'train':0, 'test':1}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        preprocess_path = os.path.join(self.root_path, self.data_path)
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.window_stride = 1 
        self.__read_data__()

        

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis = 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def fit(self, data):
        mean = data.mean()
        std = data.std()
        return mean, std

    def inverse_transform(self, output, seq_y, mean, std):
        output = output *  (mean.unsqueeze(1).unsqueeze(1) + 1)
        seq_y = seq_y * (mean.unsqueeze(1).unsqueeze(1) + 1)
        return output, seq_y   

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark , self.scaler.mean, self.scaler.std
       
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1
    
class Dataset_SYNTH_additive_reversal(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='SYNTH_additive_reversals.csv', 
                 dataset='SYNTH_additive_reversal', timeenc = 0, scale=True,  inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*1
            self.pred_len = 24*4
        self.seq_len = size[0]
        self.pred_len = size[1]
       # init
        assert flag in ['train', 'test']
        type_map = {'train':0, 'test':1}
        self.set_type = type_map[flag]
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path
        self.data_path = data_path
        preprocess_path = os.path.join(self.root_path, self.data_path)
        self.__read_data__()

        self.window_stride = 1 

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis = 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def fit(self, data):
        mean = data.mean()
        std = data.std()
        return mean, std

    def inverse_transform(self, output, seq_y, mean, std):
        output = output *  (mean.unsqueeze(1).unsqueeze(1) + 1)
        seq_y = seq_y * (mean.unsqueeze(1).unsqueeze(1) + 1)
        return output, seq_y   

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark , self.scaler.mean, self.scaler.std
       
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1
    
class Dataset_SYNTH_multiplicative(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='SYNTH_multiplicative.csv', 
                 dataset='SYNTH_multiplicative', timeenc = 0, scale=True,  inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*1
            self.pred_len = 24*4
        self.seq_len = size[0]
        self.pred_len = size[1]
       # init
        assert flag in ['train', 'test']
        type_map = {'train':0, 'test':1}
        self.set_type = type_map[flag]
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path
        self.data_path = data_path
        preprocess_path = os.path.join(self.root_path, self.data_path)
        self.__read_data__()

        self.window_stride = 1 

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis = 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def fit(self, data):
        mean = data.mean()
        std = data.std()
        return mean, std

    def inverse_transform(self, output, seq_y, mean, std):
        output = output *  (mean.unsqueeze(1).unsqueeze(1) + 1)
        seq_y = seq_y * (mean.unsqueeze(1).unsqueeze(1) + 1)
        return output, seq_y   

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark , self.scaler.mean, self.scaler.std
       
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1
    
class Dataset_SYNTH_multiplicative_reversal(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='Dataset_SYNTH_multiplicative_reversals.csv', 
                 dataset='Dataset_SYNTH_multiplicative_reversal', timeenc = 0, scale=True,  inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*1
            self.pred_len = 24*4
        self.seq_len = size[0]
        self.pred_len = size[1]
       # init
        assert flag in ['train', 'test']
        type_map = {'train':0, 'test':1}
        self.set_type = type_map[flag]
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path
        self.data_path = data_path
        preprocess_path = os.path.join(self.root_path, self.data_path)
        self.__read_data__()

        self.window_stride = 1 

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis = 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def fit(self, data):
        mean = data.mean()
        std = data.std()
        return mean, std

    def inverse_transform(self, output, seq_y, mean, std):
        output = output *  (mean.unsqueeze(1).unsqueeze(1) + 1)
        seq_y = seq_y * (mean.unsqueeze(1).unsqueeze(1) + 1)
        return output, seq_y   

    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark , self.scaler.mean, self.scaler.std
       
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1
    

def prepare_dataloader(args):
    """ Load data and prepare dataloader. """

    data_dict = {
        #'ETTh1':Dataset_ETT_hour,
        #'ETTh2':Dataset_ETT_hour,
        #'elect':Dataset_Custom,
        'SYNTHh1': Dataset_SYNTH_hour,
        'SYNTH_additive' : Dataset_SYNTH_additive ,
        'SYNTH_multiplicative' : Dataset_SYNTH_multiplicative ,
        'SYNTH_additive_reversal' : Dataset_SYNTH_additive_reversal,
        'SYNTH_multiplicative_reversal' : Dataset_SYNTH_multiplicative_reversal,
        #'DEWINDh_small' : Dataset_WIND_hour 
    }
    Data = data_dict[args.data]

    # prepare training dataset and dataloader
    shuffle_flag = False; drop_last = True; batch_size = args.batch_size
    train_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='train',
        size=[args.input_size, args.predict_step],
        inverse=args.inverse,
        dataset=args.data
    )
    print('train', len(train_set))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last)

    # prepare testing dataset and dataloader
    shuffle_flag = False; drop_last = True; batch_size = args.batch_size
    test_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='test',
        size=[args.input_size, args.predict_step],
        inverse=args.inverse,
        dataset=args.data
    )
    print('test', len(test_set))
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last)

    return train_loader, train_set, test_loader, test_set

def sample_mining_scheduler(epoch, batch_size):
    if epoch < 2:
        topk = batch_size
    elif epoch < 4:
        topk = int(batch_size * (5 - epoch) / (6 - epoch))
    else:
        topk = int(0.5 * batch_size)

    return topk


def dataset_parameters(args, dataset):
    """Prepare specific parameters for different datasets"""
    dataset2enc_in = {
        'ETTh1':7,
        'ETTh2':7,
        'elect':1,
        'SYNTHh1': 1,
        'SYNTH_additive' : 1,
        'SYNTH_multiplicative' : 1,
        'SYNTH_additive_reversal' : 1,
        'SYNTH_multiplicative_reversal' : 1,
        'DEWINDh_small' : 1 
    }
    dataset2cov_size = {
        'ETTh1':4,
        'ETTh2':4,
        'elect':3,
        'SYNTHh1': 3,
        'SYNTH_additive' : 3,
        'SYNTH_multiplicative' : 3,
        'SYNTH_multiplicative_reversal' : 3,
        'SYNTH_additive_reversal' : 3,
        'DEWINDh_small' : 3 
    }
    dataset2seq_num = {
        'ETTh1':1,
        'ETTh2':1,
        'elect':321,
        'flow': 1077,
        'SYNTHh1': 60, 
        'SYNTH_additive' : 60,
        'SYNTH_multiplicative' : 60,
        'SYNTH_multiplcative_reversal' : 60,
        'SYNTH_additive_reversal' : 60,
        'DEWINDh_small' : 60
    }
    dataset2embed = {
        'ETTh1':'DataEmbedding',
        'ETTh2':'DataEmbedding',
        'ETTm1':'DataEmbedding',
        'ETTm2':'DataEmbedding',
        'elect':'CustomEmbedding',
        'flow': 'CustomEmbedding',
        'SYNTHh1': 'CustomEmbedding',
        'SYNTH_additive' : 'CustomEmbedding',
        'SYNTH_multiplicative' : 'CustomEmbedding',
        'SYNTH_additive_reversal' : 'CustomEmbedding',
        'SYNTH_multiplicative_reversal' : 'CustomEmbedding',
        'DEWINDh_small' : 'CustomEmbedding' 
    }

    args.enc_in = dataset2enc_in[dataset]
    args.dec_in = dataset2enc_in[dataset]
    args.covariate_size = dataset2cov_size[dataset]
    args.seq_num = dataset2seq_num[dataset]
    args.embed_type = dataset2embed[dataset]

    return args


###########################################################################################################################
### MODEL COMPONENTS ######################################################################################################
    
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
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
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
    def __init__(self, d_model):
        super(TimeFeatureEmbedding, self).__init__()

        d_inp = 4
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

"""Embedding modules. The DataEmbedding is used by the ETT dataset for long range forecasting."""
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)

"""The CustomEmbedding is used by the electricity dataset and app flow dataset for long range forecasting."""
class CustomEmbedding(nn.Module):
    def __init__(self, c_in, d_model, temporal_size, seq_num, dropout=0.1):
        super(CustomEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = nn.Linear(temporal_size, d_model)
        self.seqid_embedding = nn.Embedding(seq_num, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark[:, :, :-1])\
            + self.seqid_embedding(x_mark[:, :, -1].long())

        return self.dropout(x)

"""The SingleStepEmbedding is used by all datasets for single step forecasting."""
class SingleStepEmbedding(nn.Module):
    def __init__(self, cov_size, num_seq, d_model, input_size, device):
        super().__init__()

        self.cov_size = cov_size
        self.num_class = num_seq
        self.cov_emb = nn.Linear(cov_size+1, d_model)
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.data_emb = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular')

        self.position = torch.arange(input_size, device=device).unsqueeze(0)
        self.position_vec = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)], device=device)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def transformer_embedding(self, position, vector):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = position.unsqueeze(-1) / vector
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, x):
        covs = x[:, :, 1:(1+self.cov_size)]
        seq_ids = ((x[:, :, -1] / self.num_class) - 0.5).unsqueeze(2)
        covs = torch.cat([covs, seq_ids], dim=-1)
        cov_embedding = self.cov_emb(covs)
        data_embedding = self.data_emb(x[:, :, 0].unsqueeze(2).permute(0, 2, 1)).transpose(1,2)
        embedding = cov_embedding + data_embedding

        position = self.position.repeat(len(x), 1).to(x.device)
        position_emb = self.transformer_embedding(position, self.position_vec.to(x.device))

        embedding += position_emb

        return embedding
    
###########################################################################################################################
# Attention Mechanisms ####################################################################################################
### Multihead Attention ####################################################################################################   
    
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        #self.layer_norm = GraphNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x
    
### Pyramidal Attention ####################################################################################################   

class GraphMM(torch.autograd.Function):
    '''Class to encapsulate tvm code for compiling a diagonal_mm function, in addition to calling
    this function from PyTorch
    '''

    function_dict = {}  # save a list of functions, each has a different set of parameters

    @staticmethod
    def _compile_function(dtype: str, device: str, b0: int = 4, b1: int = 8, b2: int = 8):
        '''Compiles a tvm function that computes diagonal_mm
        args:
        dtype: str in ['float64', 'float32', 'float16']
        device: str in ['cpu' or 'cuda']
        b0, b1, b2: size of tensor tiles. Very important for good performance
        '''
        import tvm  # import the full tvm library here for compilation. Don't import at the top of the file in case we don't need to compile
        from tvm.contrib import nvcc
        @tvm.register_func
        def tvm_callback_cuda_compile(code):
            """Use nvcc compiler for better perf."""
            ptx = nvcc.compile_cuda(code, target="ptx", arch='sm_52')  # use old arch for this to work on old GPUs
            return ptx

        assert dtype in ['float16', 'float32', 'float64']
        assert device in ['cpu', 'cuda']
        device = None if device == 'cpu' else device
        tgt_host="llvm"

        b = tvm.te.var('b')  # batch size
        n = tvm.te.var('n')  # sequence length
        h = tvm.te.var('h')  # number of heads
        m = tvm.te.var('m')  # hidden dimension
        w = tvm.te.var('w')  # window size
        padding = tvm.te.var('padding')  # padding
        transpose_t1 = tvm.te.var('transpose_t1')  # t1 should be transposed
        t1d3 = tvm.te.var('t1d3')  # last dimension of t1
        t3d3 = tvm.te.var('t3d3')  # last dimension of t3 (the result tensor)
        max_attn = tvm.te.var('max_attn')
        X = tvm.te.placeholder((b, n, h, t1d3), name='X', dtype=dtype)  # first tensor
        Y = tvm.te.placeholder((b, n, h, m), name='Y', dtype=dtype)  # second tensor
        k = tvm.te.reduce_axis((0, t1d3), name='k')  # dimension to sum over
        q_k_mask = tvm.te.placeholder((n, max_attn), name='q_k', dtype='int')  # dilation per head
        k_q_mask = tvm.te.placeholder((n, max_attn), name='k_q', dtype='int') # 
        output_shape = (b, n, h, t3d3)  # shape of the result tensor

        algorithm = lambda l, i, q, j: tvm.te.sum(
            tvm.te.if_then_else(
                t3d3 == m,  # if output dimension == m, then t1 is diagonaled (FIXME: This breaks if t3d3 == m == t1d3)
                tvm.te.if_then_else(
                    transpose_t1 == 0,
                    tvm.te.if_then_else(
                        q_k_mask[i, k]>=0,
                        X[l, i, q, k] * Y[l, q_k_mask[i, k], q, j],  # t1 is diagonaled
                        padding
                    ),
                    tvm.te.if_then_else(
                        q_k_mask[i, k]>=0,
                        X[l, q_k_mask[i, k], q, k_q_mask[i, k]] * Y[l, q_k_mask[i, k], q, j],  # # t1 is diagonaled and should be transposed
                        padding
                    ),
                ),
                tvm.te.if_then_else(
                    q_k_mask[i, j]>=0,
                    X[l, i, q, k] * Y[l, q_k_mask[i, j], q, k],  # t1 is not diagonaled, but the output tensor is going to be
                    padding
                )
            ), axis=k)

        Z = tvm.te.compute(output_shape, algorithm, name='Z')  # automatically generate cuda code
        s = tvm.te.create_schedule(Z.op)

        print('Lowering: \n ===================== \n{}'.format(tvm.lower(s, [X, Y, q_k_mask, k_q_mask], simple_mode=True)))

        # split long axis into smaller chunks and assing each one to a separate GPU thread/block
        ko, ki = s[Z].split(Z.op.reduce_axis[0], factor=b0)
        ZF = s.rfactor(Z, ki)

        j_outer, j_inner = s[Z].split(s[Z].op.axis[-1], factor=b1)
        i_outer, i_inner = s[Z].split(s[Z].op.axis[1], factor=b2)

        s[Z].bind(j_outer, tvm.te.thread_axis("blockIdx.x"))
        s[Z].bind(j_inner, tvm.te.thread_axis("threadIdx.y"))

        s[Z].bind(i_outer, tvm.te.thread_axis("blockIdx.y"))
        s[Z].bind(i_inner, tvm.te.thread_axis("threadIdx.z"))

        tx = tvm.te.thread_axis("threadIdx.x")
        s[Z].bind(s[Z].op.reduce_axis[0], tx)
        s[ZF].compute_at(s[Z], s[Z].op.reduce_axis[0])
        s[Z].set_store_predicate(tx.var.equal(0))

        print('Lowering with GPU splits: \n ===================== \n{}'.format(tvm.lower(s, [X, Y, q_k_mask, k_q_mask], simple_mode=True)))

        # compiling the automatically generated cuda code
        graph_mm = tvm.build(s, [X, Y, Z, q_k_mask, k_q_mask, max_attn, padding, transpose_t1, t3d3], target=device, target_host=tgt_host, name='graph_mm')
        return graph_mm

    @staticmethod
    def _get_lib_filename(dtype: str, device: str):
        base_filename = 'lib/lib_hierarchical_mm'
        return '{}_{}_{}.so'.format(base_filename, dtype, device)

    @staticmethod
    def _save_compiled_function(f, dtype: str, device: str):
        if not os.path.exists('lib/'):
            os.makedirs('lib/')
        f.export_library(GraphMM._get_lib_filename(dtype, device))

    @staticmethod
    def _load_compiled_function(dtype: str, device: str):
        # from tvm.module import load  # this can be the small runtime python library, and doesn't need to be the whole thing
        from tvm.runtime.module import load_module as load

        filename = GraphMM._get_lib_filename(dtype, device)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        potential_dirs = ['../../', '../', './', f'{current_dir}/', f'{current_dir}/../']
        for potential_dir in  potential_dirs:
            filepath = '{}{}'.format(potential_dir, filename)
            if os.path.isfile(filepath):
                print('Loading tvm binary from: {}'.format(filepath))
                return load(filepath)
        return None

    @staticmethod
    def _get_function(dtype: str, device: str):
        '''Loads the function from the disk or compile it'''
        # A list of arguments that define the function
        args = (dtype, device)
        if args not in GraphMM.function_dict:
            graph_mm = GraphMM._load_compiled_function(dtype, device)  # try to load from disk
            if not graph_mm:
                print('Tvm binary not found. Compiling ...')
                graph_mm = GraphMM._compile_function(dtype, device)  # compile
                GraphMM._save_compiled_function(graph_mm, dtype, device)  # save to disk
            # convert the tvm function into a pytorch function
            from tvm.contrib import dlpack
            graph_mm_pytorch = dlpack.to_pytorch_func(graph_mm)  # wrap it as a pytorch function
            # save the function into a dictionary to be reused
            GraphMM.function_dict[args] = graph_mm_pytorch  # save it in a dictionary for next time
        return GraphMM.function_dict[args]

    @staticmethod
    def _graph_mm(t1: torch.Tensor, t2: torch.Tensor, q_k_mask: torch.Tensor, k_q_mask: torch.Tensor,
                       is_t1_diagonaled: bool = False, transpose_t1: bool = False, padding: int = 0,
                       autoregressive: bool = False):
        '''Calls the compiled function after checking the input format. This function is called in three different modes.
        t1 x t2 = r ==> t1 and t2 are not diagonaled, but r is. Useful for query x key = attention_scores
        t1 x t2 = r ==> t1 is diagonaled, but t2 and r are not. Useful to compuate attantion_scores x value = context
        t1 x t2 = r ==> t1 is diagonaled and it should be transposed, but t2 and r are not diagonaled. Useful in some of
                            the calculations in the backward pass.
        '''
        dtype = str(t1.dtype).split('.')[1]
        device = t1.device.type
        assert len(t1.shape) == 4
        assert len(t1.shape) == len(t2.shape)
        assert t1.shape[:3] == t2.shape[:3]

        b = t1.shape[0]  # batch size
        n = t1.shape[1]  # sequence length
        h = t1.shape[2]  # number of heads
        m = t2.shape[3]  # hidden dimension
        max_attn = q_k_mask.size(1)
        if is_t1_diagonaled:
            assert t1.shape[3] == max_attn
            r = t1.new_empty(b, n, h, m)  # allocate spase for the result tensor
        else:
            assert not transpose_t1
            assert t1.shape[3] == m
            r = t1.new_empty(b, n, h, max_attn)  # allocate spase for the result tensor

        # gets function from memory, from disk or compiles it from scratch
        _graph_mm_function = GraphMM._get_function(dtype=dtype, device=device)

        # The last argument to this function is a little hacky. It is the size of the last dimension of the result tensor
        # We use it as a proxy to tell if t1_is_diagonaled or not (if t1 is diagonaled, result is not, and vice versa).
        # The second reason is that the lambda expression in `_compile_function` is easier to express when the shape
        # of the output is known
        # This functions computes diagonal_mm then saves the result in `r`
        if m == max_attn:
            # FIXME
            print('Error: the hidden dimension {m} shouldn\'t match number of diagonals {c}')
            assert False
        _graph_mm_function(t1, t2, r, q_k_mask, k_q_mask, max_attn, padding, transpose_t1, m if is_t1_diagonaled else max_attn)
        return r

    @staticmethod
    def _prepare_tensors(t):
        '''Fix `stride()` information of input tensor. This addresses some inconsistency in stride information in PyTorch.
        For a tensor t, if t.size(0) == 1, then the value of t.stride()[0] doesn't matter.
        TVM expects this value to be the `product(t.size()[1:])` but PyTorch some times sets it to `t.stride()[1]`.
        Here's an example to reporduce this issue:
            import torch
            print(torch.randn(1, 10).stride())
            > (10, 1)
            print(torch.randn(10, 1).t().contiguous().stride())
            > (1, 1)  # expected it to be (10, 1) as above
            print(torch.randn(10, 2).t().contiguous().stride())
            > (10, 1) # but gets the expected stride if the first dimension is > 1
        '''
        assert t.is_contiguous()
        t_stride = list(t.stride())
        t_size = list(t.size())
        # Fix wrong stride information for the first dimension. This occures when batch_size=1
        if t_size[0] == 1 and t_stride[0] == t_stride[1]:
            # In this case, the stride of the first dimension should be the product
            # of the sizes  of all other dimensions
            t_stride[0] = t_size[1] * t_size[2] * t_size[3]
            t = t.as_strided(size=t_size, stride=t_stride)
        return t

    min_seq_len = 16  # unexpected output if seq_len < 16

    @staticmethod
    def forward(ctx, t1: torch.Tensor, t2: torch.Tensor, q_k_mask, k_q_mask, is_t1_diagonaled: bool = False, padding: int = 0) -> torch.Tensor:
        '''Compuates diagonal_mm of t1 and t2.
        args: 
        t1: torch.Tensor = (batch_size, seq_len, num_attention_heads, hidden_size|number_of_diagonals).
            t1 can be a regular tensor (e.g. `query_layer`) or a diagonaled one (e.g. `attention_scores`)
        t2: torch.Tensor = (batch_size, seq_len, num_attention_heads, hidden_size). This is always a non-diagonaled
            tensor, e.g. `key_layer` or `value_layer`
        w: int = window size; number of attentions on each side of the word
        d: torch.Tensor or int = dilation of attentions per attention head. If int, the same dilation value will be used for all
            heads. If torch.Tensor, it should be 1D of lenth=number of attention heads
        is_t1_diagonaled: is t1 a diagonaled or a regular tensor
        padding: the padding value to use when accessing invalid locations. This is mainly useful when the padding
            needs to be a very large negative value (to compute softmax of attentions). For other usecases,
            please use zero padding.
        autoregressive: if true, return only the lower triangle
        returns: torch.Tensor = (batch_size, seq_len, num_attention_heads, hidden_size|number_of_diagonals)
            if t1 is diagonaed, result is non-diagonaled, and vice versa
        '''
        seq_len = t1.size(1)
        assert seq_len >= GraphMM.min_seq_len, 'avoid splitting errors by using seq_len >= {}'.format(GraphMM.min_seq_len)  # FIXME

        t1 = GraphMM._prepare_tensors(t1)
        t2 = GraphMM._prepare_tensors(t2)
        q_k_mask = GraphMM._prepare_tensors(q_k_mask)
        k_q_mask = GraphMM._prepare_tensors(k_q_mask)
        ctx.save_for_backward(t1, t2, q_k_mask, k_q_mask)
        ctx.is_t1_diagonaled = is_t1_diagonaled
        # output = t1.mm(t2)  # what would have been called if this was a regular matmul
        output = GraphMM._graph_mm(t1, t2, q_k_mask, k_q_mask, is_t1_diagonaled=is_t1_diagonaled, padding=padding)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        t1, t2, q_k_mask, k_q_mask = ctx.saved_tensors
        is_t1_diagonaled = ctx.is_t1_diagonaled
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()  # tvm requires all input tensors to be contiguous
        grad_output = GraphMM._prepare_tensors(grad_output)
        # http://cs231n.github.io/optimization-2/
        # https://pytorch.org/docs/master/notes/extending.html
        # grad_t1 = grad_output.mm(t2)  # what would have been called if this was a regular matmul
        grad_t1 = GraphMM._graph_mm(grad_output, t2, q_k_mask, k_q_mask, is_t1_diagonaled=not is_t1_diagonaled)
        # grad_t2 = grad_output.t().mm(t1)  # or `grad_t2 = t1.t().mm(grad_output).t()` because `(AB)^T = B^TA^T`
        if is_t1_diagonaled:
            grad_t2 = GraphMM._graph_mm(t1, grad_output, q_k_mask, k_q_mask, is_t1_diagonaled=True, transpose_t1=True)
        else:
            grad_t2 = GraphMM._graph_mm(grad_output, t1, q_k_mask, k_q_mask, is_t1_diagonaled=True, transpose_t1=True)
        return grad_t1, grad_t2, None, None, None, None, None


graph_mm = GraphMM.apply

class PyramidalAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout, normalize_before, q_k_mask, k_q_mask):
        super(PyramidalAttention, self).__init__()
        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_k, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_k * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_fc = nn.Dropout(dropout)
        self.q_k_mask = q_k_mask
        self.k_q_mask = k_q_mask

    def forward(self, hidden_states):
        residual = hidden_states

        hidden_states = hidden_states
        bsz, seq_len,  _ = hidden_states.size()

        q = hidden_states
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q)
        k = self.w_ks(hidden_states)
        v = self.w_vs(hidden_states)
        q /= math.sqrt(self.d_k)

        q = q.view(bsz, seq_len, self.n_head, self.d_k)
        k = k.view(bsz, seq_len, self.n_head, self.d_k)
        q = q.float().contiguous()
        k = k.float().contiguous()
        # attn_weights.size(): (batch_size, L, num_heads, 11)
        attn_weights = graph_mm_tvm(q, k, self.q_k_mask, self.k_q_mask, False, -1000000000)
        attn_weights = self.dropout_attn(F.softmax(attn_weights, dim=-1))

        v = v.view(bsz, seq_len, self.n_head, self.d_k)
        v = v.float().contiguous()
        # is_t1_diagonaled=True
        attn = graph_mm_tvm(attn_weights, v, self.q_k_mask, self.k_q_mask, True, 0)
        attn = attn.reshape(bsz, seq_len, self.n_head * self.d_k).contiguous()
        context = self.dropout_fc(self.fc(attn))
        context += residual

        if not self.normalize_before:
            context = self.layer_norm(context)

        return context
    
def get_mask(input_size, window_size, inner_size, device):
    """Get the attention mask of PAM-Naive"""
    # Get the size of all layers
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)

    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length, device=device)

    # get intra-scale mask
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = max(i - inner_window, start)
            right_side = min(i + inner_window + 1, start + all_size[layer_idx])
            mask[i, left_side:right_side] = 1

    # get inter-scale mask
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = (start - all_size[layer_idx - 1]) + (i - start) * window_size[layer_idx - 1]
            if i == ( start + all_size[layer_idx] - 1):
                right_side = start
            else:
                right_side = (start - all_size[layer_idx - 1]) + (i - start + 1) * window_size[layer_idx - 1]
            mask[i, left_side:right_side] = 1
            mask[left_side:right_side, i] = 1

    mask = (1 - mask).bool()

    return mask, all_size


def refer_points(all_sizes, window_size, device):
    """Gather features from PAM's pyramid sequences"""
    input_size = all_sizes[0]
    indexes = torch.zeros(input_size, len(all_sizes), device=device)

    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])
            inner_layer_idx = former_index - (start - all_sizes[j - 1])
            former_index = start + min(inner_layer_idx // window_size[j - 1], all_sizes[j] - 1)
            indexes[i][j] = former_index

    indexes = indexes.unsqueeze(0).unsqueeze(3)

    return indexes.long()


def get_subsequent_mask(input_size, window_size, predict_step, truncate):
    """Get causal attention mask for decoder."""
    if truncate:
        mask = torch.zeros(predict_step, input_size + predict_step)
        for i in range(predict_step):
            mask[i][:input_size+i+1] = 1
        mask = (1 - mask).bool().unsqueeze(0)
    else:
        all_size = []
        all_size.append(input_size)
        for i in range(len(window_size)):
            layer_size = math.floor(all_size[i] / window_size[i])
            all_size.append(layer_size)
        all_size = sum(all_size)
        mask = torch.zeros(predict_step, all_size + predict_step)
        for i in range(predict_step):
            mask[i][:all_size+i+1] = 1
        mask = (1 - mask).bool().unsqueeze(0)

    return mask


def get_q_k(input_size, window_size, stride, device):
    """
    Get the index of the key that a given query needs to attend to.
    """
    second_length = input_size // stride
    second_last = input_size - (second_length - 1) * stride
    third_start = input_size + second_length
    third_length = second_length // stride
    third_last = second_length - (third_length - 1) * stride
    max_attn = max(second_last, third_last)
    fourth_start = third_start + third_length
    fourth_length = third_length // stride
    full_length = fourth_start + fourth_length
    fourth_last = third_length - (fourth_length - 1) * stride
    max_attn = max(third_last, fourth_last)

    max_attn += window_size + 1
    mask = torch.zeros(full_length, max_attn, dtype=torch.int32, device=device) - 1

    for i in range(input_size):
        mask[i, 0:window_size] = i + torch.arange(window_size) - window_size // 2
        mask[i, mask[i] > input_size - 1] = -1

        mask[i, -1] = i // stride + input_size
        mask[i][mask[i] > third_start - 1] = third_start - 1
    for i in range(second_length):
        mask[input_size+i, 0:window_size] = input_size + i + torch.arange(window_size) - window_size // 2
        mask[input_size+i, mask[input_size+i] < input_size] = -1
        mask[input_size+i, mask[input_size+i] > third_start - 1] = -1

        if i < second_length - 1:
            mask[input_size+i, window_size:(window_size+stride)] = torch.arange(stride) + i * stride
        else:
            mask[input_size+i, window_size:(window_size+second_last)] = torch.arange(second_last) + i * stride

        mask[input_size+i, -1] = i // stride + third_start
        mask[input_size+i, mask[input_size+i] > fourth_start - 1] = fourth_start - 1
    for i in range(third_length):
        mask[third_start+i, 0:window_size] = third_start + i + torch.arange(window_size) - window_size // 2
        mask[third_start+i, mask[third_start+i] < third_start] = -1
        mask[third_start+i, mask[third_start+i] > fourth_start - 1] = -1

        if i < third_length - 1:
            mask[third_start+i, window_size:(window_size+stride)] = input_size + torch.arange(stride) + i * stride
        else:
            mask[third_start+i, window_size:(window_size+third_last)] = input_size + torch.arange(third_last) + i * stride

        mask[third_start+i, -1] = i // stride + fourth_start
        mask[third_start+i, mask[third_start+i] > full_length - 1] = full_length - 1
    for i in range(fourth_length):
        mask[fourth_start+i, 0:window_size] = fourth_start + i + torch.arange(window_size) - window_size // 2
        mask[fourth_start+i, mask[fourth_start+i] < fourth_start] = -1
        mask[fourth_start+i, mask[fourth_start+i] > full_length - 1] = -1

        if i < fourth_length - 1:
            mask[fourth_start+i, window_size:(window_size+stride)] = third_start + torch.arange(stride) + i * stride
        else:
            mask[fourth_start+i, window_size:(window_size+fourth_last)] = third_start + torch.arange(fourth_last) + i * stride

    return mask


def get_k_q(q_k_mask):
    """
    Get the index of the query that can attend to the given key.
    """
    k_q_mask = q_k_mask.clone()
    for i in range(len(q_k_mask)):
        for j in range(len(q_k_mask[0])):
            if q_k_mask[i, j] >= 0:
                k_q_mask[i, j] = torch.where(q_k_mask[q_k_mask[i, j]] ==i )[0]

    return k_q_mask

###########################################################################################################################
# Encoder-Decoder #########################################################################################################

class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True, use_tvm=False, q_k_mask=None, k_q_mask=None):
        super(EncoderLayer, self).__init__()
        self.use_tvm = use_tvm
        if use_tvm:
            from .PAM_TVM import PyramidalAttention
            self.slf_attn = PyramidalAttention(n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before, q_k_mask=q_k_mask, k_q_mask=k_q_mask)
        else:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, slf_attn_mask=None):
        if self.use_tvm:
            enc_output = self.slf_attn(enc_input)
            enc_slf_attn = None
        else:
            enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, Q, K, V, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            Q, K, V, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=window_size,
                                  stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Conv_Construct(nn.Module):
    """Convolution CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(Conv_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, window_size),
                ConvLayer(d_model, window_size),
                ConvLayer(d_model, window_size)
                ])
        else:
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, window_size[0]),
                ConvLayer(d_model, window_size[1]),
                ConvLayer(d_model, window_size[2])
                ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.permute(0, 2, 1)
        all_inputs.append(enc_input)

        for i in range(len(self.conv_layers)):
            enc_input = self.conv_layers[i](enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs


class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size)
                ])
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(ConvLayer(d_inner, window_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
        self.up = Linear(d_inner, d_model)
        self.down = Linear(d_model, d_inner)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):

        temp_input = self.down(enc_input).permute(0, 2, 1)
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)
            all_inputs.append(temp_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.up(all_inputs)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)

        all_inputs = self.norm(all_inputs)

        return all_inputs


class MaxPooling_Construct(nn.Module):
    """Max pooling CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(MaxPooling_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList([
                nn.MaxPool1d(kernel_size=window_size),
                nn.MaxPool1d(kernel_size=window_size),
                nn.MaxPool1d(kernel_size=window_size)
                ])
        else:
            self.pooling_layers = nn.ModuleList([
                nn.MaxPool1d(kernel_size=window_size[0]),
                nn.MaxPool1d(kernel_size=window_size[1]),
                nn.MaxPool1d(kernel_size=window_size[2])
                ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs


class AvgPooling_Construct(nn.Module):
    """Average pooling CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(AvgPooling_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList([
                nn.AvgPool1d(kernel_size=window_size),
                nn.AvgPool1d(kernel_size=window_size),
                nn.AvgPool1d(kernel_size=window_size)
                ])
        else:
            self.pooling_layers = nn.ModuleList([
                nn.AvgPool1d(kernel_size=window_size[0]),
                nn.AvgPool1d(kernel_size=window_size[1]),
                nn.AvgPool1d(kernel_size=window_size[2])
                ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs


class Predictor(nn.Module):

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data):
        out = self.linear(data)
        out = out
        return out

class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.d_model = opt.d_model
        self.model_type = opt.model
        self.window_size = opt.window_size
        self.truncate = opt.truncate
        if opt.decoder == 'attention':
            self.mask, self.all_size = get_mask(opt.input_size, opt.window_size, opt.inner_size, opt.device)
        else:
            self.mask, self.all_size = get_mask(opt.input_size+1, opt.window_size, opt.inner_size, opt.device)
        self.decoder_type = opt.decoder
        if opt.decoder == 'FC':
            self.indexes = refer_points(self.all_size, opt.window_size, opt.device)

        if opt.use_tvm:
            assert len(set(self.window_size)) == 1, "Only constant window size is supported."
            padding = 1 if opt.decoder == 'FC' else 0
            q_k_mask = get_q_k(opt.input_size + padding, opt.inner_size, opt.window_size[0], opt.device)
            k_q_mask = get_k_q(q_k_mask)
            self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout, \
                    normalize_before=False, use_tvm=True, q_k_mask=q_k_mask, k_q_mask=k_q_mask) for i in range(opt.n_layer)
                ])
        else:
            self.layers = nn.ModuleList([
                EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout, \
                    normalize_before=False) for i in range(opt.n_layer)
                ])

        if opt.embed_type == 'CustomEmbedding':
            self.enc_embedding = CustomEmbedding(opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout)
        else:
            self.enc_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)

        self.conv_layers = eval(opt.CSCM)(opt.d_model, opt.window_size, opt.d_bottleneck)

    def forward(self, x_enc, x_mark_enc):

        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        if self.decoder_type == 'FC':
            indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        elif self.decoder_type == 'attention' and self.truncate:
            seq_enc = seq_enc[:, :self.all_size[0]]

        return seq_enc

class Decoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, opt, mask):
        super().__init__()

        self.model_type = opt.model
        self.mask = mask

        self.layers = nn.ModuleList([
            DecoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout, \
                normalize_before=False),
            DecoderLayer(opt.d_model, opt.d_inner_hid, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout, \
                normalize_before=False)
            ])

        if opt.embed_type == 'CustomEmbedding':
            self.dec_embedding = CustomEmbedding(opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout)
        else:
            self.dec_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)

    def forward(self, x_dec, x_mark_dec, refer):
        dec_enc = self.dec_embedding(x_dec, x_mark_dec)

        dec_enc, _ = self.layers[0](dec_enc, refer, refer)
        refer_enc = torch.cat([refer, dec_enc], dim=1)
        mask = self.mask.repeat(len(dec_enc), 1, 1).to(dec_enc.device)
        dec_enc, _ = self.layers[1](dec_enc, refer_enc, refer_enc, slf_attn_mask=mask)

        return dec_enc
    
class Pyraformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.predict_step = opt.predict_step
        self.d_model = opt.d_model
        self.input_size = opt.input_size
        self.decoder_type = opt.decoder
        self.channels = opt.enc_in

        self.encoder = Encoder(opt)
        if opt.decoder == 'attention':
            mask = get_subsequent_mask(opt.input_size, opt.window_size, opt.predict_step, opt.truncate)
            self.decoder = Decoder(opt, mask)
            self.predictor = Predictor(opt.d_model, opt.enc_in)
        elif opt.decoder == 'FC':
            self.predictor = Predictor(4 * opt.d_model, opt.predict_step * opt.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, pretrain):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        if self.decoder_type == 'attention':
            enc_output = self.encoder(x_enc, x_mark_enc)
            dec_enc = self.decoder(x_dec, x_mark_dec, enc_output)

            if pretrain:
                dec_enc = torch.cat([enc_output[:, :self.input_size], dec_enc], dim=1)
                pred = self.predictor(dec_enc)
            else:
                pred = self.predictor(dec_enc)
        elif self.decoder_type == 'FC':
            enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]
            pred = self.predictor(enc_output).view(enc_output.size(0), self.predict_step, -1)

        return pred
    
###########################################################################################################################
# Running Experiments #####################################################################################################
    
def train_epoch(model, train_dataset, training_loader, optimizer, opt, epoch):
    """ Epoch operation in training phase. """

    model.train()
    total_loss = 0
    total_pred_number = 0
    for batch in tqdm(training_loader, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        # prepare data
        batch_x, batch_y, batch_x_mark, batch_y_mark, mean, std = map(lambda x: x.float().to(opt.device), batch)
        # prepare predict token
        dec_inp = torch.zeros_like(batch_y).float()

        optimizer.zero_grad()

        # forward
        if opt.decoder == 'attention':
            if opt.pretrain and epoch < 1:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, True)
                batch_y = torch.cat([batch_x, batch_y], dim=1)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)
        elif opt.decoder == 'FC':
            # Add a predict token into the history sequence
            predict_token = torch.zeros(batch_x.size(0), 1, batch_x.size(-1), device=batch_x.device)
            batch_x = torch.cat([batch_x, predict_token], dim=1)
            batch_x_mark = torch.cat([batch_x_mark, batch_y_mark[:, 0:1, :]], dim=1)
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)


        # determine the loss function
        if opt.hard_sample_mining and not (opt.pretrain and epoch < 1):
            topk = sample_mining_scheduler(epoch, batch_x.size(0))
            criterion = TopkMSELoss(topk)
        elif opt.loss == 'mse':
            criterion = torch.nn.MSELoss(reduction='none')
        # if inverse, both the output and the ground truth are denormalized.
        if opt.inverse:
            outputs, batch_y = train_dataset.inverse_transform(outputs, batch_y, mean, std)
            print(batch_y[0])
        # compute loss
        losses = criterion(outputs, batch_y)
        loss = losses.mean()
        loss.backward()

        """ update parameters """
        optimizer.step()
        total_loss += losses.sum().item()
        total_pred_number += losses.numel()

    return total_loss / total_pred_number


def eval_epoch(model, test_dataset, test_loader, opt, epoch):
    """ Epoch operation in evaluation phase. """

    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in tqdm(test_loader, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            batch_x, batch_y, batch_x_mark, batch_y_mark, mean, std = map(lambda x: x.float().to(opt.device), batch)
            dec_inp = torch.zeros_like(batch_y).float()

            # forward
            if opt.decoder == 'FC':
                # Add a predict token into the history sequence
                predict_token = torch.zeros(batch_x.size(0), 1, batch_x.size(-1), device=batch_x.device)
                batch_x = torch.cat([batch_x, predict_token], dim=1)
                batch_x_mark = torch.cat([batch_x_mark, batch_y_mark[:, 0:1, :]], dim=1)
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)

            # if inverse, both the output and the ground truth are denormalized.
            if opt.inverse:
                outputs, batch_y = test_dataset.inverse_transform(outputs, batch_y, mean, std)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    #np.save('/predictions_'+opt.data+"_"+str(opt.predict_step)+"_"+str(iter_index)+".npy", preds)
    #np.save('/trues_'+opt.data+"_"+str(opt.predict_step)+"_"+str(iter_index)+".npy", trues)

    print('test shape:{}'.format(preds.shape))
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('Epoch {}, mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(epoch, mse, mae, rmse, mape, mspe))

    return mse, mae, rmse, mape, mspe


def train(model, optimizer, scheduler, opt, model_save_dir):
    """ Start training. """

    best_mse = 100000000

    """ prepare dataloader """
    training_dataloader, train_dataset, test_dataloader, test_dataset = prepare_dataloader(opt)

    best_metrics = []
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_mse = train_epoch(model, train_dataset, training_dataloader, optimizer, opt, epoch_i)
        print('  - (Training) '
              'MSE: {mse: 8.5f}'
              'elapse: {elapse:3.3f} min'
              .format(mse=train_mse, elapse=(time.time() - start) / 60))

        mse, mae, rmse, mape, mspe = eval_epoch(model, test_dataset, test_dataloader, opt, epoch_i)

        scheduler.step()

        current_metrics = [float(mse), float(mae), float(rmse), float(mape), float(mspe)]
        if best_mse > mse:
            best_mse = mse
            best_metrics = current_metrics
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metrics": best_metrics
                },
                model_save_dir
            )

    return 


def evaluate(model, opt, model_save_dir):
    """Evaluate preptrained models"""
    best_mse = 100000000

    """ prepare dataloader """
    _, _, test_dataloader, test_dataset = prepare_dataloader(opt)

    """ load pretrained model """
    checkpoint = torch.load(model_save_dir)["state_dict"]
    model.load_state_dict(checkpoint)

    best_metrics = []
    mse, mae, rmse, mape, mspe = eval_epoch(model, test_dataset, test_dataloader, opt, 0)

    current_metrics = [float(mse), float(mae), float(rmse), float(mape), float(mspe)]
    if best_mse > mse:
        best_mse = mse
        best_metrics = current_metrics

    return best_metrics

###########################################################################################################################
# Our Simple User Interface ###############################################################################################

class PyraformerTS():
    '''
    Our custom wrapper class to provide an user-friendly interface for fitting and testing of the Pyraformer model. 
    For simplicity of use, methods included align with naming used by Keras.  
    '''
    def __init__(self, model='Pyraformer'):
        if model != 'Pyraformer':
            raise ValueError("Model not supported. Please use 'Pyraformer'.")
        # Initialize dot dictionary
        self.args = dotdict()
        # running mode
        self.args.eval = False
        # Path parameters
        self.args.data = 'SYNTHh1'
        self.data = 'SYNTHh1'
        self.args.root_path = './SYNTHDataset'
        self.args.data_path ='SYNTHh1.csv' 
        # Dataloader parameters.
        self.args.input_size = 96
        self.args.predict_step = 24
        self.args.inverse = False
        #dataset dependent parameters
        parser = argparse.ArgumentParser()
        datapars = dataset_parameters(parser,self.args.data)
        self.args.enc_in =parser.enc_in
        self.args.dec_in =parser.dec_in
        self.args.covariate_size = parser.covariate_size
        self.args.seq_num = parser.seq_num
        self.args.embed_type = parser.embed_type
        # Architecture selection.
        self.args.model = model
        self.args.decoder = 'FC' # selection: [FC, attention]
        # Training parameters.
        self.args.epoch = 8
        self.args.batch_size = 32
        self.args.pretrain = False
        self.args.hard_sample_mining = False
        self.args.dropout = 0.05
        self.args.lr = 1e-4
        self.args.lr_step = 0.1
        # Common Model parameters.
        self.args.d_model = 512
        self.args.d_inner_hid = 512
        self.args.d_k = 128
        self.args.d_v = 128
        self.args.d_bottleneck = 128
        self.args.n_head = 4
        self.args.n_layer = 4
        # Pyraformer parameters.
        self.args.window_size = [4, 4, 4]
        self.args.inner_size = 3
        # CSCM structure. selection: [Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct]
        self.args.CSCM = 'Bottleneck_Construct'
        self.args.truncate = False
        self.args.use_tvm = False
        # Experiment repeat times.
        self.args.iter_num = 1 
        if torch.cuda.is_available():
            self.args.device = torch.device("cuda")
        else:
            self.args.device = torch.device('cpu')
        """ prepare model """
        self.model = Pyraformer(self.args).eval()
        self.model.to(self.args.device)

    def compile(self, learning_rate=1e-4, loss='mse'):
        '''
        Compiles the Pyraformer model for training.
        Args:
            learning_rate (float): Learning rate to be used. Default: '1e-4'.
            loss (str): Loss function to be used. Default: 'mse'.
            early_stopping_patience (int): Amount of epochs to beak training loop after no validation performance improvement. Default: 3.
        '''
        if loss != 'mse':
            raise ValueError("Loss function not supported. Please use 'mse'.")
        self.args.lr = learning_rate
        self.args.loss = loss
      

    def fit(self, data = "SYNTHh1", data_root_path = "./SYNTHDataset/", batch_size = 32 , epochs = 5 , pred_len = 24):
        """ Main function. """
        possible_datasets = ['SYNTHh1', 'SYNTHh2', 'SYNTH_additive' , 'SYNTH_multiplicative', 'DEWINDh_large', 'DEWINDh_small','SYNTH_multiplicative_reversals','SYNTH_additive_reversals']
        if data not in possible_datasets:
            raise ValueError("Dataset not supported. Please use one of the following: 'SYNTHh1', 'SYNTHh2', SYNTH_additive', 'SYNTH_multiplicative', 'DEWINDh_large', 'DEWINDh_small'.")
        possible_predlens = [24, 48, 96, 168, 336, 720]
        if pred_len not in possible_predlens:
            raise ValueError('Prediction length outside current experiment scope. Please use either 24, 48, 96, 168, 336, 720.')
        self.args.data = data
        self.args.root_path = data_root_path
        self.args.data_path = f'{self.data}.csv'
        self.args.epoch = epochs
        self.args.batch_size = batch_size
        self.args.predict_step = pred_len
        self.model_save_dir = 'Pyraformer/trained_models/{}/{}/{}.pth'.format(self.args.data, self.args.input_size, 
                                                                           self.args.predict_step)
        os.makedirs(self.model_save_dir, exist_ok=True)
        if self.args.eval:
            best_metrics = evaluate(self.model, self, self.model_save_dir)
        else:
            print('Beginning to fit the model with the following arguments:')
            print(f'{self.args}')
            print('='*150)
            """ optimizer and scheduler """
            optimizer = optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()), self.args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=self.args.lr_step)
            self.model = train(self.model, optimizer, scheduler, self.args, self.model_save_dir)
        return 
  




