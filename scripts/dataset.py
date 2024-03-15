'''
Contains a class for assembling earth systems data into a coherent table.
'''

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from data import grab_dataset

class EarthSystemsDataset(Dataset):
    '''
    pyTorch Dataset to supply a neural network with time series data
    '''

    def __init__(self, data_var_names, y_vals, add_index=False, val_frac=None,
                 timeframe='monthly', lags=2, mode='rnn', normalize=False):
        '''
        :data_var_names: (list-like of str) Names of variable names to use. Will be passed
            to the `grab_dataset` function.
        :y_vals: (list-like of str) Names of the y variables to predict.
        :add_index: (bool) whether or not to include the time index as a variable
        :val_frac: (float) Fraction of the end of the time series to use as validation data.
        :timeframe: (str) Either 'monthly' or 'yearly'; describes the frequency of samples
            to be used in the data.
        :lags: (int) Number of time step lags to use
        :mode: (str) Should be one of ['rnn', 'ann']. Since an RNN and ANN require data in
            different formats, this argument will indicate that.
        is formatted.
        :normalize: (bool) Whether to normalize the data to [0,1] or not
        '''
        assert mode in ('rnn', 'ann'), \
            f'ERROR: {mode} is not a valid value for `mode`. It should be either "rnn" or "ann"'
        assert (isinstance(val_frac, float) and 0 <= val_frac < 1) or val_frac is None, \
            f'ERROR {val_frac} is not a valid value for `val_frac`. It should be between 0 and 1'

        self.data_var_names = data_var_names
        self.y_vals = y_vals
        self.add_index = add_index
        self.val_frac = val_frac
        self.timeframe = timeframe
        self.lags = lags
        self.mode = mode
        self.normalize = normalize
        
        '''
        self.data is thte current working dataset. It can be a training set (self.train_data),
        or a validation set (self.val_data). We can also select all possible data (self.full_data),
        if we want to find the error over the entire dataset.

        It will select full data by default.
        '''

        raw_datasets = [grab_dataset(var_name, timeframe=timeframe) for var_name in data_var_names]
        self.full_data = EarthSystemsDataset.trim_data(raw_datasets) 

        if add_index:
            self.full_data['index'] = list(range(len(self.full_data)))
            self.data_var_names = self.data_var_names + ['index']

        if normalize:
            normalizer = MinMaxScaler()
            self.full_data = pd.DataFrame(normalizer.fit_transform(self.full_data),
                                          index=self.full_data.index,
                                          columns=self.full_data.columns)

        self.data = self.full_data

        if val_frac:
            train_stop = int((1-val_frac) * self.data.shape[0])
            self.train_data = self.data.iloc[:train_stop]
            self.val_data = self.data.iloc[train_stop:]
        else:
            self.train_data = self.data
            self.val_data = None

        self.data = self.train_data

        
    def __len__(self):
        return self.data.shape[0] - self.lags

    def __getitem__(self, index):
        '''
        Returns the last p time steps for all variables (where p is the number of lags)
            For ANN: Flattens the data into a 1D vector, in the format 
            [last p values of x1, last p values of x2, ..., last p values of xn]
            
            For RNN: Returns data in a 2D matrix, with size (p, number of variables)
        '''

        result = self.data.iloc[index:index+self.lags]
        targets = self.data[self.y_vals].iloc[index+self.lags].to_numpy()

        if self.mode == 'rnn':
            return result.to_numpy(), targets
        else:
            return result.to_numpy().T.flatten(), targets
        
    def train_mode(self):
        # Prepares the dataset for training
        self.data = self.train_data

    def val_mode(self):
        # Prepares the dataset for validation
        self.data = self.val_data

    def full_mode(self):
        # Uses all of the data (in self.data)
        self.data = self.full_data

    def split(self, train_ind, val_ind):
        '''
        Manually choose validation and training data using a year/month index

        return: None
        '''
        self.train_data = self.full_data.loc[train_ind]
        if val_ind is not None:
            self.val_data = self.full_data.loc[val_ind]
    
    def print_info(self):
        print('Earth Systems Time Series Data Overview.')
        print(f'Timeframe={self.timeframe} for {self.mode.upper()} training. Using {self.lags} lags.')
        print(f'Training data: {self.train_data.shape[0] - self.lags} points. '+
              f'Validation data: {self.val_data.shape[0] if self.val_data is not None else 0} points ({self.val_frac*100.}%).')
        print(f'Normalizing data to [0,1]' if self.normalize else 'Not normalizing data')
        print('Variables:')
        for var_name in self.full_data.columns:
            print(f'  {var_name}: mean={self.full_data[var_name].mean():.5f}, std={self.full_data[var_name].std():.5f}')
        print('Predicting for:')
        for var_name in self.y_vals:
            print(f'  {var_name}')


    @staticmethod
    def trim_data(all_data):
        '''
        This function trims the time series data so that they start and end on the same date.

        :all_data (list-like of pd.DataFrame): List of DataFrames to trim

        return: unified DataFrame of all the trimmed data
        '''

        trimmed_data = []

        for df in all_data:
            trimmed_data.append(df.set_index(['year', 'month']))

        return trimmed_data[0].join(trimmed_data[1:], how='inner')
                
            