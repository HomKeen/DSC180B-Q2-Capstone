'''
Helper functions and classes for constructing, training, and saving pyTorch neural network (NN) models. 
'''

import torch
from torch import nn, jit
from torch.utils.data import DataLoader
from torch.nn import Flatten, LayerNorm, GRUCell, Dropout
import sys

from util import Timer

class FilteringLayer(nn.Module):
    '''
    A filter that acts as a "gate" for preceding variables. Should be trained with L1 regularization.
    '''
    def __init__(self, shape):
        super().__init__()
        self.filter = nn.Parameter(torch.ones(shape))

    def forward(self, x):
        return x * self.filter

class EarthSystemsNN(nn.Module):
    '''
    A standard feedforward neural network
    '''
    def __init__(self, sequence):
        super().__init__()
        self.network = sequence

    def forward(self, x):
        return self.network(x)
    
class EarthSystemsRNN(nn.Module):
    '''
    A recurrent neural network. Can accept any type of RNN layers, such as LSTM or GRU. It also 
    has the option to utilize all outputs all time steps or only the last time step, when passing 
    to the dense layers.
    '''
    def __init__(self, rnn_layers, fc_layers, last_only=False):
        super().__init__()

        self.rnn_layers = rnn_layers
        self.fc_layers = fc_layers
        self.flatten = Flatten()
        self.last_only = last_only


    def forward(self, x):
        # x has shape (batch_size, lags, features)
        res = x
        for rnn_layer in self.rnn_layers:
            res, _ = rnn_layer(res)
        
        if self.last_only:
            res = res[:, -1, :]
        else:
            res = self.flatten(res)
        return self.fc_layers(res)
    
class GrangerComponent(nn.Module):
    '''
    A RNN for discovering Granger causality. This network predicts for only one output variable
    '''
    def __init__(self, rnn_layers, fc_layers, input_size, lags, reg_lags=True, reg_features=True, last_only=False):
        '''
        :rnn_layers: (nn.Sequential) Sequential of RNN-type layers.
        :fc_layers: (nn.Sequential) Sequential of dense (i.e., fully connected) layers.
        :input_size: (int) Number of features in the input.
        :lags: (int) Number of time lags to use.
        :reg_lags: (bool) Whether or not to regularize to discover Granger causality for lags.
        :reg_features: (bool) Whether or not to regularize to discover Granger causality for features. 
        :last_only: (bool) Whether or not to only use the last time step output of each RNN layer. Setting this
                    to true will drastically reduce runtime.
        '''
        super().__init__()

        self.feature_filter = (FilteringLayer(input_size) if reg_features else None)
        self.lags_filter = (FilteringLayer((lags, 1)) if reg_lags else None)

        self.rnn_layers = rnn_layers
        self.fc_layers = fc_layers
        self.flatten = Flatten()
        
        self.reg_lags = reg_lags
        self.reg_features = reg_features
        self.last_only = last_only

    def forward(self, x):
        # x has shape (batch_size, lags, features)
        res = (self.feature_filter(x) if self.reg_features else x)
        res = (self.lags_filter(res) if self.reg_lags else res)

        for rnn_layer in self.rnn_layers:
            res, _ = rnn_layer(res)
        
        if self.last_only:
            res = res[:, -1, :]
        else:
            res = self.flatten(res)
        
        return self.fc_layers(res)

    def regularize(self, lam):
        # force some of the filtering features to be 0 with L1 regularization, helping us decide Granger causality
        reg_features = (lam * torch.abs(self.feature_filter.filter).sum() if self.reg_features else 0)
        reg_lags = (lam * torch.abs(self.lags_filter.filter).sum() if self.reg_lags else 0)

        return reg_features + reg_lags
    
class GrangerRNN(nn.Module):
    '''
    A RNN for discovering Granger causality. This network predicts for all variables at once via multiple 
    independent GrangerComponents. Combining them into a single loss function which vastly 
    improves training time.
    '''
    def __init__(self, layer_func, n_models, input_size, lags, reg_lags=True, reg_features=True, last_only=False):
        '''
        :layer_func: (function) A function that should return a tuple (rnn_layers, fc_layers) which are each
                    nn.Sequential of RNN-type layers and dense layers, respectively. This function should take
                    an input size and output size as arguments.
        :n_models: (int) Number of variables to predict for; one for each model.
        :input_size: (int) Number of features in the input
        :lags: (int) Number of time lags to use
        :reg_lags: (bool) Whether or not to regularize to discover Granger causality for lags
        :reg_features: (bool) Whether or not to regularize to discover Granger causality for features 
        :last_only: (bool) Whether or not to only use the last time step output of each RNN layer. Setting this
                    to true will drastically reduce runtime.
        '''

        super().__init__()

        self.models = nn.ModuleList()
        for _ in range(n_models):
            rnn_layers, fc_layers = layer_func(input_size, 1)
            self.models.append(
                GrangerComponent(rnn_layers, fc_layers, input_size, lags, reg_lags=reg_lags, 
                                 reg_features=reg_features, last_only=last_only)
                )

    def forward(self, x):
        all_out = [model(x) for model in self.models]
        
        return torch.stack(all_out, dim=1).squeeze()
    
    def regularize(self, lam):
        reg_loss = 0
        for model in self.models:
            reg_loss += model.regularize(lam)
        return reg_loss  
    


class GRULayerNorm(jit.ScriptModule):
    '''
    One GRU layer that applies layer normalization after each time step.
    '''

    def __init__(self, input_size, hidden_size, dropout=None, bias=True, device=None, dtype=None):
        
        super().__init__()
        self.gru_cell = GRUCell(input_size,
                           hidden_size,
                           bias=bias,
                           device=device,
                           dtype=dtype)
        
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = (Dropout(dropout) if dropout is not None else None)

        self.hidden_size = hidden_size
        self.dtype = dtype

    @jit.script_method
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        output = torch.jit.annotate(List[torch.Tensor], [])
        hx = torch.zeros(seq_len, self.hidden_size, device=x.device)

        for i in range(len(x)):
            hx = self.layer_norm(self.gru_cell(x[i], hx))
            output += [hx]
        
        return self.dropout(torch.stack(output)) if self.dropout is not None else torch.stack(output), hx.unsqueeze(0)



    
class Trainer:
    '''
    Helper class for easily training a model. It can:
        - Train a model with only a few lines
        - Print information while training 
        - Save a model and training info
        - Train from scratch or resume from a previous iteration
        - Compute training or validation error
        - Use desired batch size, device, learning rate, etc.
    '''

    def __init__(self, model, loss_fn, optimizer, dataset, batch_size=1, save_path=None, 
                 preload=None, device='cpu', save_freq=1, val_freq='batch', verbose=True):
        
        '''
        :model (torch.nn.Module): Trainable model
        :loss_fn: torch loss function
        :optimizer: torch optimizer, already initialized with model parameters
        :dataset: An EarthSystemsDataset instance, already initialized
        :batch_size (int): Number of samples per batch
        :save_path (str): Path to .pth file to save model to
        :device (str or torch.device): Device (a cpu or a gpu) to train on
        :save_freq (int): Save model to file every {save_freq} epochs
        :val_freq (str or None): one of: 
                'batch' -- compute and print validation error after each group of batches; 
                'epoch' -- compute and print validation error after each epoch; 
                None -- don't compute validation error until the very end
        :verbose: (bool) Whether to print training information at the beginning 
        '''

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.batch_size = batch_size
        self.save_path = save_path
        self.device = device
        self.save_freq = save_freq
        self.val_freq = val_freq
        self.verbose = verbose

        # save lists of training/validation errors (updated every epoch)
        self.train_errors = []
        self.val_errors = []

        self.cur_epoch = 0

        dataset.train_mode()
        self.dataset = dataset
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        if preload:
            self.load_model(preload)

        if verbose:
            print('\nTRAINING OVERVIEW\n-------------------------------')
            print('DATA:\n')
            self.dataset.print_info()
            print('-------------------------------')
            print('OPTIMIZER:\n', optimizer, '\n-------------------------------') 
            print('LOSS FUNCTION:\n', loss_fn, '\n-------------------------------')
            print('MODEL ARCHITECTURE:\n', model, '\n-------------------------------')
            print(f'OTHER:')
            print(f'Loaded saved weights from {preload}' if preload else f'Not loading weights')
            print(f'Will save model to {save_path}' if save_path else 'WARNING: WILL NOT SAVE MODEL')
            print(f'Training with batch size of {batch_size}')
            print(f'Running on device {device}')

            print('Ready to train\n')
            sys.stdout.flush()
    

    def train_loop(self, lam=None):
        '''
        Trains the model for one epoch (a single pass through the data).

        :lam: L1 regularization constant for the Granger filter(s)

        return: None
        '''
        self.model.train()
        self.dataset.train_mode()
        size, n_batches, batch_size = len(self.data_loader.dataset), len(self.data_loader), self.data_loader.batch_size
        timer = Timer()
       
        for batch, (X, y) in enumerate(self.data_loader):
            self.model.train()
            self.dataset.train_mode()

            X = X.to(self.device, dtype=torch.float)
            y = y.to(self.device, dtype=torch.float)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            if lam:
                loss += self.model.regularize(lam)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if (batch % 5 == 0 or batch == n_batches-1) and batch > 0:
                loss, current = loss.item(), (batch+1) * batch_size
                elapsed = timer.get(reset=True)
                print(f'Batch {batch+1:>3d}/{n_batches}, loss: {loss:>7f}  [{current:>5d}/{size:>5d}] ({elapsed:.3f}s)', end=' ')

                if self.dataset.val_frac and self.val_freq == 'batch':
                    self.print_error(types=['val'], end='')
                print('')
            
    def run_training(self, epochs, lam=None):
        '''
        Train the model for a specified number of epochs.

        :lam: L1 regularization constant for the Granger filter(s)

        return: None
        '''
        # lam tells us that the model has weights to be custom regularized with its regularize() function
        main_timer, loop_timer = Timer(), Timer()
        print(f'Beginning training from epoch {self.cur_epoch+1} for {epochs} epochs')

        for _ in range(epochs):
            self.cur_epoch += 1
            print(f'Epoch {self.cur_epoch}\n-------------------------------')
            self.train_loop(lam=lam)

            if self.save_path and self.cur_epoch % self.save_freq == 0:
                self.save_model()
                print(f'Saved to {self.save_path}')

            # save and/or print errors
            if self.dataset.val_frac and self.val_freq == 'epoch':
                self.print_error(record=True, subsets=['val'])
            else:
                self.train_errors.append(self.get_error('train'))
                self.val_errors.append(self.get_error('val'))
            
            print(f'Took {loop_timer.get(reset=True):.3f}s total\n-------------------------------\n')
            sys.stdout.flush()
            
        if self.dataset.val_frac and self.val_freq is None:
            self.print_error()

        print(f'Took {main_timer.get():.4f} seconds')
        print('Done!')
        sys.stdout.flush()

    
    def get_error(self, subset):
        '''
        Returns the error of the model on either the training or validation set

        :subset: (str) One of 'train' or 'val', indicating error for the training or validation set

        :return: (float) The error on the specified subset of data
        '''
        self.model.eval()
        if subset == 'val':
            return self.get_val_error()[0]
        else:
            return self.get_train_error()[0]
        
    def get_train_error(self):
        '''
        Get the train error as well as the predictions on the train data

        return: tuple of (error (float), predictions (list)) 
        '''
        self.dataset.train_mode()

        n_batches = len(self.data_loader)
        all_pred = []
        total_loss = 0.

        with torch.no_grad():
            for X, y in self.data_loader:
                X = X.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.float)
                pred = self.model(X)
                all_pred.append(pred)
                total_loss += self.loss_fn(pred, y).item()
                
        self.model.train()
        self.dataset.train_mode()
        return total_loss / n_batches, all_pred
    
    def get_val_error(self):
        '''
        Get the validation error as well as the predictions on the validation data. The validation
        predictions are computed by assuming we know nothing past the training data; the model's
        predictions serve as inputs to the next time step.

        return: tuple of (error (float), predictions (list)) 
        '''
        prev = torch.from_numpy(self.dataset.train_data.iloc[-self.dataset.lags:].to_numpy()).to(self.device, dtype=torch.float)
        pred = []
        true = []

        ind = self.dataset.data.columns.get_indexer(self.dataset.y_vals)

        if self.dataset.val_data is None:
            return None
        
        with torch.no_grad():
            for i in range(self.dataset.val_data.shape[0]):
                new_pred = self.model(prev[-self.dataset.lags:].unsqueeze(0))
                pred.append(new_pred)
                prev = torch.cat((prev, 
                                torch.from_numpy(self.dataset.val_data.iloc[i].to_numpy())
                                .unsqueeze(0)
                                .to(self.device, dtype=torch.float)))
                prev[-1, ind] = new_pred
                true.append(self.dataset.val_data.iloc[i][ind])
        return self.loss_fn(torch.stack(pred).squeeze(), torch.tensor(true).squeeze().to(self.device)), pred


    
    def print_error(self, record=False, subsets=['train' ,'val'], end='\n'):
        '''
        Prints training and/or validation errors. Can also record the error for retrieval later.

        :record: (bool) Whether or not to save the error to self.train_error or self.val_error
        :subsets: (list) Must contain 'train', 'val', or both; specifies which portions of 
                    the dataset to print/record the error for
        :end: (str) What to print at the end after the error. Used for formatting purposes.

        return: None
        '''
        if 'train' in subsets:
            train_err = self.get_error("train")
            if record:
                self.train_errors.append(train_err)
            print(f'train loss: {train_err:>7f}', end=end)
        if 'val' in subsets:
            val_err = self.get_error("val")
            if record:
                self.val_errors.append(val_err)
            print(f'val loss: {val_err:>7f}', end=end)

    def save_model(self, all_params=True):
        '''
        Save the model to the file specified by `self.save_path`. 

        :all_params: (bool) Whether or not to save all information including training info, or just
                    the model state_dict. The latter is helpful if we do not wish to continue training,
                    only for testing and predicting.
        
        return: None
        '''
        if all_params:
            torch.save({
                'epoch': self.cur_epoch,
                'train_ind': self.dataset.train_data.index,
                'val_ind': self.dataset.val_data.index if self.dataset.val_data is not None else None,
                'train_errors': self.train_errors,
                'val_errors': self.val_errors,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.save_path)
        else:
            torch.save(self.model.state_dict(), self.save_path)

    def load_model(self, model_path):
        '''
        Load a previously trained model to continue training. The previous model must have been saved 
        with `all_params=True`.

        :model_path: (str) Path to the model to load

        return: None
        '''
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.cur_epoch = checkpoint['epoch']
        
        val_ind = checkpoint['val_ind']
        train_ind = checkpoint['train_ind']

        self.train_errors = checkpoint['train_errors']
        self.val_errors = checkpoint['val_errors']

        if self.verbose:
            print(f'\tUsing {len(val_ind) if val_ind is not None else 0} validation points')
            print(f'\tUsing {len(train_ind)} training points')

        self.dataset.split(train_ind, val_ind)

    def set_learning_rate(self, alpha):
        '''
        Manually set the learning rate.

        :alpha: (float) The new learning rate
        '''
        for group in self.optimizer.param_groups:
            group['lr'] = alpha
        
