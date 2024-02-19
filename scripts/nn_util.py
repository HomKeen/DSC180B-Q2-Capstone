'''
Helper functions for constructing, training, and saving pyTorch neural network (NN) models. 
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Flatten

from util import Timer

class EarthSystemsNN(nn.Module):
    def __init__(self, sequence):
        super().__init__()
        self.network = sequence

    def forward(self, x):
        return self.network(x)
    
class EarthSystemsRNN(nn.Module):
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
    
class Trainer:
    '''
    Helper class for easily training a model
    '''

    def __init__(self, model, loss_fn, optimizer, dataset,
                 batch_size=1, save_path=None, preload=None, device='cpu'):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.batch_size = batch_size
        self.save_path = save_path
        self.device = device

        self.cur_epoch = 0

        dataset.train_mode()
        self.dataset = dataset
        self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        print('\nTRAINING OVERVIEW\n-------------------------------')
        print('OPTIMIZER:\n', optimizer, '\n-------------------------------') 
        print('LOSS FUNCTION:\n', loss_fn, '\n-------------------------------')
        print('MODEL ARCHITECTURE:\n', model, '\n-------------------------------')
        print(f'OTHER:')
        if preload:
            print(f'Loading saved weights from {preload}')
            self.load_model(preload)
        print(f'Will save model to {save_path}' if save_path else 'WARNING: WILL NOT SAVE MODEL')
        print(f'Training with batch size of {batch_size}')
        print(f'Running on device {device}')

        
        print('Ready to train\n')


    

    def train_loop(self):
        '''
        Trains the model for one epoch (a single pass through the data)
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

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if (batch % 5 == 0 or batch == n_batches-1) and batch > 0:
                loss, current = loss.item(), (batch+1) * batch_size
                elapsed = timer.get(reset=True)
                print(f'Batch {batch+1:>3d}/{n_batches}, loss: {loss:>7f}  [{current:>5d}/{size:>5d}] ({elapsed:.3f}s)', end=' ')

                if self.dataset.val_frac:
                    val_loss = self.get_error('val')
                    print(f'val loss: {val_loss:>7f}', end='')
                print('')

    def get_error(self, subset):
        '''
        Returns the MSE of the model on either the training or validation set
        '''

        self.model.eval()
        if subset == 'val':
            self.dataset.val_mode()
        else:
            self.dataset.train_mode()

        n_batches = len(self.data_loader)
        total_loss = 0.

        with torch.no_grad():
            for X, y in self.data_loader:
                X = X.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.float)
                pred = self.model(X)
                total_loss += self.loss_fn(pred, y).item()
        self.model.train()
        self.dataset.train_mode()
        return total_loss / n_batches

        
    
    def run_training(self, epochs):
        main_timer, loop_timer = Timer(), Timer()
        print(f'Beginning training from epoch {self.cur_epoch+1} for {epochs} epochs')

        for _ in range(epochs):
            self.cur_epoch += 1
            print(f'Epoch {self.cur_epoch} \n-------------------------------')
            self.train_loop()
            if self.save_path:
                self.save_model(self.cur_epoch)
                print(f'Saved to {self.save_path}')
            print(f'Took {loop_timer.get(reset=True):.3f}s total\n-------------------------------\n')

        print(f'Took {main_timer.get():.4f} seconds')
        print('Done!')
        
    
    def save_model(self, epoch, all_params=True):
        if all_params:
            torch.save({
                'epoch': epoch,
                'train_ind': self.dataset.train_data.index,
                'val_ind': self.dataset.val_data.index if self.dataset.val_data else [],
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.save_path)
        else:
            torch.save(self.model.state_dict(), self.save_path)

    def load_model(self, model_path):
        '''
        Load a previously trained model to continue training
        '''
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.cur_epoch = checkpoint['epoch']
        self.val_ind = checkpoint['val_ind']
        self.train_ind = checkpoint['train_ind']
        