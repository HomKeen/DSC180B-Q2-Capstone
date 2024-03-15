'''
Script for detecting time-lagged causality (Granger causality) among the variables
'''
import torch
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns

from torch import nn, optim
from torch.nn import Sequential, GRU, Linear, ReLU, Sequential
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from dataset import EarthSystemsDataset
from nn_util import GrangerRNN, EarthSystemsRNN, GRULayerNorm, Trainer
from util import Timer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Defined below are several architectures of RNNs that we experimented with. The most successful was `rnn_layers4`, 
which was also the simplest. This is because all the other layers ended up overfitting the data which only 
had ~480 data points.
'''
def rnn_layers1(in_size, label_size):
    # this is a lastonly sequence
    # in_size is number of variables
    h_size1 = 25
    h_size2 = 50
    h_size3 = 80

    rnn_layers = [
        GRU(in_size, h_size1, batch_first=True, num_layers=1),
        GRU(h_size1, h_size2, batch_first=True, num_layers=1),
        GRU(h_size2, h_size3, batch_first=True, num_layers=1)
    ]

    fc_layers = [
        ReLU(),
        Linear(h_size3, label_size)
    ]

    return Sequential(*rnn_layers), Sequential(*fc_layers)

def rnn_layers2(in_size, label_size):
    # this is a lastonly sequence
    # in_size is number of variables
    h_size1 = 50
    h_size2 = 100

    rnn_layers = [
        GRULayerNorm(in_size, h_size1),
        GRULayerNorm(h_size1, h_size2)
    ]


    fc_layers = [
        ReLU(),
        Linear(h_size2, label_size)
    ]

    return Sequential(*rnn_layers), Sequential(*fc_layers)

def rnn_layers3(in_size, label_size):
    # this is a lastonly sequence
    # in_size is number of variables
    h_size1 = 100
    h_size2 = 250

    rnn_layers = [
        GRULayerNorm(in_size, h_size1),
        GRULayerNorm(h_size1, h_size2)
    ]

    fc_layers = [
        ReLU(),
        Linear(h_size2, label_size),
    ]

    return Sequential(*rnn_layers), Sequential(*fc_layers)

def rnn_layers4(in_size, label_size):
    # in_size is number of variables
    # this is a lastonly sequence
    h_size1 = 6
    h_size2 = 12

    rnn_layers = [
        GRULayerNorm(in_size, h_size1),
        GRULayerNorm(h_size1, h_size2)
    ]


    fc_layers = [
        ReLU(),
        Linear(h_size2, label_size)
    ]

    return Sequential(*rnn_layers), Sequential(*fc_layers)

def rnn_layers5(in_size, label_size):
    # in_size is number of variables
    # this is a lastonly sequence
    h_size1 = 25
    h_size2 = 50
    h_size3 = 80
    rnn_layers = [
        GRULayerNorm(in_size, h_size1),
        GRULayerNorm(h_size1, h_size2),
        GRULayerNorm(h_size2, h_size3)
    ]


    fc_layers = [
        ReLU(),
        Linear(h_size3, label_size)
    ]

    return Sequential(*rnn_layers), Sequential(*fc_layers)


def rnn_layers6(in_size, label_size):
    # in_size is number of variables
    # this is a lastonly sequence
    h_size1 = 15
    h_size2 = 30
    h_size3 = 60
    rnn_layers = [
        GRULayerNorm(in_size, h_size1),
        GRULayerNorm(h_size1, h_size2),
        GRULayerNorm(h_size2, h_size3)
    ]


    fc_layers = [
        ReLU(),
        Linear(h_size3, label_size)
    ]

    return Sequential(*rnn_layers), Sequential(*fc_layers)





def find_best_lag(all_lags, k):
    '''
    Performs grid search for a given set of lag values to try, and records the error for each one.
    A lag value that results in the lowest error is the best one to use.

    :all_lags: (list-like of int) List of lag values to try.
    :k: (int) Number of runs to perform per lag, to get a more stable outcome.

    return: None
    '''
    all_var_names = ['global_temp', 'petroleum', 'elec_fossil', 'elec_clean', 'co2', 'ch4']
    y_vals = ['temp_change', 'petroleum', 'elec_fossil', 'elec_clean', 'co2_average', 'ch4_average']

    for yi, y_val in enumerate(y_vals):
        print(f'[{yi+1}/{len(y_vals)}] Finding best lag for y={y_val}')
        for li, lag in enumerate(all_lags):
            timer = Timer()
            print(f'\t[{li+1}/{len(all_lags)}] {k} runs for lag={lag}: ')
            print('\t\t', end='')

            if not os.path.exists(f'logs/lags/{y_val}'):
                os.makedirs(f'logs/lags/{y_val}')

            out = open(f'logs/lags/{y_val}/{y_val}_lag{lag}.out', 'w')
            sys.stdout = out
            
            data = EarthSystemsDataset(all_var_names, y_vals=[y_val], val_frac=0.1, lags=lag, mode='rnn', normalize=True)
            data.train_mode()
            epochs = 250
            errs = []

            # run each model k times to get an average
            for i in range(k):
                sys.stdout = out
                rnn_layers, fc_layers = rnn_layers2(len(all_var_names), 1, lag)
                model = EarthSystemsRNN(rnn_layers, fc_layers, last_only=True).to(DEVICE)
                loss_fn = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)
            
                trainer = Trainer(model, loss_fn, optimizer, dataset=data, batch_size=10, 
                                  save_path=None, preload=None, device=DEVICE, val_freq='epoch')
            
                trainer.run_training(epochs)
                val_err = trainer.get_error('val')
                errs.append(val_err)

                sys.stdout = sys.__stdout__
                print(f'{val_err:.5f}', end=' | ')
                sys.stdout.flush()


            print(f'\n\t\tAvg error: {np.mean(val_err):.5f} (took {timer.get(True):.5f}s)')
            sys.stdout.flush()


def get_importances(model_path, data):
    '''
    Retrieve the 'importances' that each variable has for predicting each other variable,
    as determined by the given model during training.

    :model_path: (str) Path to the trained model

    return: pd.DataFrame of size (num_variables, num_variables). The entry in row i and column j
            tells us the 'importance' that series i has for predicting future values of series j. 
    '''
    model = GrangerRNN(rnn_layers4, len(data.y_vals), len(data.data_var_names), 
                       lags=data.lags, reg_lags=False, last_only=True).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    df_imp = pd.DataFrame(columns=data.data.columns, index=data.data.columns)
    for i in range(len(model.models)):
        df_imp[data.data.columns[i]] = model.models[i].feature_filter.filter.tolist()

    return df_imp

def get_all_importances(model_paths):
    '''
    Retrieve all the 'importances' that each variable has for predicting each other variable,
    as determined by the models during training. Averages over each model provided.

    :model_paths: (list-like of str) Paths to all models.

    return: pd.DataFrame with index of size (num variables * num models), and one column for 
            each variable. 
    '''
    data_var_names = ['global_temp', 'petroleum', 'elec_fossil', 'elec_clean', 'co2', 'ch4']
    y_vals = ['temp_change', 'petroleum', 'elec_fossil', 'elec_clean', 'co2_average', 'ch4_average']
    lags = 30

    data = EarthSystemsDataset(data_var_names, y_vals=y_vals, add_index=True, val_frac=0.03, lags=lags, mode='rnn', normalize=True)
    data.train_mode()

    df_all_imp = pd.DataFrame(columns=['model_num', 'var']+data.data.columns.tolist())

    for model_num, model_path in enumerate(model_paths):
        df_imp = get_importances(model_path, data)
        df_imp['model_num'] = model_num
        df_all_imp = pd.concat([df_all_imp, df_imp.reset_index().rename(columns={'level_0': 'var'})])

    return df_all_imp

def plot_importance_hist(all_imp, indep_var, dep_var):
    '''
    Plots the histogram of importance of `indep_var` for forecasting `dep_var`, over all the given models.

    :all_imp: pd.DataFrame. Output of get_all_importances.
    :indep_var: (str) Name of the independent variable.
    :dep_var: (str) Name of the dependent variable.

    return: None
    '''
    vals = all_imp[all_imp['var'] == indep_var][dep_var]

    plt.hist(vals)
    plt.show()


'''
Uncomment the following block to run code that tests for the best lag value
'''

# if __name__ == '__main__':
#     lags = list(range(4, 35))
#     k = 5
#     find_best_lag(lags, 5)

if __name__ == '__main__':
    data_var_names = ['global_temp', 'petroleum', 'elec_fossil', 'elec_clean', 'co2', 'ch4']
    y_vals = ['temp_change', 'petroleum', 'elec_fossil', 'elec_clean', 'co2_average', 'ch4_average']
    lags = 30
    lam = 1e-3
    lr = 2e-4
    batch_size = 16
    save_freq = 25 # Save the model every 25 epochs
    preload = 'models/rnn_granger.pth' 
    save_path = 'models/rnn_granger.pth'

    data = EarthSystemsDataset(data_var_names, y_vals=y_vals, add_index=False, val_frac=0.03, lags=lags, mode='rnn', normalize=True)
    data.train_mode()

    model = GrangerRNN(rnn_layers4, len(y_vals), len(data.data.columns), lags=lags, reg_lags=False, last_only=True).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.NAdam(model.parameters(), lr=lr)

    print(f'Using random seed {torch.seed()}')
    print(f'LAMBDA={lam}')
    trainer = Trainer(model, loss_fn, optimizer, dataset=data, batch_size=batch_size, save_path=save_path, 
                    preload=preload, device=DEVICE, save_freq=save_freq, val_freq='epoch')

    # UNCOMMENT TO RUN TRAINING
    # trainer.run_training(250, lam=lam)


    # A training error <0.009 is considered overfit. Overfitting is not ideal, but acceptable for our objective
    # A validation error <0.02 is preferred, but again not essential
    print(f'Training error: {trainer.get_error("train")}; validation error: {trainer.get_error("val")}')
          

    # Get the training and validation errors 
    data.train_mode()
    with torch.no_grad():
        pred_loader = DataLoader(data, batch_size=1, shuffle=False)
        pred = [data.data.reset_index()[y_vals].iloc[0]]*lags \
                + [model(X.to(DEVICE, dtype=torch.float)) for X, y in pred_loader]
        _, val_pred = trainer.get_val_error()
        pred = pred + val_pred
    pred = torch.tensor(pred)


    # Plot predictions vs. true values for all variables
    fig, axes = plt.subplots(6, 1, figsize=(15,18))
    for i in range(len(axes)):
        axes[i].set_title(data.data.columns[i])
        axes[i].plot(pred[:,i], label='pred')
        axes[i].plot(data.full_data.reset_index()[y_vals[i]], alpha=0.5, label='true')
        axes[i].axvline(data.train_data.shape[0], color='red')

        axes[i].legend()
    fig.tight_layout()
    

    # Plot validation error during training
    plt.figure('errors', figsize=(18,8))

    plt.plot(trainer.val_errors[50:], label='val')
    plt.plot(trainer.train_errors, label='train')
    plt.legend()

    # View feature filters, averaged over all 30 models
    model_paths = [f'models/rnn_granger{i}.pth' for i in range(30)]
    all_imp = get_all_importances(model_paths)

    df_gc = all_imp.groupby('var').mean().drop(index='index').reindex(['temp_change', 'petroleum', 'elec_fossil', 'elec_clean', 'co2_average', 'ch4_average'])
    print(df_gc)

    
    # Plot heatmaps
    plt.figure('GC')
    sns.heatmap(df_gc)
    # If we threshold by saying that >0.01 implies Granger causality, we get the following
    plt.figure('GC_thresh')
    sns.heatmap(df_gc > 0.01)

    plt.show()

