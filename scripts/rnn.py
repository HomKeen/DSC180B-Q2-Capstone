# %%
import torch
import sys
import os
import numpy as np
# import matplotlib as mpl
# mpl.use('TkAgg')

from torch import nn, optim
from torch.nn import Sequential, LSTM, GRU, Linear, ReLU, ELU, Sequential
from torch.utils.data import  DataLoader
from matplotlib import pyplot as plt

from dataset import EarthSystemsDataset 
from nn_util import EarthSystemsRNN, GRULayerNorm, Trainer
from util import Timer

# %% [markdown]
# ## Reccurent Neural Network (RNN)

# %%
def rnn_layers1(in_size, label_size, lags):
    # this is a lastonly sequence
    # in_size is number of variables
    h_size1 = 25
    h_size2 = 50
    h_size3 = 80

    rnn_layers = [
        GRU(in_size, h_size1, batch_first=True, num_layers=1),
        GRU(h_size1, h_size2, batch_first=True, num_layers=1),
        GRU(h_size2, h_size3, batch_first=True, num_layers=1)
        # GRULayerNorm(in_size, h_size1),
        # GRULayerNorm(h_size1, h_size2),
        # GRULayerNorm(h_size2, h_size3)
    ]

    fc_layers = [
        ReLU(),
        Linear(h_size3, label_size)
    ]

    return Sequential(*rnn_layers), Sequential(*fc_layers)

def rnn_layers2(in_size, label_size, lags):
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

def rnn_layers3(in_size, label_size, lags):
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

def rnn_layers4(in_size, label_size, lags):
    # in_size is number of variables
    # this is a lastonly sequence
    h_size1 = 500

    rnn_layers = [
        # GRU(in_size, h_size1, batch_first=True, num_layers=1)
        GRULayerNorm(in_size, h_size1)
    ]


    fc_layers = [
        ReLU(),
        Linear(h_size1, label_size)
    ]

    return Sequential(*rnn_layers), Sequential(*fc_layers)

def rnn_layers5(in_size, label_size, lags):
    # in_size is number of variables
    # this is a lastonly sequence
    h_size1 = 25
    h_size2 = 50
    h_size3 = 80
    rnn_layers = [
        # GRU(in_size, h_size1, batch_first=True, num_layers=1)
        GRULayerNorm(in_size, h_size1),
        GRULayerNorm(h_size1, h_size2),
        GRULayerNorm(h_size2, h_size3)
    ]


    fc_layers = [
        ReLU(),
        Linear(h_size3, label_size)
    ]

    return Sequential(*rnn_layers), Sequential(*fc_layers)


def rnn_layers6(in_size, label_size, lags):
    # in_size is number of variables
    # this is a lastonly sequence
    h_size1 = 15
    h_size2 = 30
    h_size3 = 60
    rnn_layers = [
        # GRU(in_size, h_size1, batch_first=True, num_layers=1)
        GRULayerNorm(in_size, h_size1),
        GRULayerNorm(h_size1, h_size2),
        GRULayerNorm(h_size2, h_size3)
    ]


    fc_layers = [
        ReLU(),
        Linear(h_size3, label_size)
    ]

    return Sequential(*rnn_layers), Sequential(*fc_layers)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
def find_best_lag(all_lags, k):
    all_var_names = ['global_temp', 'petroleum', 'elec_fossil', 'elec_clean', 'co2', 'ch4']
    # y_vals = ['temp_change', 'petroleum', 'elec_fossil', 'elec_clean', 'co2_average', 'ch4_average']
    y_vals = ['temp_change', 'elec_fossil', 'co2']

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
                model = EarthSystemsRNN(rnn_layers, fc_layers, last_only=True).to(device)
                loss_fn = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-3)
            
                trainer = Trainer(model, loss_fn, optimizer, dataset=data, batch_size=10, 
                                  save_path=None, preload=None, device=device, val_freq='epoch')
            
                trainer.run_training(epochs)
                val_err = trainer.get_error('val')
                errs.append(val_err)

                sys.stdout = sys.__stdout__
                print(f'{val_err:.5f}', end=' | ')
                sys.stdout.flush()


            print(f'\n\t\tAvg error: {np.mean(val_err):.5f} (took {timer.get(True):.5f}s)')
            sys.stdout.flush()
            # print(f'\n\t[{li+1}/{len(all_lags)}] Error for lag={lag}: {val_err} (took {timer.get(True):.4f}s)')


# %%
# lags = list(range(4, 35))
# k = 5
# find_best_lag(lags, 5)
# sys.exit()

# %%
# all_var_names = ['global_temp', 'petroleum', 'elec_fossil', 'elec_clean', 'co2', 'ch4']
# y_vals = ['temp_change', 'petroleum', 'elec_fossil', 'elec_clean', 'co2_average', 'ch4_average']
# # all_lags = [2, 5, 10, 15, 20, 25, 30, 35, 40]
# lag = 14

# for yi, y_val in enumerate(y_vals):
#     # each variable gets its turn as the dependent variable

#     # cur_vars.pop(yi)
#     # cur_vars = [all_var_names[yi]] + cur_vars # arrange variables so the y-variable is first
    
#     print(f'[({yi+1}/{len(y_vals)})] Running predictions for y={y_val}')


#     for i, var_name in enumerate(all_var_names):
#         # drop each variable one-by-one to see the effect on the error (always keep the y-variable, obviously)
#         cur_vars = all_var_names.copy()
#         if i != yi:
#             cur_vars.remove(var_name)
#             dropped = var_name
#         else:
#             dropped = 'none'
#         print(f'\t[{i+1}/{len(all_var_names)}] Dropping {dropped}', end='')


#         # create path for model saving
#         if not os.path.exists(f'models/{y_val}'):
#             os.makedirs(f'models/{y_val}')
#         save_path = f'models/{y_val}/rnn_{y_val}_{dropped}.pth'
#         timer = Timer()

#         # create path for logs output
#         if not os.path.exists(f'logs/{y_val}/'):
#             os.makedirs(f'logs/{y_val}/')
        

#         out_path = f'logs/{y_val}/rnn_{y_val}_{dropped}.out'

#         sys.stdout = open(out_path, 'w')
#         # sys.stderr = sys.stdout
#         # sys.stderr = open(err_path, 'w')

#         print(f'Using variables {cur_vars}. Dropped {dropped}')
#         print(f'Predicting {y_val} with {lag} lags')
#         # print(f'Current best error: {min_err} (lag={best_lag})\n')

#         data = EarthSystemsDataset(cur_vars, y_vals=[y_val], val_frac=0.1, lags=lag, mode='rnn')
#         data.train_mode()

#         rnn_layers, fc_layers = rnn_layers1(len(cur_vars), 1, lag)
#         model = EarthSystemsRNN(rnn_layers, fc_layers, last_only=False).to(device)
#         loss_fn = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=1e-5)
#         epochs = 1500

#         trainer = Trainer(model, loss_fn, optimizer, dataset=data, batch_size=10, 
#                             save_path=None, preload=None, device=device, val_freq='epoch')
        
#         trainer.run_training(epochs)
#         val_err = trainer.get_error('val')
#         # all_errs.append(val_err)

#         # if the error with this lag value is lower than the previous best, save this model
#         # if val_err < min_err:
#         # print(f'\nNew error of {val_err} (lag={lag}) better than current best of {min_err} (lag={best_lag})')
#         print(f'Finished with validation error: {val_err}')
#         print(f'Saving to {save_path}')

#         # min_err = val_err
#         # best_lag = lag

#         trainer.save_path = save_path
#         trainer.save_model()

#         # else:
#             # print(f'\nNew error of {val_err} (lag={lag}) is worse than current best of {min_err} (lag={best_lag})')
#             # print(f'Not saving this model')

#         sys.stdout = sys.__stdout__
#         print(f'\t| Val error: {val_err} | Took {timer.get(reset=True) :.4f}s')
            
#         # sys.stdout = open(f'logs/{y_val}/drop_{dropped}/summary.out', 'w')
#         # print(f'Summary of lag values for y={y_val} and dropping {dropped}')
#         # print('LAG   ERROR')
#         # for lag, err in zip(all_lags, all_errs):
#         #     print(f'{lag} --- {err:.5f}')
        
#         # sys.stdout = sys.__stdout__





# %%
# all_var_names = ['global_temp', 'petroleum', 'elec_fossil', 'elec_clean', 'co2', 'ch4']
# y_vals = ['temp_change', 'petroleum', 'elec_fossil', 'elec_clean', 'co2_average', 'ch4_average']
# all_lags = [2, 5, 10, 15, 20, 25, 30, 35, 40]


# for yi, y_val in enumerate(y_vals):
#     # each variable gets its turn as the dependent variable

#     # cur_vars.pop(yi)
#     # cur_vars = [all_var_names[yi]] + cur_vars # arrange variables so the y-variable is first
    
#     print(f'[({yi+1}/{len(y_vals)})] Running predictions for y={y_val}')


#     for i, var_name in enumerate(all_var_names):
#         # drop each variable one-by-one to see the effect on the error (always keep the y-variable, obviously)
#         cur_vars = all_var_names.copy()
#         if i != yi:
#             cur_vars.remove(var_name)
#             dropped = var_name
#         else:
#             dropped = 'none'
#         print(f'\t[{i+1}/{len(all_var_names)}] Dropping {dropped}')

#         min_err, best_lag = float('inf'), -1
#         all_errs = []

#         # create path for model saving
#         if not os.path.exists(f'models/{y_val}'):
#             os.makedirs(f'models/{y_val}')
#         save_path = f'models/{y_val}/rnn_{y_val}_{dropped}.pth'

#         for li, lag in enumerate(all_lags):
#             timer = Timer()
#             print(f'\t\t[{li+1}/{len(all_lags)}] Using lags={lag}', end=' ')

#             # create path for logs output
#             if not os.path.exists(f'logs/{y_val}/drop_{dropped}'):
#                 os.makedirs(f'logs/{y_val}/drop_{dropped}')
            

#             out_path = f'logs/{y_val}/drop_{dropped}/rnn_{y_val}_{dropped}_{lag}.out'
#             # err_path = f'logs/{y_val}/rnn_{y_val}_{i}_{lag}.err'

#             sys.stdout = open(out_path, 'w')
#             # sys.stderr = sys.stdout
#             # sys.stderr = open(err_path, 'w')

#             print(f'Using variables {cur_vars}. Dropped {dropped}')
#             print(f'Predicting {y_val} with {lag} lags')
#             print(f'Current best error: {min_err} (lag={best_lag})\n')

#             data = EarthSystemsDataset(cur_vars, y_vals=[y_val], val_frac=0.1, lags=lag, mode='rnn')
#             data.train_mode()

#             rnn_layers, fc_layers = rnn_layers1(len(cur_vars), 1, lag)
#             model = EarthSystemsRNN(rnn_layers, fc_layers, last_only=False).to(device)
#             loss_fn = nn.MSELoss()
#             optimizer = optim.Adam(model.parameters(), lr=1e-5)
#             epochs = 1000

#             trainer = Trainer(model, loss_fn, optimizer, dataset=data, batch_size=10, 
#                                 save_path=None, preload=None, device=device, val_freq='epoch')
            
#             trainer.run_training(epochs)
#             val_err = trainer.get_error('val')
#             all_errs.append(val_err)

#             # if the error with this lag value is lower than the previous best, save this model
#             if val_err < min_err:
#                 print(f'\nNew error of {val_err} (lag={lag}) better than current best of {min_err} (lag={best_lag})')
#                 print(f'Saving to {save_path}')

#                 min_err = val_err
#                 best_lag = lag

#                 trainer.save_path = save_path
#                 trainer.save_model()

#             else:
#                 print(f'\nNew error of {val_err} (lag={lag}) is worse than current best of {min_err} (lag={best_lag})')
#                 print(f'Not saving this model')

#             sys.stdout = sys.__stdout__
#             print(f'(took {timer.get(reset=True) :.4f}s)')
            
#         sys.stdout = open(f'logs/{y_val}/drop_{dropped}/summary.out', 'w')
#         print(f'Summary of lag values for y={y_val} and dropping {dropped}')
#         print('LAG   ERROR')
#         for lag, err in zip(all_lags, all_errs):
#             print(f'{lag} --- {err:.5f}')
        
#         sys.stdout = sys.__stdout__





# %%
from torch.utils.data import DataLoader

# %%
data_var_names = ['global_temp', 'petroleum', 'elec_fossil', 'elec_clean', 'co2', 'ch4']
y_vals = ['temp_change']
lags = 14
preload = 'models/rnn_temp_seq0.pth' #'models/rnn_temp_h3_lastonly.pth'
save_path = None #'models/rnn_temp_only.pth'

data = EarthSystemsDataset(data_var_names, y_vals=y_vals, val_frac=0.15, lags=lags, mode='rnn', normalize=True)
data.train_mode()
rnn_layers, fc_layers = rnn_layers6(len(data_var_names), len(y_vals), lags)

model = EarthSystemsRNN(rnn_layers, fc_layers, last_only=True).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.NAdam(model.parameters(), lr=1e-5)

print(f'Using random seed {torch.seed()}')
trainer = Trainer(model, loss_fn, optimizer, dataset=data, batch_size=5, save_path=save_path, 
                preload=preload, device=device, save_freq=25, val_freq='epoch')

# trainer.run_training(350)

# %%
trainer.get_error('train'), trainer.get_error('val')

# %%
data.train_mode()
with torch.no_grad():
    pred_loader = DataLoader(data, batch_size=1, shuffle=False)
    pred = [data.data.reset_index()[y_vals[0]][0]]*lags + [model(X.to(device, dtype=torch.float)) for X, y in pred_loader]
    _, val_pred = trainer.get_val_error()
    pred = pred + val_pred

# %%
plt.figure(figsize=(15,8))
plt.plot(pred)
plt.plot(data.full_data.reset_index()[y_vals[0]], alpha=0.5)
# print(len(pred))
# print(data.data.reset_index()[y_vals[0]].shape)
plt.axvline(data.train_data.shape[0], color='red')

# %%
plt.figure(figsize=(15,8))

plt.plot(trainer.val_errors, label='val')
plt.plot(trainer.train_errors, label='train')
plt.legend()

# %%



