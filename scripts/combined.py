'''
Script for combining the time-lagged (Granger causality) recurrent neural network with SCMs. This is
done by fitting an SCM to the residuals of the time-lagged predictions.
'''

import torch
import pandas as pd

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCMBased.ANM.ANM import ANM

from dataset import EarthSystemsDataset
from nn_util import GrangerRNN
from scripts.time_lagged import rnn_layers4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_residuals(model_path, num_models, data, lags):
    '''
    Get the residuals (y_hat - y) for a given model

    :model_path: (str) Path to saved model.
    :num_models: (int) Number of variables (i.e., number of GrangerComponents trained in the model).
    :data: (EarthSystemsDataset) an initialized dataset to get residuals for.
    :lags: (int) Number of time lags used during training.

    return: torch.Tensor of shape (num_points, num_features) containing the residual for each time step
            for each feature.
    '''
    model = GrangerRNN(rnn_layers4, num_models, len(data.data_var_names), 
                       lags=lags, reg_lags=False, last_only=True).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        pred_loader = DataLoader(data, batch_size=32, shuffle=False)
        all_pred = torch.cat([model(X.to(DEVICE, dtype=torch.float))-y for X, y in pred_loader], dim=0)
        all_pred = torch.cat([torch.zeros(lags, num_models), all_pred], dim=0)

        
    return all_pred

def get_all_residuals(model_paths, lags):
    '''
    Get ALL residuals for a set of models. Will be returned as a Tensor of shape (num_models, num_datapoints, num_features)

    :model_paths: (list-like of str) Paths to all the models to get residuals for.
    :lags: (int) Number of time lags used during training.

    return: torch.Tensor of shape (num_models, num_points, num_features) containing the residuals for each model
            for each time step for each feature.
            
    '''
    data_var_names = ['global_temp', 'petroleum', 'elec_fossil', 'elec_clean', 'co2', 'ch4']
    y_vals = ['temp_change', 'petroleum', 'elec_fossil', 'elec_clean', 'co2_average', 'ch4_average']

    data = EarthSystemsDataset(data_var_names, y_vals=y_vals, add_index=True, val_frac=0.03, lags=lags, mode='rnn', normalize=True)
    data.train_mode()

    all_res = []
    for model_path in model_paths:
        all_res.append(get_residuals(model_path, len(y_vals), data, lags))
    
    return torch.stack(all_res)


if __name__ == '__main__':
    data_var_names = ['global_temp', 'petroleum', 'elec_fossil', 'elec_clean', 'co2', 'ch4']
    y_vals = ['temp_change', 'petroleum', 'elec_fossil', 'elec_clean', 'co2_average', 'ch4_average']
    lags = 30
    model_paths = [f'models/rnn_granger{i}.pth' for i in range(30)]
    all_res = get_all_residuals(model_paths, lags)
    # Calculate the mean residuals across all models to get a more stable outcome
    mean_res = all_res.mean(dim=0)
    # Using PC algorithm, find the SCM
    cg2 = pc(mean_res.numpy())
    

    # Plotting the mean residuals for each variable
    data = EarthSystemsDataset(data_var_names, y_vals=y_vals, add_index=True, val_frac=0.03, lags=lags, mode='rnn', normalize=True)
    data.train_mode()

    fig, axes = plt.subplots(6, 1, figsize=(15,18))
    fig.suptitle('Residuals')
    fig.tight_layout()
    for i in range(len(axes)):
        axes[i].set_title(data.data.columns[i])
        axes[i].plot(mean_res[:,i].tolist(), label='mean residuals')
        axes[i].plot(data.full_data.reset_index()[y_vals[i]], alpha=0.5, label='variable')
        axes[i].axvline(data.train_data.shape[0], color='red')

        axes[i].legend()

    
    # Visualize the instantaneous relations
    plt.figure('Causal Graph of Residuals')
    cg2.draw_pydot_graph(labels=data_var_names)

    # Use ANMs to determine causal directions
    temp_data = pd.DataFrame(mean_res, columns = data_var_names)
    anm = ANM()

    X = temp_data['elec_clean']
    y = temp_data['elec_fossil']
    p_value_forward, p_value_backward = anm.cause_or_effect(X.array.reshape(-1,1),y.array.reshape(-1,1))
    print('ANM p-values for elec_clean -- elec_fossil:')
    print(p_value_forward, p_value_backward)

    X = temp_data['petroleum']
    y = temp_data['ch4']
    p_value_forward, p_value_backward = anm.cause_or_effect(X.array.reshape(-1,1),y.array.reshape(-1,1))
    print('ANM p-values for petroleum -- ch4:')
    print(p_value_forward, p_value_backward)