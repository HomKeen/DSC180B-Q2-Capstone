# %%
import torch

from torch import nn, optim
from torch.nn import Sequential, RNN, LSTM, Linear, ReLU, Sequential
from torch.utils.data import  DataLoader
from matplotlib import pyplot as plt

from dataset import EarthSystemsDataset 
from nn_util import EarthSystemsRNN, Trainer

# %% [markdown]
# ## Reccurent Neural Network (RNN)

# %%
def rnn_layers1(in_size, label_size, lags):
    # in_size is number of variables
    h_size1 = 100
    h_size2 = 250
    h_size3 = 400

    rnn_layers = [
        LSTM(in_size, h_size1, batch_first=True, num_layers=1),
        LSTM(h_size1, h_size2, batch_first=True, num_layers=1),
        # LSTM(h_size2, h_size3, batch_first=True, num_layers=1)

        # Flatten(),
    ]

    fc_layers = [
        ReLU(),
        Linear(lags*h_size2, (lags*h_size2)//4),
        ReLU(),
        Linear((lags*h_size2)//4, (lags*h_size2)//8),
        ReLU(),
        Linear((lags*h_size2)//8, label_size)

    ]

    return Sequential(*rnn_layers), Sequential(*fc_layers)

def rnn_layers2(in_size, label_size, lags):
    # in_size is number of variables
    h_size1 = 100
    h_size2 = 250
    h_size3 = 400
    rnn_layers = [
        LSTM(in_size, h_size1, batch_first=True, num_layers=1),
        LSTM(h_size1, h_size2, batch_first=True, num_layers=1),
        LSTM(h_size2, h_size3, batch_first=True, num_layers=1)
        # RNN(in_size*2, in_size*3, batch_first=True),
        # Flatten(),
    ]

    fc_layers = [
        Linear(h_size3, label_size)
    ]

    return Sequential(*rnn_layers), Sequential(*fc_layers)

# r = rnn_layers1(4, 1)
# a = torch.randn(40, 15, 4)
# 
# r(a)[0].shape

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


torch.set_default_dtype(torch.float64)

# %%
data_var_names = ['global_temp', 'elec_fossil', 'elec_clean', 'co2', 'ch4']
y_vals = ['co2_average']
lags = 25
save_path = None #'rnn_1.pth'

data = EarthSystemsDataset(data_var_names, y_vals=y_vals, val_frac=None, lags=lags, mode='rnn')
data.train_mode()
rnn_layers, fc_layers = rnn_layers1(len(data_var_names), len(y_vals), lags)

model = EarthSystemsRNN(rnn_layers, fc_layers, last_only=False).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

trainer = Trainer(model, loss_fn, optimizer, dataset=data, batch_size=10, save_path=save_path, device=device)

# %%
trainer.run_training(200)

# %%


# %%
trainer.get_error('train')

# %%
with torch.no_grad():
    pred_loader = DataLoader(data, batch_size=1, shuffle=False)
    pred = [data.data.reset_index()[y_vals[0]][0]]*lags + [model(X) for X, y in pred_loader]

# %%
data.train_mode()
plt.figure(figsize=(15,8))
plt.plot(pred)
plt.plot(data.data.reset_index()[y_vals[0]])
print(len(pred))
print(data.data.reset_index()[y_vals[0]].shape)

# %%
from data import grab_dataset

# %%
a = grab_dataset('elec_clean')
a.plot(kind='line', y='elec_clean')

# %%
b = grab_dataset('elec_fossil')
b.plot(kind='line', y='elec_fossil')

# %%



