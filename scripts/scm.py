'''
Script for running the SCM using CD-NOD for instantaneous causal relations. CD-NOD is an extension
of the PC algorithm and is able to handle nonstationary data.

This algorithm is NOT indicative of our true results; it only looks at causal relations shared
between variables during the same time step.
'''
import numpy as np

from causallearn.search.ConstraintBased.CDNOD import cdnod
from dataset import EarthSystemsDataset


if __name__ == '__main__':
    data_var_names = ['global_temp', 'elec_fossil', 'elec_clean', 'co2', 'ch4', 'petroleum']
    y_vals = ['temp_change']
    lags = 1

    earth_data = EarthSystemsDataset(data_var_names, y_vals=y_vals, val_frac=0.1, lags=lags, mode='ann')
    earth_data.full_mode()
    earth_data.data


    d = earth_data.data.to_numpy()
    cg = cdnod(d, np.arange(earth_data.data.shape[0]).reshape(-1,1) / 10.0, indep_test='kci')
    cg.draw_pydot_graph(labels=data_var_names+['index'])