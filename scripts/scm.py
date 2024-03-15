'''
Script for running the SCM using CD-NOD for instantaneous causal relations. CD-NOD is an extension
of the PC algorithm and is able to handle nonstationary data.

This algorithm is NOT indicative of our true results; it only considers causal relations shared
between variables during the same time step.
'''
import numpy as np

from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.search.FCMBased.ANM.ANM import ANM

from dataset import EarthSystemsDataset
from multi_anm import M_ANM


if __name__ == '__main__':
    data_var_names = ['global_temp', 'elec_fossil', 'elec_clean', 'co2', 'ch4', 'petroleum']
    y_vals = ['temp_change']
    lags = 1

    earth_data = EarthSystemsDataset(data_var_names, y_vals=y_vals, val_frac=0.1, lags=lags, mode='ann')
    earth_data.full_mode()
    earth_data.data

    # Compute the SCM using CD-NOD and draw the graph 
    d = earth_data.data.to_numpy()
    cg = cdnod(d, np.arange(earth_data.data.shape[0]).reshape(-1,1) / 10.0, indep_test='kci')
    cg.draw_pydot_graph(labels=data_var_names+['index'])

    # Determine causal directions with additive noise models (ANM)
    anm = ANM()
    manm = M_ANM()

    # Univariate ANM
    X = earth_data.data['ch4_average']
    y = earth_data.data['petroleum']
    conf = range(len(earth_data.data))

    p_value_forward, p_value_backward = anm.cause_or_effect(X.array.reshape(-1, 1),y.array.reshape(-1, 1))
    print('Univariate ANM p-values:')
    print(p_value_forward, p_value_backward)

    # Multivariate ANM
    time = range(len(earth_data.data))
    temp_data =earth_data.data
    temp_data['Time'] = time
    X = earth_data.data[['ch4_average']]
    y = earth_data.data[['petroleum']]
    conf = earth_data.data[['Time']]

    p_value_forward, p_value_backward = manm.cause_or_effect(X,y, conf)
    print('Multivariate ANM p-values:')
    print(p_value_forward, p_value_backward)