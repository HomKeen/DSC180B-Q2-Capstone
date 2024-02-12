# DSC180B-Q2-Capstone
By Keenan Hom, Dean Carrion, Nithilan Muruganandham, and Arnav Khanna

# Package Requirements

All Python package requirements are in `general_requirements.txt` and `requirements.txt`. `general_requirements.txt` contain all package requirements, *without* specific version requirements. `requirements.txt` contain all packages *with* specific version requirements. 

The code is tested in Python 3.9.13, but should work for any relatively recent version of Python. We recommend creating an Anaconda environment in Python 3.9.13 using package versions in `requirements.txt` for the most reliable reproducibility.

Please note that the `causal-learn` package can only be installed with `pip`.

## File Overview

The project is still in development, which is why much of our code is still in notebooks. All Python scripts and notebooks can be found in the `scripts` folder. In our final submission, nearly every file will be a `.py` file. An overview of each file is as follows:

- Tests for stationarity in variables are in `stationarity.ipynb`. This will likely remain in a notebook in our final submission, since the visualization of data/tables and short markdown comments are important for understanding the key takeaways. 

- Functionality for cleaning and retrieving data is in `data.py`. 

- Development for creating and training artificial neural networks (ANN) are in `ann.ipynb`. In this file, we only test standard feedforward NNs for the sake of creating a baseline for RNNs, which will be our final NN architecture.

- Development for creating and training recurrent neural networks (RNN) are in `rnn.ipynb`.

- Experiments for creating the Structural Causal Models (SCMS) are in `scm_test_1.ipynb` and `scm_test_2.ipynb`.

- Experiments in performing EDA are in `eda.ipynb` and `eda-2.ipynb`, although these are outdated. They are in the `old_notebooks` folder for now, since there is not much EDA to perform on this type of data. Any EDA that we want to do will be little enough to fit in `stationarity.ipynb`. 


## Next Steps

### General

- Add new data in, for both types of models. We have several new datasets ready to go (global crop yield, livestock, forestation, gasoline consumption, ice sheet mass, mean humidity). We started with only 4 variables to keep complexity low as we figure out our methodology, and it is extremely easy to add new datasets once our methods our finished.


### Structural Causal Models

- Fix a few issues, such as ANM and PC models disagreeing on whether there is a link between two variables. This could be due to a few reasons
    - The non-stationarity of the data causes the models to fail. In this case, simply differencing our data would fix it.
    - Choices of kernels must be customized between each pair of variables. In this case, we would have to manually pick kernels for each pair instead of using the default *Gaussian*, which would be tedious.
    - Too many exogenous variables, violating our causal assumptions. In this case, adding more data (which we already plan to do) should suffice.

- Once issues are fixed, and we determine a complete causal graph, we can start determining the actual functions that representing edges in the graph. This will allow us to predict into the future.

### Neural Networks

While the hyperparemeters are not optimized, it looks like our model has been able to grasp the general behavior of global temperature change as it slowly climbs upward over time. Our immediate next steps are:

- Optimize hyperparameters. Grid search works well here, since both our dataset and network are small.
- Begin removing variables one at a time and re-training, in order to determine Granger causality. Since the dataset and network architecture is small, we can do this many times and perform a t-test for higher reliability.
- Once the SCMs determine a causal graph, we may use smaller NNs to serve as functions that represent edges in the graph. Then we can predict future values. 
