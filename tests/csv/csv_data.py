"""
    author: @lima84

    Data handling functions using numpy.
"""


from numpy import loadtxt, savetxt, concatenate
from numpy.random import rand, randn
from scipy.signal import lfilter

def gen_data(Ao, Bo, N):
    """
    author: @lima84
    Generates a set of input and output data following:
        y(t) = Go(q)*u(t) + Ho(q)*e(t),
    where G(q) = Bo(q)/Ao(q) and H(q) = 1/Ao(q).

    Parameters
    ----------
    Ao : array_like
    Ao(q) polynomial coefficients.
    Bo : array_like
    Bo(q) polynomial coefficients.
    N : int
    Number of samples for the dataset.
    Returns
    -------
    data : ndarray
    Dataset array in the form of [input, output].
    """
    # Replicates the following experiment:
    # y(t) = Go(q)*u(t) + Ho(q)*e(t),
    # where u(t) is the system input and e(t) white noise
    u = -1 + 2*rand(N, 1)   # Defines input signal
    e = 0.01*randn(N, 1)    # Emulates gaussian white noise with std = 0.01

    # Calculates the y ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q)
    y = lfilter(Bo, Ao, u, axis=0) + lfilter([1], Ao, e, axis=0)

    # Concatenates input and output signals
    data = concatenate([u,y], axis=1)
    return data

def load_data(filename, delim=","):
    """
    author: @lima84
    Loads a dataset into a variable. Default format is .csv
    
    Parameters
    ----------
    filename : string
    Name of the file (with extension) from which the dataset is loaded.
    delim : string, optional
        Column delimiter. Default is "," for .csv files.
    Returns
    -------
    data : ndarray
    Loaded dataset.
    """
    try:
        data = loadtxt(filename, dtype=float, delimiter=delim)
        print("csv_data.py::load_data -- Successfully loaded " + filename + ".")
        return data
    except:
        print("csv_data.py::load_data -- Error loading data.")

def save_data(data, filename="data.csv", delim=","):
    """
    author: @lima84
    Saves a dataset into a file. Default format is .csv

    Parameters
    ----------
    data : ndarray
        Dataset to be saved in the file.
    filename : string, optional
    Name of the file (with extension) where the dataset shall be saved. Default is "data.csv"
    delim : string, optional
        Column delimiter. Default is "," for .csv files.
    Returns
    -------

    """
    try:
        savetxt(filename, data, delimiter=delim)
        print("csv_data.py::save_data -- Successfully saved data as " + filename + ".")
    except:
        print("csv_data.py::save_data -- Error saving data.")

#%%
# CSV files example

# Define system and the number of samples
Ao = [1, -1.2, 0.36]
Bo = [0, 0.5, 0.1]
N = 50

# Generates test data
data = gen_data(Ao, Bo, N)

# Saves generated data into 
save_file = "save_test.csv"
save_data(data, save_file)

# Loads previoulsy saved data
load_file = "load_test.csv"
loaded_data = load_data(load_file)
