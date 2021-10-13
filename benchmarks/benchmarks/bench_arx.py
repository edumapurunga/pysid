"""
    This benchmark is intended to test the time for which the arx method from the prediction error method 
    peforms the tasks.
"""
# Imports
import numpy as np
import scipy.signal as sig
try:
    from pysid import arx
except ImportError:
    pass
    

# Define the class to be tested

class Arx:
    params = [
    [1, 2, 3, 4, 5, 6],
    [0, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 6],
    [100, 1000, 10000]
    ]

    param_names = ['na', 'nb', 'nk', 'N']


    # Setting for the benchmark
    def setup(self, na, nb, nk, N):
        
        # Complex number
        j = np.complex(0, 1)
        # Sample System
        A = np.poly([0.88, 0.99, 0.6 + 0.5*j, 0.6 - 0.5*j, 0.77, 0.87])
        B = np.poly([0.4, 0.66, 0.65, 0.44, 0.52])
        # Generate the input 
        self.u = np.random.randn(N, 1)
        # Generate the output
        self.y = sig.lfilter(B, A, u, axis=0) + sig.lfilter([1], A, 0.1*np.random.randn(N, 1), axis=0)
        
        
    def time_arx_siso(self, na, nb, nk, N):
        A, B = arx(na, nb, nk, self.u, self.y)
    #def time_arx_simo(self):

    #def time_arx_miso(self):

    #def time_arx_mimo(self):
