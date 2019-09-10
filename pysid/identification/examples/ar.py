"""
    In this example we use the pysId library to estimate an AR model
"""
#Import Libraries
from numpy import convolve
from numpy.random import randn       #To generate the experiment
from scipy.signal import lfilter     #To generate the data
from pysid import ar                 #To estimate an arx model
#True System
na = 4
#with the following true parameters
Ao = convolve(convolve([1, -0.5],[1, -0.7]), convolve([1, -0.8], [1, -0.9]))
#True parameter vector
#Generate the experiment
#The true system is generates by the following relation:
# S: Ao(q)y(t) = e(t),
#with u(t) the input and e white noise.
#Number of Samples
N = 100
#Generate gaussian white noise with standat deviation 0.01
e = 0.01*randn(N, 1)
#Calculate the y through S (ARX: G(q) = B(q)/A(q) and H(q) = 1/A(q))
y = lfilter([1], Ao, e, axis=0)
#Estimate the model using Yule-Walker method
Ayw = ar(na, y)
#Estimate the model using Burg method
Aburg = ar(na, y, 'burg')
#Estimate the model using pem
Apem = ar(na, y, 'pem')
