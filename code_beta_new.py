import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as pyplot
from scipy.integrate import solve_ivp

#Define PINN function
def create_pinn(input_dim =1, output_dim=4, hidden_layers=2, hidden_dim =20):
    """
       input_dim: input time points t
       output_dim: return 4 outputs
    """
    #initial conditions
    N = 1000
    S0 = 990
    E0 = 5
    I_u0 = 3
    I_r0 = 2

    beta0 = 0.3
    sigma0 = 0.2
    gamma_u = 0.1
    gamma_r = 0.1
    p0 = 0.6

    ##Input layer
    w1 = torch.nn.parameter(torch.randn(input_dim, hidden_dim)*0.1)
    b1 = torch.nn.parameter(torch.zeros(hidden_dim))

    #Hidden layers
    hidden_weights = []
    hidden_biases = []

    for i in range(hidden_layers):
        w = torch.nn.parameter(torch.randn(hidden_dim, hidden_dim)*0.1)
        b = torch.nn.parameter(torch.zeros(hidden_dim))

    #output layer
    w_out = torch.nn.parameter(torch.randn(hidden_dim, output_dim)*0.1)

    #initialize biases at t=0
    #S= 990/1000= 0.99
    #E= 5/1000= 0.005
    #I_u0 = 3/1000 =0.003
    #I_r0 = 2/1000 = 0.002
    b_out = torch.nn.parameter(torch.tensor([0.99, 0.005, 0.003, 0.002], dtype= torch.float32))

    #collect all parameters
    params = [w1, b1, hidden_weights.append(w), hidden_biases.append(b), w_out, b_out]

    return params








