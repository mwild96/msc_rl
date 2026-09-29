import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Define your neural networks in this class.
# Use the __init__ method to define the architecture of the network
# and define the computations for the forward pass in the forward method.

class LinearNetwork(nn.Module):
    def __init__(self,inputDim,outputDim):
        super(LinearNetwork, self).__init__()
        self.linear = nn.Linear(inputDim,outputDim)
    def forward(self, inputs) :
        outputs = self.linear(inputs)
        return outputs

class ValueNetwork(nn.Module):
    def __init__(self,inputDim,outputDim):
        self.linear1 = nn.Linear(6,100)
        self.linear2 = nn.Linear(100,50)
        self.linear3 = nn.Linear(50,outputDim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        outputs = F.relu(self.linear3(x))
        return outputs
