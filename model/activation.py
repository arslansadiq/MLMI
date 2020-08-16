import torch
import torch.nn as nn

def pdf_cauchy_distribution(tensor):
    '''
    this fuction takes the output from neural netwrok's layer and implements a 
    kernal function which acts as an activation function.
    
    input:
    tensor: output of neural network's layer computation (w*x + b)
    
    output:
    also a tensor which after going to pdf cauchy distribution fucntion
    which is f(x) = 1/(1+x^2)
    '''
    
    return (1 / (1 + torch.mul(tensor, tensor)))


class cauchy_activation(nn.Module):
    def __init__(self):
        super(cauchy_activation, self).__init__()
        
    def forward(self, x):
        return pdf_cauchy_distribution(x)