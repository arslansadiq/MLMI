import torch
import torch.nn.functional as F
#import json
#import argparse
from model.inhibited_softmax import inhibited_softmax as IS
#from model.model import LeNet_Inhibited as li


def nll_loss(output, target, l2_reg, config):
    '''
    par = argparse.ArgumentParser(description='PyTorch Template')
    par.add_argument('-c', '--config', default = 'config.json',
                     type=str, help = 'config file path (default: None)')
    args = par.parse_args()
    config = json.load(open(args.config))   
    '''     
    if(config['activity_regularizer'] == True):
        activity_regularizer = config['activity_regularizer_panelty']
        activity_regularizer = activity_regularizer*torch.sum(output)
    elif(config['activity_regularizer'] == False):
        activity_regularizer = 0.0
        
    weights = torch.tensor([
        1.80434783, 0.34583333, 1.25757576, 0.84693878, 1.06410256, 1.09210526,
        0.25, 3.45833333, 0.32170543, 0.94318182])
    #weights = weights.to('cuda')
    target = target.long()
    #print(target)
    #exit
    return (F.nll_loss(torch.log(IS(output , config)), target, weight=weights) + activity_regularizer + l2_reg)
