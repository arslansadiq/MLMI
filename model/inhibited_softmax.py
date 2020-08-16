import torch
#import json
#import argparse

def inhibited_softmax(tensor, config):
    '''
    par = argparse.ArgumentParser(description='PyTorch Template')
    par.add_argument('-c', '--config', default = 'config.json',
                     type=str, help = 'config file path (default: None)')
    args = par.parse_args()
    config = json.load(open(args.config))
    '''
    if (config['Inhibited_softmax'] == True):
        c = config['c']
        inhibition = torch.exp(c*torch.ones((1, 1)))
    elif (config['Inhibited_softmax'] == False):
        inhibition = 0
    #inhibition = inhibition.to('cuda')
    return (torch.exp(tensor)/(torch.exp(tensor).sum(dim=1).view(-1,1) + inhibition))
