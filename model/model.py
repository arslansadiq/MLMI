import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision.models as models
from model.activation import pdf_cauchy_distribution as PCD
from model.activation import cauchy_activation
import json
import argparse

class LeNet_Inhibited(BaseModel):
    def __init__(self, num_classes=10):
        super(LeNet_Inhibited, self).__init__()
        '''
        Initialize the model structure for inhibited softmax.
        Derived form the paper:
        https://arxiv.org/abs/1810.01861
        '''
        
        par = argparse.ArgumentParser(description='Model_Lenet')
        par.add_argument('-c', '--config', default = 'config.json', type=str, help = 'config file path (default: None)')
        args = par.parse_args()
        config = json.load(open(args.config))
        """Model Architacture"""
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, bias=True)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, bias=True)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120, bias=True)
        self.fc2 = nn.Linear(in_features=120, out_features=84, bias=True)
        self.fc3 = nn.Linear(in_features=84, out_features=10, bias=config['arch']['last_layer_bias'])
    
    def forward(self, x):
        '''
        This function implements the computations of the 5-layered 
        LeNet_Inhibited model. It takes in the image or bath of images
        and passes it/them through the whole model.
        '''
        
        '''First Layer''' 
        x = self.conv1(x)       # 2d-Convolution
        x = F.relu(x)           # applying activaion function
        x = F.avg_pool2d(x, 2, stride=2)   # apply average-pooling to extract import features
        
        '''Second Layer'''
        x = self.conv2(x)       # 2d-Convolution
        x = F.relu(x)           # applying activation function
        x = F.avg_pool2d(x, 2, stride=2)   # apply average-pooling to extract import features
        
        '''Third Layer'''
        x = x.view(-1, 16*5*5)  # flattening the whole output of second layer
        x = self.fc1(x)         # applying the linear computation (w*x + b)
        x = F.relu(x)           # applying relu activation
        
        '''Fourth Layer'''
        x = self.fc2(x)         # applying the linear computation (w*x + b)
        x = PCD(x)              # applying pdf-cauchy-distribution activation
        
        '''Fifth Layer'''
        x = self.fc3(x)         # applying the linear computation (w*x)
        
        return x
        
    
class CustomNetwork(BaseModel):
    def __init__(self, num_classes=10):
        super(CustomNetwork, self).__init__()
        
        par = argparse.ArgumentParser(description='Model_CustomNetwork')
        par.add_argument('-c', '--config', default = 'config.json', type=str, help = 'config file path (default: None)')
        args = par.parse_args()
        config = json.load(open(args.config))
        
        '''Model Architacture'''
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=80, kernel_size=3, bias=True, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features = 80, affine=True)
        self.conv1_drop = nn.Dropout2d(p=0.25)
        self.conv2 = nn.Conv2d(in_channels=80, out_channels=160, kernel_size=3, bias=True, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features = 160, affine=True)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.conv3 = nn.Conv2d(in_channels=160, out_channels=240, kernel_size=3, bias=True, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(num_features = 240, affine=True)
        self.conv3_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(in_features=240*4*4, out_features=200, bias=True)
        self.fc1_drop = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=200, out_features=100, bias=True)
        self.fc3 = nn.Linear(in_features=100, out_features=num_classes, bias=config['arch']['last_layer_bias'])
        
    def forward(self, x):
        '''
        This function implements the CustomNetwork computations by
        taking in an image or batch of images
        '''
        
        '''First Layer'''
        x = self.batchnorm1((self.conv1(x)))
        x = F.relu(x)
        x = self.conv1_drop(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        '''Second Layer'''
        x = self.batchnorm2((self.conv2(x)))
        x = F.relu(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        '''Third Layer'''
        x = self.batchnorm3((self.conv3(x)))
        x = F.relu(x)
        x = self.conv3_drop(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        '''Fourth Layer'''
        x = x.view(-1, 240*4*4)
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.fc1_drop(x)
        
        '''Fifth Layer'''
        x = self.fc2(x)
        x = PCD(x)
        
        '''Sixth Layer'''
        x = self.fc3(x)
        
        return x
        

class Resnet18(BaseModel):
    def __init__(self, classes=10):
        super(Resnet18, self).__init__()
        par = argparse.ArgumentParser(description='Model_resnet18')
        par.add_argument('-c', '--config', default = 'config.json', type=str, help = 'config file path (default: None)')
        args = par.parse_args()
        config = json.load(open(args.config))
        
        self.resnet = models.resnet18(pretrained = True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, classes, bias=config['arch']['last_layer_bias'])
        self.resnet.layer4[1].relu = cauchy_activation()
        
        ct = 0
        for child in self.resnet.children():
            #print("child ", ct, ": \n", child)
            if ct<7:
                for param in child.parameters():
                    param.requires_grad = False
            ct += 1
        #exit
        
    def forward(self, x_input):
        output = self.resnet(x_input)
        return output 
		
    
class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
