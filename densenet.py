import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict



class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features=nn.Sequential(OrderedDict([
            ("conv0",nn.Conv2d(3,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)),
            ("norm0",nn.BatchNorm2d(64,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)),
            ("relu0",nn.ReLU(inplace=True)),
            ("pool0",nn.MaxPool2d(kernel_size=3,stride=2,padding=2,dilation=1,ceil_mode=False)),
            ("denseblock1",_DenseBlock(64,6,32)),
            ("transition1",_Transition(256)),
            ("denseblock2",_DenseBlock(128,12,32)),
            ("transition2",_Transition(512)),
            ("denseblock3",_DenseBlock(256,24,32)),
            ("transition3",_Transition(1024)),
            ("denseblock4",_DenseBlock(512,16,32)),
            ("norm5",nn.BatchNorm2d(1024,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True))
        ]))

        self.classifier=nn.Linear(in_features=1024,out_features=1000,bias=True)

    def forward(self,x):
        y=self.features(x)
        logits=self.classifier(y)
        return logits     


class _Transition(nn.Module):
    def __init__(self,in_channels:int):
        super().__init__()
        self.norm=nn.BatchNorm2d(in_channels,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.relu=nn.ReLU(inplace=True)
        self.conv=nn.Conv2d(in_channels,in_channels//2,kernel_size=(1,1),stride=(1,1),bias=False)
        self.pool=nn.AvgPool2d(kernel_size=2,stride=2,padding=0)      


    def forward(self,x):
        x=self.norm(x)
        x=self.relu(x)    
        x=self.conv(x)
        x=self.pool(x)

        return x

class _DenseBlock(nn.Module):
    def __init__(self,in_channels:int,n_denselayers:int,growth_rate_channels:int):
        super().__init__()
        for i in range(n_denselayers):
            layer=_DenseLayer(in_channels+(i*growth_rate_channels))
            self.add_module(f"denselayer{i+1}",layer)  
          
    
    def forward(self,x):
        inter_med=None
        for name ,module in self.add_module.items():
            out=module(x)
            x=torch.cat([x,out],dim=1)

        return x   


class _DenseLayer(nn.Module):
    def __init__(self,in_channels:int):
        super().__init__()
        self.norm1=nn.BatchNorm2d(in_channels,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(in_channels,128,kernel_size=(1,1),stride=(1,1),bias=False)
        self.norm2=nn.BatchNorm2d(128,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.relu2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(128,32,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
    def forward(self,x):
        x=self.norm1(x)
        x=self.relu1(x)
        x=self.conv1(x)
        x=self.norm2(x)
        x=self.relu2(x)
        x=self.conv2(x)

        return x
        



