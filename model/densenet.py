import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict





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
        for name, module in self.named_children():
            out = module(x)
            x = torch.cat([x, out], dim=1)

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
        




class DenseNet_MimicCxr(nn.Module):
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

        self.proj = nn.Conv2d(1024, 512, kernel_size=1)
        self.apply(self.init_weight)

    def forward(self,x):
        y=self.features(x)
        hidden_state=self.proj(y)
        return hidden_state
    
    def init_weight(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weights,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Conv2d):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @classmethod
    def load_hf_pretrained(cls):
        
        model=DenseNet_MimicCxr()
        sd=model.state_dict()
        sd_keys = list(sd.keys())

        model_hf=models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        sd_hf=model_hf.state_dict()
        sd_hf_keys=list(sd_hf.keys())

        sd_keys=[k for k  in sd_keys if k in sd_hf_keys]
        sd_hf_keys=[k for k in sd_hf_keys if k in sd_keys]

        assert len(sd_keys)==len(sd_hf_keys) , f"state_dict do not match"

        for k in sd_keys:
            if k in sd_hf and sd[k].shape == sd_hf[k].shape:
                sd[k] = sd_hf[k]

        model.load_state_dict(sd)        
        print("loading succesful")
        return model




