import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from torchvision import models
decoder_name="microsoft/biogpt"




class ImageToText(nn.Module):
    def __init__(self):
        super().__init__()


        self.encoder=models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features=self.encoder.classifier.in_features
        self.encoder.classifier=nn.Linear(in_features,1024)


        self.decoder=AutoModelForCausalLM.from_pretrained(decoder_name)
        self.decoder_tokenizer=AutoTokenizer.from_pretrained(decoder_name)

    def forward(self,image,report=None):
        image_embed=self.encoder(image)
        
        
            
