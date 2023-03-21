import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch
from pytorch_ops import soft_rank, soft_sort
from tqdm import tqdm
import tensorflow as tf

class Data(Dataset):
    def __init__(self, X, y = None, transforms=None):
        self.X = X
        self.y = y
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):

        data = self.X[i, :]
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data
        

def loss_nn(y_pred, y_true, model , lamda = 0.01, alpha = 1.2):
    
    y_pred = soft_rank(y_pred.reshape(1,-1), regularization_strength=1)
    
    y_true = soft_rank(y_true.reshape(1,-1), regularization_strength=1)
    
    return nn.SmoothL1Loss()(y_pred,y_true)  + lamda*torch.sum(torch.abs(model.fc[4].weight)) + lamda*2*torch.sum(torch.abs(model.fc[2].weight))


class Net(nn.Module):
    def __init__(self, n_input, n_out):
        super().__init__()
        self.fc = nn.Sequential(
          nn.Linear(n_input, 50),
          nn.Tanh(),
          nn.Linear(50, 50),
          nn.Tanh(),
          nn.Linear(50, n_out)
        )
    
    def __init__(self):
        super().__init__()
    
    def build(self, X_train, y_train):
        n_input = np.array(X_train).shape[1]
        n_out = 1
        self.fc = nn.Sequential(
          nn.Linear(n_input, 50),
          nn.Tanh(),
          nn.Linear(50, 50),
          nn.Tanh(),
          nn.Linear(50, n_out)
        )
    
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.fc(x)
        return x
    
    def predict(self,x):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = Data(np.array(x))
        testloader = torch.utils.data.DataLoader(x, batch_size=len(x), num_workers=0)
        with torch.no_grad():
            for batch_idx, inputs in enumerate(testloader):
                inputs = inputs.to(device)
                outputs = self(inputs)
            outputs = outputs.reshape(-1).detach().numpy()
        return outputs
        
    
    def fit(self, X_train, y_train):
        
        
        trainset = Data(np.array(X_train),np.array(y_train))
     
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                            shuffle=True, num_workers=0)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.build(X_train, y_train)
        
        self = self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        train_epoch = 200
        loss_train_list = []
        pbar = range(train_epoch)
        for epoch in pbar:  
                self.train()
                train_loss = 0.0
                for i, (inputs, y_true) in enumerate(trainloader, 0):
                    inputs, y_true = inputs.to(device), y_true.to(device) 
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = loss_nn(outputs, y_true, self, lamda=0.03)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * len(y_true)
                
                loss_train = train_loss/len(trainset)
                
    
