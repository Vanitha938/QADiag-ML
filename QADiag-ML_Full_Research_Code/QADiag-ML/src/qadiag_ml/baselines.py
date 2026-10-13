import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from .data import STREAMS

def concatenate(feat):return np.concatenate([feat[k] for k in STREAMS],axis=1)

def sklearn_baseline(name,seed=42):
    choices={'logistic':LogisticRegression(max_iter=2000,solver='saga',penalty='elasticnet',l1_ratio=.5,C=1.,random_state=seed),
        'random_forest':RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=seed,n_jobs=-1),
        'gradient_boosting':HistGradientBoostingClassifier(max_iter=200,random_state=seed),
        'mlp':MLPClassifier(hidden_layer_sizes=(128,64),max_iter=250,early_stopping=True,random_state=seed)}
    return choices[name]

class NeuralBaseline(nn.Module):
    """concatenation, transformer, graph and late fusion baseline choices."""
    def __init__(self,input_dims,kind='fusion',dropout=.2):
        super().__init__();self.kind=kind
        self.enc=nn.ModuleDict({k:nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,32)) for k,d in input_dims.items()})
        if kind=='transformer':self.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(32,4,64,dropout=dropout,batch_first=True),2)
        if kind=='gnn':self.msg=nn.ModuleList([nn.Linear(32,32) for _ in range(2)])
        if kind=='late_fusion':self.heads=nn.ModuleList([nn.Linear(32,3) for _ in range(3)])
        self.head=nn.Sequential(nn.Linear(96 if kind=='fusion' else 32,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,3))
    def forward(self,inputs):
        x=torch.stack([self.enc[k](inputs[k]) for k in STREAMS],1)
        if self.kind=='late_fusion':return torch.stack([head(x[:,j]) for j,head in enumerate(self.heads)],dim=0).mean(0)
        if self.kind=='transformer':x=self.tr(x)
        if self.kind=='gnn':
            for layer in self.msg:x=torch.relu(layer((x+x.mean(1,keepdim=True))/2))
        return self.head(x.flatten(1) if self.kind=='fusion' else x.mean(1))

class TemperatureScaler(nn.Module):
    def __init__(self):super().__init__();self.log_t=nn.Parameter(torch.zeros(()))
    def forward(self,logits):return logits/self.log_t.exp().clamp(min=.05)
    def fit(self,logits,labels):
        optimizer=torch.optim.LBFGS([self.log_t],lr=.05,max_iter=100)
        def closure():
            optimizer.zero_grad();loss=nn.functional.cross_entropy(self(logits),labels);loss.backward();return loss
        optimizer.step(closure);return self
