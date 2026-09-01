"""Small CPU-friendly non-linear recommendation models for V2 model search."""
from __future__ import annotations
import torch
from torch import nn


class DeepFM(nn.Module):
    def __init__(self, dim, fields, k=8):
        super().__init__(); self.linear=nn.Embedding(dim,1); self.embed=nn.Embedding(dim,k)
        self.deep=nn.Sequential(nn.Linear(fields*k,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.normal_(self.embed.weight,std=.01); nn.init.zeros_(self.linear.weight)
    def forward(self,x):
        e=self.embed(x); s=e.sum(1); fm=.5*((s*s).sum(1)-(e*e).sum((1,2)))
        return self.linear(x).sum(1).squeeze(1)+fm+self.deep(e.flatten(1)).squeeze(1)


class DCN(nn.Module):
    def __init__(self, dim, fields, k=8, layers=2):
        super().__init__(); self.embed=nn.Embedding(dim,k); width=fields*k
        self.cross_w=nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(layers)])
        self.cross_b=nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(layers)])
        self.deep=nn.Sequential(nn.Linear(width,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.out=nn.Linear(width+32,1); nn.init.normal_(self.embed.weight,std=.01)
        for w in self.cross_w: nn.init.normal_(w,std=.01)
    def forward(self,x):
        x0=self.embed(x).flatten(1); cross=x0
        for w,b in zip(self.cross_w,self.cross_b): cross=x0*(cross*w).sum(1,keepdim=True)+b+cross
        return self.out(torch.cat([cross,self.deep(x0)],1)).squeeze(1)


def make_model(name, dim, fields):
    return DeepFM(dim,fields) if name=="deepfm" else DCN(dim,fields)
