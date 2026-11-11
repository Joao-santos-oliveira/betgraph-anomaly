"""Discriminacao auto-supervisionada, sem rotulos de injecao."""
import torch
from torch import nn
from torch.nn import functional as F


def corromper(x_dict,active,generator):
    result={k:v.clone() for k,v in x_dict.items()}
    for kind,x in x_dict.items():
        ids=torch.where(active[kind])[0]
        result[kind][ids]=x[ids[torch.randperm(len(ids),generator=generator)]]
    return result


class Contraste(nn.Module):
    def __init__(self,embedding):
        super().__init__()
        self.heads=nn.ModuleDict({k:nn.Bilinear(embedding,embedding,1) for k in ('user','event')})

    def forward(self,normal,corrupt,active):
        terms=[]
        for kind,head in self.heads.items():
            a=normal[kind][active[kind]];b=corrupt[kind][active[kind]]
            if not len(a):continue
            summary=torch.sigmoid(a.mean(dim=0,keepdim=True)).expand_as(a)
            logits=torch.cat([head(a,summary).flatten(),head(b,summary).flatten()])
            labels=torch.cat([torch.ones(len(a)),torch.zeros(len(b))])
            terms.append(F.binary_cross_entropy_with_logits(logits,labels))
        if not terms:raise ValueError('Sem nos ativos')
        return torch.stack(terms).mean()
