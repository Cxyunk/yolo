"""
CATA.py
This module contains the implementation of the CATA (Cross-Attention Aggregation Transformer) model.
The CATA model is a transformer-based model that combines cross-attention and aggregation mechanisms
to process and analyze time-series data.
Author: [Cxyunk]
Date: [25.6.13]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from inspect import isfunction

def exists(x):
    return x is not None0

def expand_dim(t,dim,k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def similarity(x,means):
    return torch.ensium('bld,cd->blc',x,means)

def dists_and_buckets(x, means):
    dists = similarity(x,means)
    _,buckets = torch.argmax(dists,dim=-1)
    return dists, buckets

def batched_bincount(index,num_classes,dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim,index,torch.ones_like(index,dtype=index.dtype))
    return out

def center_iter(x, means, bucket = None):
    b,l,d,dtype,num_tokens = *x.shape, x.dtype, means.shape[0]
    if not exists(bucket):
        _,buckets = dists_and_buckets(x,means)

    bins = batched_bincount(bucket,num_tokens).sum(0,keepdim=True)
    zero_mask = bins.long()==0

    means_ = buckets.new_zeros(b,num_tokens,d,dtype=dtype)
    means_.scatter_add_(-2,expand_dim(buckets,-1,d),x)
    means_=F.normalize(means_.sum(0,keepdim=True),dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)  # 中心更新
    means = means.squeeze(0)  # 移除batch维度
    return means



class IRCA(nn.Module):
    def __init__(self, dim ,qk_dim,heads):
        super().__init__()
        self.heads = heads
        self.qk_dm = qk_dim
        self.dim = dim
        self.to_v = nn.Linear(dim,dim,bias=False)
        self.to_k = nn.Linear(dim,qk_dim,bias=False)

    def forward(self,normed_x,x_means):
        x = normed_x
        if self.training:
            x_global = center_iter(F.normalize(normed_x,dim=-1),F.normalize(x_means,dim=-1))
        else:
            x_global = x_means

        k = self.to_k(x_global)
        v = self.to_v(x_global)
        k = rearrange(k,'n (h dim_head)->h n dim_head',h=self.heads)
        v = rearrange(v,'n (h dim_head)->h n dim_head',h=self.heads)

        return k,v,x_global.detach()





class IASA(nn.Module):
    def __init__(self,dim,qk_dim,heads,group_size):
        super().__init__()
        self.heads = heads
        """
        线性层用于坐标变换
        """
        self.to_q = nn.Linear(dim,qk_dim,bias=False)
        self.to_k = nn.Linear(dim,qk_dim,bias=False)
        self.to_v = nn.Linear(dim,dim,bias=False)
        self.proj = nn.Linear(dim,dim,bias=False)
        self.group_size = group_size

    def forward(self, normed_x,idx_last,k_globals,v_globals):
        x = normed_x
        B,N,_ = x.shape

        q,k,v = self.to_q(x),self.to_k(x),self.to_v(x)

        q = torch.gather(q,dim=-2,index = idx_last.expand(q.shape))
        k = torch.gather(k,dim=-2,index = idx_last.expand(k.shape))
        v = torch.gather(v,dim=-2,index = idx_last.expand(v.shape))

        gs=min(self.group_size,N)
        ng = (N + gs -1)//gs
        pad_n = ng * gs - N

        paded_q = torch.cat((q,torch.flip(q[:,N-pad_n-gs:N,:],dims=[-2])),dim =-2)
        paded_q = rearrange(paded_q,'b (ng gs) h d -> b ng h gs d',ng=ng,h=self.heads)

        paded_k = torch.cat((k,torch.flip(k[:,N-pad_n-gs:N,:],dims=[-2])),dim =-2)
        paded_k = paded_k.unfold(-2,2*gs,gs)
        paded_k = rearrange(paded_k,'b ng (h d) gs -> b ng h gs d',h = self.heads)

        paded_v = torch.cat((v,torch.flip(v[:,N-pad_n-gs:N,:],dims=[-2])),dim =-2)
        paded_v = paded_v.unfold(-2,2*gs,gs)
        paded_v = rearrange(paded_v,'b ng (h d) gs -> b ng h gs d',h = self.heads)

        out1 = F.scaled_dot_product_attention(paded_q,paded_k,paded_v)

        k_globals = k_globals.reshape(1,1,*k_globals.shape).expand(B,ng,-1,-1,-1)
        v_globals = v_globals.reshape(1,1,*v_globals.shape).expand(B,ng,-1,-1,-1)

        out2 = F.scaled_dot_product_attention(paded_q,k_globals,v_globals)

        out = out1 + out2
        out = rearrange(out,'b ng h gs d -> b (ng gs) (h d)')[:,:N,:]

        out = out.scatter_(dim=-2,index=idx_last.expand(out.shape),src=out)
        out = self.proj(out)

        return out

class dwconv(nn.Module):
    def __init__(self,hidden_features,kernel_size=3):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features,hidden_features,kernel_size=kernel_size,stride = 1,padding=(kernel_size - 1)//2,dilation=1,
            groups = hidden_features),
            nn.GELU()
        )
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x=x.transpose(1,2).view(x.shape[0],self.hidden_features,x_size[0],x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1,2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,kernel_size = 5,act_function=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_function()
        self.dwconv = dwconv(hidden_features = hidden_features,kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features,out_features)

    def forward(self,x,x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x+self.dwconv(x,x_size)
        x = self.fc2(x)
        return x

class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.norm = nn.Layernorm(dim)
        self.fn = fn

    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)



class TAB(nn.Module):
    def __init__(self,dim,qk_dim,mlp_dim,heads,n_iter=3,num_tokens=8,group_size=128,ema_decay=0.999):
        super().__init__()

        self.n_iter=n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens

        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim,ConvFFN(dim,mlp_dim))
        
        self.IASA = IASA(dim,qk_dim,heads,group_size)
        self.IRCA = IRCA(dim,qk_dim,heads)

        self.register_buffer('means',torch.rand(num_tokens,dim))
        self.register_buffer('initted',torch.tensor(False))

        self.conv1x1 = nn.Conv2d(dim,dim,1,bias=False)

    def forward(self,x):
        