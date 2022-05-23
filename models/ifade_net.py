import torch
import torch.nn as nn



class _block(nn.Module):
    def __init__(self, dim):
        super(_block, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm = nn.GroupNorm(dim, dim) 
        self.act = nn.Softplus()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class _block_bn(nn.Module):
    def __init__(self, dim):
        super(_block_bn, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm = nn.BatchNorm2d(dim) 
        self.act = nn.Softplus()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ODEf(nn.Module):
    """
    Keep input and output shapes as [B, 16, 32, 32]
    """
    def __init__(self, dim):
        super(ODEf, self).__init__()
        self.block1 = _block(dim)
        self.block2 = _block(dim)
        self.block3 = nn.Conv2d(dim, dim, 3, 1, 1) # (size - K + 2*P)/S + 1 = size is True when K = 3, P = 1, S = 1

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        return out

class ODEf_smaller(nn.Module):
    """
    Keep input and output shapes as [B, 16, 32, 32]
    """
    def __init__(self, dim):
        super(ODEf_smaller, self).__init__()
        self.block1 = _block_bn(dim)
        # self.block2 = _block_bn(dim)
        self.block3 = nn.Conv2d(dim, dim, 3, 1, 1) # (size - K + 2*P)/S + 1 = size is True when K = 3, P = 1, S = 1

    def forward(self, x):
        out = self.block1(x)
        # out = self.block2(out)
        out = self.block3(out)
        return out

class ODELinearf(nn.Module):
    """
    Keep input and output shapes as [B, 16, 32, 32]
    """
    def __init__(self, dim):
        super(ODEf, self).__init__()
        self.b = nn.Conv2d(dim, dim, 3, 1, 1) # (size - K + 2*P)/S + 1 = size is True when K = 3, P = 1, S = 1

    def forward(self, x):
        out = self.b(x)
        return out

"""
From
https://github.com/DiffEqML/torchdyn/blob/master/tutorials/module4-model/m4f_stable_neural_odes.ipynb
"""
# Vanilla Version of stable neural flows
class Stable(nn.Module):
    """Stable Neural Flow"""
    def __init__(self, net, depthvar=False, controlled=False):
        super().__init__()
        self.net, self.depthvar, self.controlled = net, depthvar, controlled
        
    def forward(self, x):
        with torch.set_grad_enabled(True):
            bs, n = x.shape[0], x.shape[1] // 2
            x = x.requires_grad_(True)
            eps = self.net(x).sum()
            out = -torch.autograd.grad(eps, x, allow_unused=False, create_graph=True)[0] 
        out = out[:,:-1] if self.depthvar else out
        out = out[:,:-2] if self.controlled else out
        return out

class IQuantize(nn.Module):
    def __init__(self, dim):
        super(IQuantize, self).__init__()
        self.betas = torch.tensor([1.0]*dim)

    def forward(self, x):
        for channel_idx in range(len(x)):
            x[channel_idx] = torch.floor(self.betas[channel_idx]*x[channel_idx])/self.betas[channel_idx]
        return x