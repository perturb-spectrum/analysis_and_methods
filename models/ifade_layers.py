"""
Ours, ode_mnist_layers, sodef_layers: all 3 content mixed together
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import geotorch
from torch.nn.parameter import Parameter
from torchdiffeq import odeint_adjoint as odeint

from .dm_wide_resnet import CIFAR10_MEAN, CIFAR10_STD, DMWideResNet, Swish
from .sodef_layers import Identity, MLP_OUT_ORTH1024, ODEfunc_mlp, MLP_OUT_LINEAR
from .sodef_layers import ODEBlock as ODEBlock_SODEF
from .ode_mnist_layers import conv1x1, ResBlock, norm, Flatten, ConcatConv2d
from .ode_mnist_layers import ODEfunc as ODEfunc_mnist
from .ode_mnist_layers import ODEBlock as ODEBlock_mnist

class ODEfunc(nn.Module):
    """
    For f0: 
        input size: [16, 32, 32]
        we need output size to also be the same!
            output = (input - kernel + 2* padding)/stride + 1
            So, kernel = 3, stride = 1, padding = 1 satisifes! This is what was used in ODEfunc_mnist which has been copied below.
    """
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0
        self.eq = 0

    def forward(self, t, x):
        self.nfe += 1
        # print(self.nfe, x.shape)
        out = self.norm1(x)
        # print(out.shape)
        out = self.relu(out)
        # print(out.shape)
        out = self.conv1(t, out)
        # print(out.shape)
        out = self.norm2(out)
        # print(out.shape)
        out = self.relu(out)
        # print(out.shape)
        out = self.conv2(t, out)
        # print(out.shape)
        out = self.norm3(out)
        # print(out.shape)
        return out

class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

def ifade_ode():
    odefeature_layers = ODEBlock(ODEfunc(16)) # for f0: init_conv(channels_in = 3, channels_out = 16)
    return odefeature_layers


def ode_mnist_model():
    downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]
    feature_layers = [ODEBlock_mnist(ODEfunc_mnist(64))]
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]
    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers)

    return model

def rebuffi_sodef():
    model = DMWideResNet(num_classes=10,
                         depth=70,
                         width=16,
                         activation_fn=Swish,
                         mean=CIFAR10_MEAN,
                         std=CIFAR10_STD)
    model.logits = Identity()
    fc_features = MLP_OUT_ORTH1024()
    odefunc = ODEfunc_mlp(0)
    odefeature_layers = ODEBlock_SODEF(odefunc)
    odefc_layers = MLP_OUT_LINEAR()
    model_dense = nn.Sequential(odefeature_layers, odefc_layers)
    new_model = nn.Sequential(model, fc_features, model_dense)
    
    return new_model