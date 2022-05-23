import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_withoutReLU(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_withoutReLU, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_preReLU(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_preReLU, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(F.relu(out))
        out = self.layer2(F.relu(out))
        out = self.layer3(F.relu(out))
        out = self.layer4(F.relu(out))
        out = F.avg_pool2d(F.relu(out), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def before_f0(self, x):
        out = self.conv1(x)
        return out

    def after_f0(self, x):
        out = self.bn1(x)
        out = self.layer1(F.relu(out))
        out = self.layer2(F.relu(out))
        out = self.layer3(F.relu(out))
        out = self.layer4(F.relu(out))
        out = F.avg_pool2d(F.relu(out), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def before_f1(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(F.relu(out))
        return out

    def after_f0_before_f1(self, x):
        out = self.bn1(x)
        out = self.layer1(F.relu(out))
        return out

    def after_f1(self, x):
        out = self.layer2(F.relu(x))
        out = self.layer3(F.relu(out))
        out = self.layer4(F.relu(out))
        out = F.avg_pool2d(F.relu(out), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def before_f2(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(F.relu(out))
        out = self.layer2(F.relu(out))
        return out

    def after_f1_before_f2(self, x):
        out = self.layer2(F.relu(x))
        return out

    def before_f3(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(F.relu(out))
        out = self.layer2(F.relu(out))
        out = self.layer3(F.relu(out))
        return out

    def after_f2_before_f3(self, x):
        out = self.layer3(F.relu(x))
        return out

    def after_f2(self, x):
        out = self.layer3(F.relu(x))
        out = self.layer4(F.relu(out))
        out = F.avg_pool2d(F.relu(out), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def before_f4_avgpool(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.layer1(F.relu(out))
        out = self.layer2(F.relu(out))
        out = self.layer3(F.relu(out))
        out = self.layer4(F.relu(out))
        return out

    def after_f3_before_f4(self, x):
        out = self.layer4(F.relu(x))
        out = F.avg_pool2d(F.relu(out), 4)
        out = out.view(out.size(0), -1)
        return out

    def after_f3(self, x):
        out = self.layer4(F.relu(x))
        out = F.avg_pool2d(F.relu(out), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def after_f4_avgpool(self, x):
        out = F.avg_pool2d(F.relu(x), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def before_bnf0(self, x):
        out = self.conv1(x)
        out = self.bn1(x)
        return out

    def after_bnf0(self, x):
        out = self.layer1(F.relu(x))
        out = self.layer2(F.relu(out))
        out = self.layer3(F.relu(out))
        out = self.layer4(F.relu(out))
        out = F.avg_pool2d(F.relu(out), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def after_bnf0_before_f1(self, x):
        out = self.layer1(F.relu(x))
        return out

    def after_f3_before_lastReLU(self, x):
        out = self.layer4(F.relu(x))
        return out

    def after_lastReLU_before_f4(self, x):
        out = F.avg_pool2d(F.relu(x), 4)
        out = out.view(out.size(0), -1)
        return out

    def after_f4(self, x):
        out = self.linear(x)
        return out

    def forward_with_all_feature(self, x):
        out0 = self.conv1(x)
        out0_post_bn = self.bn1(out0)
        out1 = self.layer1(F.relu(out0_post_bn))
        out2 = self.layer2(F.relu(out1))
        out3 = self.layer3(F.relu(out2))
        out4 = self.layer4(F.relu(out3))
        out5 = F.avg_pool2d(F.relu(out4), 4)
        out = out5.view(out5.size(0), -1)
        out = self.linear(out)
        return out0, out0_post_bn, out1, out2, out3, out4, out5, out

    def forward_with_everything(self, x):
        out0 = self.bn1(self.conv1(x))
        out0_r = F.relu(out0)
        out1 = self.layer1(out0_r)
        out1_r = F.relu(out1)
        out2 = self.layer2(out1_r)
        out2_r = F.relu(out2)
        out3 = self.layer3(out2_r)
        out3_r = F.relu(out3)
        out4 = self.layer4(out3_r)
        out4_r = F.relu(out4)
        out5 = F.avg_pool2d(out4_r, 4)
        out = out5.view(out5.size(0), -1)
        out = self.linear(out)
        return out0, out0_r, out1, out1_r, out2, out2_r, out3, out3_r, out4, out4_r, out5, out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def before_f0(self, x):
        out = self.conv1(x)
        return out

    def after_f0(self, x):
        out = F.relu(self.bn1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def after_f0_before_f1(self, x):
        out = F.relu(self.bn1(x))
        out = self.layer1(out)
        return out

    def after_f1_before_f2(self, x):
        out = self.layer2(x)
        return out

    def after_f2_before_f3(self, x):
        out = self.layer3(x)
        return out

    def after_f3_before_f4(self, x):
        out = self.layer4(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def after_f4(self, x):
        out = self.linear(x)
        return out
    
    def forward_with_conv1_feature(self, x):
        out0 = self.conv1(x)
        out = F.relu(self.bn1(out0))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out0, out

    def forward_with_bn1_feature(self, x):
        out0 = self.bn1(self.conv1(x))
        out = F.relu(out0)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out0, out

    def forward_with_relu_feature(self, x):
        out0 = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out0)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out0, out

    def forward_with_layer1_feature(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out0 = self.layer1(out)
        out = self.layer2(out0)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out0, out

    def forward_with_all_feature(self, x):
        out0 = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = F.avg_pool2d(out4, 4)
        out = out5.view(out5.size(0), -1)
        out = self.linear(out)
        return out0, out1, out2, out3, out4, out5, out

    def forward_with_penultimate_feature(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out0 = out.view(out.size(0), -1)
        out = self.linear(out0)
        return out0, out

    def forward_with_layer4_feature(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out0 = self.layer4(out)
        out = F.avg_pool2d(out0, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out0, out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet18_preReLU():
    return ResNet_preReLU(BasicBlock_withoutReLU, [2, 2, 2, 2]) 


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
