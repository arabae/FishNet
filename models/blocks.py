import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pooling import MaxPool2d


class Bottleneck_RU(nn.Module): # bottleneck residual unit
    def __init__(self, in_channel, ratio):
        super(Bottleneck_RU, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel//2, kernel_size=1, bias=False),

            nn.BatchNorm2d(in_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel//2, in_channel//2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),

            nn.BatchNorm2d(in_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel//2, int(in_channel*ratio), kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.bottleneck(x)


class Classificiation(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(Classificiation, self).__init__()
        self.pred = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel//2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel//2, n_classes, kernel_size=1, bias=True)
        )

    def forward(self, x):
        return self.pred(x)


class FrontFeatureExtractor(nn.Module):
    def __init__(self, out_channel):
        super(FrontFeatureExtractor, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, out_channel//2, kernel_size=1, bias=False),

            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, out_channel//2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),

            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel//2, out_channel, kernel_size=1, bias=False),

            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.bottleneck(x)

class Refinement_block(nn.Module): # Up/Down common part
    def __init__(self, in_channel, ratio, typ):
        """

        """
        super(Refinement_block, self).__init__()
        self.k = 2
        self.typ = typ

        self.M = Bottleneck_RU(in_channel, ratio)

        if self.typ == 'tail':
            self.channel_increase = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, in_channel*ratio, kernel_size=1, bias=False)
            )
        elif self.typ == 'body':
            def r(input_): # r function -- channel-wise reduction
                batch_, channel_, height_, width_ = input_.size()
                input_ = input_.view(batch_, channel_//self.k, self.k, height_, width_).sum(2)
                return input_
            
            self.r = r
    
    def forward(self, x):
        res = x
        out = self.M(x)

        if self.typ == 'tail':
            res = self.channel_increase(res)
        elif self.typ == 'body':
            res = self.r(res)
        
        out += res

        return out


class Squeeze_Exicitation(nn.Module): # Squeeze-and-Excitation
    def __init__(self, in_channel):
        super(Squeeze_Exicitation, self).__init__()
        self.se = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), # convert 1x1

            nn.Conv2d(in_channel, in_channel//16, kernel_size=1), # The reduction ratio r is set to 16 by default ([11]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel//16, in_channel, kernel_size=1),
            
            nn.Sigmoid()
        )
    
    #  (2, 2) *channel-wise attentive operation* (1, 1) -> (2, 2)
    def forward(self, x):
        out = self.se(x).repeat(1, 1, 2, 2)
        out += x
        return out
