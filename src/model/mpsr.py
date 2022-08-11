from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.non_local import Nonlocal_CA

def make_model(args, parent=False):
    return MPSR(args)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ERB(nn.Module):
    def __init__(self, kernel_size, n_feats):
        super(ERB, self).__init__()
        self.conv3X3_1 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv3X3_2 = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        c1 = self.relu(self.conv3X3_1(x))
        c2 = self.conv3X3_2(c1)
        output = c1 + c2 + x
        return output


class RCAG(nn.Module):
    def __init__(self, kernel_size, n_feats):
        super(RCAG, self).__init__()
        n_blocks = 8
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(ERB(kernel_size, n_feats))
        self.body = nn.Sequential(*modules_body)
        self.ca = CALayer(n_feats)
        self.c = nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2)

    def forward(self, input):
        body = self.body(input)
        ca = self.ca(body)
        c = self.c(ca)
        output = c + input
        return output
class MPDFE(nn.Module):
    def __init__(self, n_feats):
        super(MPDFE, self).__init__()
        n_blocks = 5
        modules_upper = nn.ModuleList()
        modules_lower = nn.ModuleList()
        for i in range(n_blocks):
            modules_upper.append(RCAG(kernel_size=3, n_feats=n_feats))
        modules_upper.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        for i in range(n_blocks):
            modules_lower.append(RCAG(kernel_size=5, n_feats=n_feats))
        modules_lower.append(nn.Conv2d(n_feats, n_feats, 5, padding=2))
        self.upper = nn.Sequential(*modules_upper)
        self.lower = nn.Sequential(*modules_lower)
        self.ca = CALayer(n_feats)

    def forward(self, x):
        upper = self.upper(x)
        lower = self.lower(x)
        ca = self.ca(x + upper + lower)
        return ca


class MPSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MPSR, self).__init__()

        n_feats = 64
        scale = args.scale[0]

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, 3)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = MPDFE(n_feats)
        self.tail = common.Upsampler(conv, scale, n_feats)
        self.reconstruction = nn.Conv2d(n_feats, args.n_colors, 3, padding=1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.body(x)
        x = self.reconstruction(self.tail(x))
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

