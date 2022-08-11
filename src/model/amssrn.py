from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.non_local import Nonlocal_CA

def make_model(args, parent=False):
    return SRDenseNet(args)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.PReLU(),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RB(nn.Module):
    def __init__(self, n_feats):
        super(RB, self).__init__()
        self.conv3X3 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.rb = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1),
                                nn.PReLU(),
                                nn.Conv2d(n_feats, n_feats, 3, padding=1)
                                )

    def forward(self, x):
        conv3X3 = self.conv3X3(x)
        rb = self.rb(x)
        output = x + conv3X3 + rb
        return output

class ASPP(nn.Module):
    def __init__(self, n_feats):
        super(ASPP, self).__init__()
        self.d1 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats,
                            kernel_size=3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats,
                            kernel_size=3, padding=2, dilation=2)
        self.d3 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats,
                            kernel_size=3, padding=3, dilation=3)

        self.act = nn.PReLU()

    def forward(self, x):
        d1 = self.act(self.d1(x))
        d2 = self.act(self.d2(x))
        d3 = self.act(self.d3(x))
        concatenation = torch.cat([d1, d2, d3], dim=1)
        return concatenation

class DB(nn.Module):
    def __init__(self, in_channels):
        super(DB, self).__init__()
        self.c1 = RB(in_channels)
        self.c2 = RB(in_channels)
        self.c3 = RB(in_channels)
        self.c4 = RB(in_channels)
        self.ca = CALayer(in_channels*5)
        self.c5 = nn.Conv2d(in_channels*5, in_channels, 1)

    def forward(self, input):
        out_c1 = self.c1(input)
        out_c2 = self.c2(out_c1)
        out_c3 = self.c3(out_c2)
        out_c4 = self.c4(out_c3)
        concatenation_1 = torch.cat([input, out_c1, out_c2, out_c3, out_c4], dim=1)
        ca = self.ca(concatenation_1)
        out_c5 = self.c5(ca)
        out_fused = out_c5 + input
        return out_fused

class DB_ASPP(nn.Module):
    def __init__(self, in_channels):
        super(DB_ASPP, self).__init__()
        self.c1 = RB(in_channels)
        self.c2 = RB(in_channels)
        self.c3 = RB(in_channels)
        self.c4 = RB(in_channels)
        self.aspp = ASPP(in_channels*4)
        self.ca = CALayer(in_channels*12)
        self.c5 = nn.Conv2d(in_channels*12, in_channels, 1)

    def forward(self, input):
        out_c1 = self.c1(input)
        out_c2 = self.c2(out_c1)
        out_c3 = self.c3(out_c2)
        out_c4 = self.c4(out_c3)
        concatenation_1 = torch.cat([out_c1, out_c2, out_c3, out_c4], dim=1)
        aspp = self.aspp(concatenation_1)
        ca = self.ca(aspp)
        out_c5 = self.c5(ca)
        out_fused = out_c5 + input
        return out_fused

class FPN_Fusion(nn.Module):
    def __init__(self, num_features, n_feats=64):
        super(FPN_Fusion, self).__init__()
        module_fusion = nn.ModuleList()
        for _ in range(num_features):
            module_fusion.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.fusion = nn.Sequential(*module_fusion)

    def forward(self, feature_list):
        fused_last = self.fusion[0](feature_list[-1])
        fusion = []
        fusion.append(fused_last)
        for i in range(len(feature_list)-1):
            fused = self.fusion[i+1](feature_list[-(i+2)]+feature_list[-(i+1)])
            fusion.append(fused)
        return fusion


class AMSSRN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(AMSSRN, self).__init__()

        n_feats = 64
        n_blocks = 8
        kernel_size = 3
        scale = args.scale[0]

        self.n_blocks = n_blocks

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks // 2):
            modules_body.append(DB(n_feats))
        for i in range(n_blocks // 2):
            modules_body.append(DB_ASPP(n_feats))

        # define tail module
        # modules_tail = [nn.Conv2d(in_channels=n_feats, out_channels=n_feats * scale * scale,
        #                           kernel_size=3, padding=1),
        #                 nn.PixelShuffle(scale),
        #                 nn.Conv2d(in_channels=n_feats, out_channels=args.n_colors,
        #                           kernel_size=3, padding=1)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.fpn_fusion = FPN_Fusion(n_blocks + 3)
        self.feature_bank = nn.Conv2d(in_channels=(n_blocks + 3) * n_feats,
                                      out_channels=n_feats, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.non_local_1 = Nonlocal_CA(in_feat=n_feats, inter_feat=n_feats // 8)
        self.non_local_2 = Nonlocal_CA(in_feat=n_feats, inter_feat=n_feats // 8)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = common.Upsampler(conv, scale, n_feats)
        self.reconstruction = nn.Conv2d(n_feats, args.n_colors, 3, padding=1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        head = x
        x = self.non_local_1(x)
        non_local_1 = x
        MSRB_out = []
        MSRB_out.append(head)
        MSRB_out.append(non_local_1)

        for i in range(self.n_blocks):
            x = self.body[i](x)
            x = x + self.gamma * non_local_1
            MSRB_out.append(x)

        x = self.non_local_2(x)
        MSRB_out.append(x)
        fused_msrb_out = self.fpn_fusion(MSRB_out)
        feature_bank = self.feature_bank(torch.cat(fused_msrb_out, 1))
        bottleneck = head + feature_bank
        x = self.reconstruction(self.tail(bottleneck))
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

if __name__ == '__main__':
    from option import args
    msrn = MSRN(args)
    param = [param.numel() for param in msrn.parameters()]
    print(sum(param))
