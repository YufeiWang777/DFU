from abc import ABC

import torch
import torch.nn as nn
from scipy.stats import truncnorm
import math

import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fb import calculate_FB_bases


import numpy as np
from matplotlib import pyplot as plt

def tensor2np(t):
    c5_np = t.detach().data.cpu().numpy()
    c5_np = c5_np.squeeze(0)
    c5_np = c5_np.transpose((1, 2, 0))
    return c5_np

def colortensor(t):
    c5_np = t.detach().data.cpu().numpy()
    c5_np = c5_np.squeeze(0)
    c5_np = c5_np.transpose((1, 2, 0))
    c5_np = np.sum(c5_np, axis=-1, keepdims=True)
    plt.imshow(c5_np)
    plt.show()


def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)

class Basic2d(nn.Module, ABC):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out

class Basic2dTrans(nn.Module, ABC):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=2, padding=1, output_padding=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class BasicBlock(nn.Module, ABC):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, act=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.act = act

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.act:
            out = self.relu(out)
        return out

class StoDepth_BasicBlock(nn.Module, ABC):
    expansion = 1

    def __init__(self, prob, m, multFlag, inplanes, planes, stride=1, downsample=None):
        super(StoDepth_BasicBlock, self).__init__()
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = m
        self.multFlag = multFlag

    def forward(self, x):

        identity = x.clone()

        if self.training:
            if torch.equal(self.m.sample(), torch.ones(1)):

                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
            else:
                # Resnet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False

                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
        else:

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            if self.multFlag:
                out = self.prob * out + identity
            else:
                out = out + identity

        out = self.relu(out)

        return out

class Guide(nn.Module, ABC):

    def __init__(self, input_planes, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = Basic2d(input_planes*2, input_planes, norm_layer)

    def forward(self, feat, weight):
        weight = torch.cat((feat, weight), dim=1)
        weight = self.conv(weight)
        return weight

class Contaction(nn.Module, ABC):

    def __init__(self, input_planes):
        super().__init__()

        self.conv1 = Conv3x3(input_planes * 2, input_planes)

    def forward(self, feat1, feat2):
        feat = torch.cat((feat1, feat2), dim=1)
        feat = self.conv1(feat)

        return feat

def bases_list(ks, num_bases):
    len_list = ks // 2
    b_list = []
    for i in range(len_list):
        kernel_size = (i+1)*2+1
        normed_bases, _, _ = calculate_FB_bases(i+1)
        normed_bases = normed_bases.transpose().reshape(-1, kernel_size, kernel_size).astype(np.float32)[:num_bases, ...]

        pad = len_list - (i+1)
        bases = torch.Tensor(normed_bases)
        bases = F.pad(bases, (pad, pad, pad, pad, 0, 0)).view(num_bases, ks*ks)
        b_list.append(bases)
    return torch.cat(b_list, 0)

class DCF(nn.Module, ABC):

    def __init__(self, input_planes, weight_planes, input_ks=3):
        super().__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        self.num = input_ks * input_ks
        self.stride = 1
        self.kernel_size = input_ks
        self.padding = int((input_ks - 1) / 2)
        self.dilation = 1
        self.input_planes = input_planes

        # PARAMETERS FOR DCF
        self.num_bases = 6
        bias = True

        bases = bases_list(self.kernel_size, self.num_bases)
        self.register_buffer('bases', torch.Tensor(bases).float())
        self.tem_size = len(bases)
        bases_size = self.num_bases * len(bases)
        self.conv_bases = nn.Sequential(
            nn.Conv2d(input_planes + weight_planes, input_planes, kernel_size=3, padding=1, stride=1),
            # nn.BatchNorm2d(input_planes),
            nn.Tanh(),
            nn.Conv2d(input_planes, bases_size, kernel_size=1, padding=0),
            # nn.BatchNorm2d(bases_size),
            nn.Tanh()
            )

        # TODO
        self.coef = Parameter(torch.Tensor(input_planes, input_planes*self.num_bases, 1, 1))
        # self.coef = Parameter(torch.Tensor(input_planes, self.num_bases, 1, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(input_planes))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # self.br = nn.Sequential(
        #     norm_layer(num_features=input_planes),
        #     nn.ReLU(inplace=True),
        # )
        # self.conv3 = Basic2d(input_planes, input_planes, norm_layer)

    def forward(self, feat, weight):

        N, C, H, W = feat.shape
        H = H // self.stride
        W = W // self.stride
        drop_rate = 0.0

        weight = torch.cat((feat, weight), dim=1)
        bases = self.conv_bases(F.dropout2d(weight, p=drop_rate, training=self.training)).view(N, self.num_bases, self.tem_size, H, W) # BxMxMxHxW
        bases = torch.einsum('bmkhw, kl->bmlhw', bases, self.bases)

        # l = k*k
        x = F.unfold(F.dropout2d(feat, p=drop_rate, training=self.training), kernel_size=self.kernel_size,
                     stride=self.stride,
                     padding=self.padding).view(N, self.input_planes, self.kernel_size * self.kernel_size, H, W)
        bases_out = torch.einsum('bmlhw, bclhw-> bcmhw', bases.view(N, self.num_bases, -1, H, W),
                                 x).reshape(N, self.input_planes * self.num_bases, H, W)
        bases_out = F.dropout2d(bases_out, p=drop_rate, training=self.training)

        # TODO
        out = F.conv2d(bases_out, self.coef, self.bias)

        # out = self.br(out)
        # out = self.conv3(out)

        return out

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.coef.size(1))

        nn.init.kaiming_normal_(self.coef, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            self.bias.data.zero_()

class HiddenEncoder(nn.Module, ABC):
    def __init__(self, args, block=StoDepth_BasicBlock, multFlag=True, layers=(2, 2, 2, 2), norm_layer=nn.BatchNorm2d):
        super(HiddenEncoder, self).__init__()

        self.args = args
        self._norm_layer = norm_layer
        bc = args.bc_gru

        self.multFlag = multFlag
        prob_0_L = (1, args.prob_bottom_hiddeencoder)
        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0] - prob_0_L[1]
        self.prob_step = self.prob_delta / (sum(layers) - 1)

        in_channels = bc
        self.conv_lidar = Basic2d(1, in_channels, norm_layer=None, kernel_size=5, padding=2)

        self.inplanes = in_channels
        self.layer1_lidar = self._make_layer(block, in_channels * 2, layers[0], stride=1)

        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_lidar = self._make_layer(block, in_channels * 4, layers[1], stride=2)

        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_lidar = self._make_layer(block, in_channels * 8, layers[2], stride=2)

        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_lidar = self._make_layer(block, in_channels * 8, layers[3], stride=2)

        self._initialize_weights()

    def forward(self, c):

        c = self.conv_lidar(c)
        c = self.layer1_lidar(c)
        c = self.layer2_lidar(c)
        c = self.layer3_lidar(c)
        c = self.layer4_lidar(c)
        c = torch.tanh(c)

        return c

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob_now]))
        layers = [block(self.prob_now, m, self.multFlag, self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        self.prob_now = self.prob_now - self.prob_step
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob_now]))
            layers.append(block(self.prob_now, m, self.multFlag, self.inplanes, planes))
            self.prob_now = self.prob_now - self.prob_step

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        def truncated_normal_(num, mean=0., std=1.):
            lower = -2 * std
            upper = 2 * std
            X = truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
            samples = X.rvs(num)
            output = torch.from_numpy(samples)
            return output

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                data = truncated_normal_(m.weight.nelement(), mean=0, std=math.sqrt(1.3 * 2. / n))
                data = data.type_as(m.weight.data)
                m.weight.data = data.view_as(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-r) * h + r * q

        return h

# class ConvGRU_DCF(nn.Module):
#     def __init__(self, hidden_dim=128, input_dim=192+128):
#         super(ConvGRU_DCF, self).__init__()
#         self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
#         self.convq = DCF(hidden_dim, input_dim)

#     def forward(self, h, x):
#         hx = torch.cat([h, x], dim=1)

#         r = torch.sigmoid(self.convr(hx))
#         q = torch.tanh(self.convq(r * h, x))

#         h = (1-r) * h + r * q

#         return h

class ConvGRU_DCF(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU_DCF, self).__init__()
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = DCF(hidden_dim, input_dim)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(r * h, x))

        h = r * h + (1-r) * q

        return h

class ContextEncoder(nn.Module, ABC):

    def __init__(self, input_dim, output_dim, block=BasicBlock, norm_layer=nn.BatchNorm2d):
        super(ContextEncoder, self).__init__()

        m = input_dim // output_dim
        if m == 1:
            inter_dim = output_dim
        else:
            inter_dim = output_dim * (m//2)

        self.convf1 = Basic2d(input_dim, inter_dim, norm_layer=norm_layer, kernel_size=1, padding=0)
        self.ref1 = block(inter_dim, inter_dim, norm_layer=norm_layer, act=True)

        self.convf2 = Basic2d(inter_dim, output_dim, norm_layer=norm_layer, kernel_size=1, padding=0)
        self.ref2 = block(output_dim, output_dim, norm_layer=norm_layer, act=True)

    def forward(self, context):

        f1 = self.convf1(context)
        f1 = self.ref1(f1)
        f2 = self.convf2(f1)
        f2 = self.ref2(f2)

        return f2

class UpsampleBlock(nn.Module, ABC):

    def __init__(self, input_dim, output_dim):
        super(UpsampleBlock, self).__init__()

        self.transconv = nn.ConvTranspose2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3,
                                       stride=2, padding=1, output_padding=1, bias=True)

    def forward(self, context):

        f1 = self.transconv(context)
        # f1 = torch.tanh(f1)

        return f1

class DepthHead(nn.Module, ABC):

    def __init__(self, inter_dim, block=BasicBlock, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.convd1 = Basic2d(1, inter_dim, norm_layer=None, kernel_size=3, padding=1)
        self.convd2 = Basic2d(inter_dim, inter_dim, norm_layer=None, kernel_size=3, padding=1)

        self.convf1 = Basic2d(inter_dim, inter_dim, norm_layer=None, kernel_size=3, padding=1)
        self.convf2 = Basic2d(inter_dim, inter_dim, norm_layer=None, kernel_size=3, padding=1)

        self.conv = Basic2d(inter_dim * 2, inter_dim * 2, norm_layer=None, kernel_size=3, padding=1)
        self.ref = block(inter_dim * 2, inter_dim * 2, norm_layer=norm_layer, act=False)

        self.conv_bases = nn.Sequential(
            nn.Conv2d(inter_dim * 2, inter_dim, kernel_size=1, padding=0, stride=1),
            # nn.BatchNorm2d(bc),
            # nn.Tanh(),
            nn.Conv2d(inter_dim, 1, kernel_size=1, padding=0),
            # nn.BatchNorm2d(bases_size),
            # nn.Tanh()
        )

    def forward(self, x, depth):

        d1 = self.convd1(depth)
        d2 = self.convd2(d1)

        f1 = self.convf1(x)
        f2 = self.convf2(f1)

        input_feature = torch.cat((d2, f2), dim=1)
        input_feature = self.conv(input_feature)
        feature = self.ref(input_feature)

        out = self.conv_bases(feature)

        out = out + depth

        return out


class Model(nn.Module, ABC):

    def __init__(self, args, block=StoDepth_BasicBlock, multFlag=True, layers=(2, 2, 2, 2, 2),
                 norm_layer=nn.BatchNorm2d, guide=Guide):
        super().__init__()
        self.args = args
        self.dep_max = None
        self.kernel_size = args.kernel_size
        self._norm_layer = norm_layer
        self.preserve_input = True
        bc = args.bc
        bc_gru = args.bc_gru
        in_channels = bc * 2
        gru_channels = bc_gru * 8

        self.multFlag = multFlag
        prob_0_L = (1, args.prob_bottom)
        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0] - prob_0_L[1]
        self.prob_step = self.prob_delta / (sum(layers) - 1)

        self.cnet_totiny_0 = HiddenEncoder(args)
        self.cnet_totiny_1 = HiddenEncoder(args)
        self.cnet_totiny_2 = HiddenEncoder(args)

        self.conv_img = Basic2d(3, in_channels, norm_layer=norm_layer, kernel_size=5, padding=2)
        self.conv_lidar = Basic2d(1, in_channels, norm_layer=None, kernel_size=5, padding=2)

        self.inplanes = in_channels
        self.layer1_img, self.layer1_lidar = self._make_layer(block, in_channels * 2, layers[0], stride=1)
        self.guide1 = guide(in_channels * 2)

        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_img, self.layer2_lidar = self._make_layer(block, in_channels * 4, layers[1], stride=2)
        self.guide2 = guide(in_channels * 4)

        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_img, self.layer3_lidar = self._make_layer(block, in_channels * 8, layers[2], stride=2)
        self.guide3 = guide(in_channels * 8)

        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_img, self.layer4_lidar = self._make_layer(block, in_channels * 8, layers[3], stride=2)
        self.guide4 = guide(in_channels * 8)

        self.inplanes = in_channels * 8 * block.expansion
        self.layer5_img, self.layer5_lidar = self._make_layer(block, in_channels * 8, layers[4], stride=2)

        self.layer4d = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.augment4d = Contaction(in_channels * 8)
        self.context_features0_totiny = ContextEncoder(input_dim=in_channels * 8, output_dim=gru_channels)
        self.gru0_totiny = ConvGRU_DCF(hidden_dim=gru_channels, input_dim=gru_channels)
        self.gru0_totiny_1 = ConvGRU_DCF(hidden_dim=gru_channels, input_dim=gru_channels)
        self.gru0_totiny_2 = ConvGRU_DCF(hidden_dim=gru_channels, input_dim=gru_channels)
        self.upproj0_totiny = nn.Sequential(
            Basic2dTrans(gru_channels,    gru_channels//2, norm_layer),
            Basic2dTrans(gru_channels//2, gru_channels//4, norm_layer),
            Basic2dTrans(gru_channels//4, gru_channels//8, norm_layer)
        )
        self.depthhead0_totiny = DepthHead(inter_dim=gru_channels//8)

        self.layer3d = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.augment3d = Contaction(in_channels * 8)
        self.context_features1_totiny = ContextEncoder(input_dim=in_channels * 8, output_dim=gru_channels)
        self.upsample1_totiny = UpsampleBlock(gru_channels, gru_channels)
        self.upsample1_totiny_1 = UpsampleBlock(gru_channels, gru_channels)
        self.upsample1_totiny_2 = UpsampleBlock(gru_channels, gru_channels)
        self.gru1_totiny = ConvGRU_DCF(hidden_dim=gru_channels, input_dim=gru_channels)
        self.gru1_totiny_1 = ConvGRU_DCF(hidden_dim=gru_channels, input_dim=gru_channels)
        self.gru1_totiny_2 = ConvGRU_DCF(hidden_dim=gru_channels, input_dim=gru_channels)
        self.upproj1_totiny = nn.Sequential(
            Basic2dTrans(gru_channels,    gru_channels//2, norm_layer),
            Basic2dTrans(gru_channels//2, gru_channels//8, norm_layer)
        )
        self.depthhead1_totiny = DepthHead(inter_dim=gru_channels//8)

        self.layer2d = Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer)
        self.augment2d = Contaction(in_channels * 4)
        self.context_features2_totiny = ContextEncoder(input_dim=in_channels * 4, output_dim=gru_channels//2)
        self.upsample2_totiny = UpsampleBlock(gru_channels, gru_channels//2)
        self.upsample2_totiny_1 = UpsampleBlock(gru_channels, gru_channels//2)
        self.upsample2_totiny_2 = UpsampleBlock(gru_channels, gru_channels//2)
        self.gru2_totiny = ConvGRU_DCF(hidden_dim=gru_channels//2, input_dim=gru_channels//2)
        self.gru2_totiny_1 = ConvGRU_DCF(hidden_dim=gru_channels//2, input_dim=gru_channels//2)
        self.gru2_totiny_2 = ConvGRU_DCF(hidden_dim=gru_channels//2, input_dim=gru_channels//2)
        self.upproj2_totiny = nn.Sequential(
            Basic2dTrans(gru_channels//2, gru_channels//8, norm_layer)
        )
        self.depthhead2_totiny = DepthHead(inter_dim=gru_channels//8)

        self.layer1d = Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer)
        self.augment1d = Contaction(in_channels * 2)
        self.conv = Basic2d(in_channels * 2, in_channels, norm_layer)
        self.augment0d = Contaction(in_channels)
        self.context_features3_totiny = ContextEncoder(input_dim=in_channels, output_dim=gru_channels//8)
        self.upsample3_totiny = UpsampleBlock(gru_channels//2, gru_channels//8)
        self.upsample3_totiny_1 = UpsampleBlock(gru_channels//2, gru_channels//8)
        self.upsample3_totiny_2 = UpsampleBlock(gru_channels//2, gru_channels//8)
        self.gru3_totiny = ConvGRU_DCF(hidden_dim=gru_channels//8, input_dim=gru_channels//8)
        self.gru3_totiny_1 = ConvGRU_DCF(hidden_dim=gru_channels//8, input_dim=gru_channels//8)
        self.gru3_totiny_2 = ConvGRU_DCF(hidden_dim=gru_channels//8, input_dim=gru_channels//8)
        self.depthhead3_totiny = DepthHead(inter_dim=gru_channels//8)

        self._initialize_weights()


    def forward(self, sample):

        depth = sample['dep']
        img, lidar = sample['rgb'], sample['ip']
        d_clear = sample['dep_clear']
        if self.args.depth_norm:
            bz = lidar.shape[0]
            self.dep_max = torch.max(lidar.view(bz,-1),1, keepdim=False)[0].view(bz,1,1,1)
            lidar = lidar/(self.dep_max +1e-4)
            depth = depth/(self.dep_max +1e-4)

        net_0 = self.cnet_totiny_0(depth)
        net_1 = self.cnet_totiny_1(depth)
        net_2 = self.cnet_totiny_2(depth)

        c0_img = self.conv_img(img)
        c0_lidar = self.conv_lidar(depth)

        c1_img = self.layer1_img(c0_img)
        c1_lidar = self.layer1_lidar(c0_lidar)
        c1_lidar_dyn = self.guide1(c1_lidar, c1_img)

        c2_img = self.layer2_img(c1_img)
        c2_lidar = self.layer2_lidar(c1_lidar_dyn)
        c2_lidar_dyn = self.guide2(c2_lidar, c2_img)

        c3_img = self.layer3_img(c2_img)
        c3_lidar = self.layer3_lidar(c2_lidar_dyn)
        c3_lidar_dyn = self.guide3(c3_lidar, c3_img)

        c4_img = self.layer4_img(c3_img)
        c4_lidar = self.layer4_lidar(c3_lidar_dyn)
        c4_lidar_dyn = self.guide4(c4_lidar, c4_img)

        c5_img = self.layer5_img(c4_img)
        c5_lidar = self.layer5_lidar(c4_lidar_dyn)


        depth_predictions = []
        c5 = c5_img + c5_lidar
        dc4 = self.layer4d(c5)
        # c4 = dc4 + c4_lidar_dyn
        c4 = self.augment4d(c4_lidar_dyn, dc4)

        if self.preserve_input:
            mask = torch.sum(d_clear > 0.0, dim=1, keepdim=True)
            mask = (mask > 0.0).type_as(d_clear)
            lidar = (1.0 - mask) * lidar + mask * d_clear
        else:
            lidar = lidar
        lidar = lidar.detach()

        feature0 = self.context_features0_totiny(c4)
        net0_0 = self.gru0_totiny(net_0, feature0)
        net0_1 = self.gru0_totiny_1(net_1, net0_0)
        net0_2 = self.gru0_totiny_2(net_2, net0_1)
        net0_upproj = self.upproj0_totiny(net0_2)
        output = self.depthhead0_totiny(net0_upproj, lidar)
        depth_predictions.append(output)


        dc3 = self.layer3d(c4)
        # c3 = dc3 + c3_lidar_dyn
        c3 = self.augment3d(c3_lidar_dyn, dc3)

        if self.preserve_input:
            mask = torch.sum(d_clear > 0.0, dim=1, keepdim=True)
            mask = (mask > 0.0).type_as(d_clear)
            output = (1.0 - mask) * output + mask * d_clear
        else:
            output = output
        output = output.detach()

        feature1 = self.context_features1_totiny(c3)
        net0_up_0 = self.upsample1_totiny(net0_0)
        net0_up_1 = self.upsample1_totiny_1(net0_1)
        net0_up_2 = self.upsample1_totiny_2(net0_2)
        net1_0 = self.gru1_totiny(net0_up_0, feature1)
        net1_1 = self.gru1_totiny_1(net0_up_1, net1_0)
        net1_2 = self.gru1_totiny_2(net0_up_2, net1_1)
        net1_upproj = self.upproj1_totiny(net1_2)
        output = self.depthhead1_totiny(net1_upproj, output)
        depth_predictions.append(output)


        dc2 = self.layer2d(c3)
        # c2 = dc2 + c2_lidar_dyn
        c2 = self.augment2d(c2_lidar_dyn, dc2)

        if self.preserve_input:
            mask = torch.sum(d_clear > 0.0, dim=1, keepdim=True)
            mask = (mask > 0.0).type_as(d_clear)
            output = (1.0 - mask) * output + mask * d_clear
        else:
            output = output
        output = output.detach()

        feature2 = self.context_features2_totiny(c2)
        net1_up_0 = self.upsample2_totiny(net1_0)
        net1_up_1 = self.upsample2_totiny_1(net1_1)
        net1_up_2 = self.upsample2_totiny_2(net1_2)
        net2_0 = self.gru2_totiny(net1_up_0, feature2)
        net2_1 = self.gru2_totiny_1(net1_up_1, net2_0)
        net2_2 = self.gru2_totiny_2(net1_up_2, net2_1)
        net2_upproj = self.upproj2_totiny(net2_2)
        output = self.depthhead2_totiny(net2_upproj, output)
        depth_predictions.append(output)


        dc1 = self.layer1d(c2)
        # c1 = dc1 + c1_lidar_dyn
        c1 = self.augment1d(c1_lidar_dyn, dc1)
        c1 = self.conv(c1)
        # c0 = c1 + c0_lidar
        c0 = self.augment0d(c0_lidar, c1)

        if self.preserve_input:
            mask = torch.sum(d_clear > 0.0, dim=1, keepdim=True)
            mask = (mask > 0.0).type_as(d_clear)
            output = (1.0 - mask) * output + mask * d_clear
        else:
            output = output
        output = output.detach()

        feature3 = self.context_features3_totiny(c0)
        net2_up_0 = self.upsample3_totiny(net2_0)
        net2_up_1 = self.upsample3_totiny_1(net2_1)
        net2_up_2 = self.upsample3_totiny_2(net2_2)
        net3_0 = self.gru3_totiny(net2_up_0, feature3)
        net3_1 = self.gru3_totiny_1(net2_up_1, net3_0)
        net3_2 = self.gru3_totiny_2(net2_up_2, net3_1)
        output = self.depthhead3_totiny(net3_2, output)
        depth_predictions.append(output)

        if self.args.depth_norm:
            depth_predictions = [i * self.dep_max for i in depth_predictions]

        output = {'results': depth_predictions}

        return output

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        img_downsample, depth_downsample = None, None
        if stride != 1 or self.inplanes != planes * block.expansion:
            img_downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            depth_downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob_now]))
        img_layers = [block(self.prob_now, m, self.multFlag, self.inplanes, planes, stride, img_downsample)]
        depth_layers = [block(self.prob_now, m, self.multFlag, self.inplanes, planes, stride, depth_downsample)]
        self.prob_now = self.prob_now - self.prob_step
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob_now]))
            img_layers.append(block(self.prob_now, m, self.multFlag, self.inplanes, planes))
            depth_layers.append(block(self.prob_now, m, self.multFlag, self.inplanes, planes))
            self.prob_now = self.prob_now - self.prob_step

        return nn.Sequential(*img_layers), nn.Sequential(*depth_layers)

    def _initialize_weights(self):
        def truncated_normal_(num, mean=0., std=1.):
            lower = -2 * std
            upper = 2 * std
            X = truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
            samples = X.rvs(num)
            output = torch.from_numpy(samples)
            return output

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                data = truncated_normal_(m.weight.nelement(), mean=0, std=math.sqrt(1.3 * 2. / n))
                data = data.type_as(m.weight.data)
                m.weight.data = data.view_as(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


