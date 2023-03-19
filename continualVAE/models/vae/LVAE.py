from __future__ import print_function
import torch.nn as nn
import torch
import pprint
import torch.nn.functional as F

from helpers.distributions import nll, nll_activation, kl_gaussian
from helpers.utils import float_type

EPS_STD = 1e-6


class LVAE(nn.Module):
    def __init__(self, input_shape, arch, **kwargs):
        super(LVAE, self).__init__()
        # grab the meta config and print for
        self.config = kwargs['kwargs']
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)

        self.input_shape = input_shape
        self.input_chans = input_shape[0]
        self.out_chans = self.input_chans * 2

        self.build_encoder(arch)
        self.build_decoder(arch)
        if self.config['cuda']:
            self.cuda()

    def build_encoder(self, hidden_dim):
        self.u1 = ConvRMLP(self.input_chans, 32, hidden_dim, 2, 2)
        self.u2 = ConvRMLP(hidden_dim, 32, hidden_dim, 2, 2)
        self.u3 = ConvRMLP(hidden_dim, 32, hidden_dim, 2, 2)
        self.u4 = ConvRMLP(hidden_dim, 32, hidden_dim, 2, 2)
        if self.input_shape[1] == 32:
            self.u5 = ConvRMLP(hidden_dim, 32, hidden_dim, 2, 2)
        elif self.input_shape[1] == 64:
            self.u5 = ConvRMLP_multiDS(hidden_dim,
                                       32,
                                       hidden_dim,
                                       2,
                                       2,
                                       num_downsample=2)

    def build_decoder(self, hidden_dim):
        if self.input_shape[1] == 32:
            self.d4 = ConvTRMLP(16, 32, hidden_dim, 2, 2)
        elif self.input_shape[1] == 64:
            self.d4 = ConvTRMLP_multiDS(16,
                                        32,
                                        hidden_dim,
                                        2,
                                        2,
                                        num_upsample=2)

        self.d3 = ConvTRMLP(16, 32, hidden_dim, 2, 2)
        self.d2 = ConvTRMLP(16, 32, hidden_dim, 2, 2)
        self.d1 = ConvTRMLP(16, 32, hidden_dim, 2, 2)
        self.d0 = ConvTRMLP(16, self.out_chans, hidden_dim, 2, 2)

        self.d5 = nn.Linear(2, 32, bias=False)

    def iso_gauss_param(self, logits):
        feature_size = logits.size(1)
        assert feature_size % 2 == 0

        mu = logits[:, 0:int(feature_size / 2)]
        logvar = logits[:, int(feature_size / 2):]
        std = F.softplus(logvar) / 0.6931 + EPS_STD
        return mu, std

    def iso_gauss_sample(self, mu, std, noise=None):
        if noise is None:
            noise = float_type(self.config['cuda'])(std.size()).normal_()
        assert noise.shape == std.shape
        z = std.mul(noise).add_(mu)

        return z, noise

    def prec_weighted_com(self, mu1, std1, mu2, std2):
        prec1 = std1**(-2)
        prec2 = std2**(-2)
        std = (prec1 + prec2)**(-0.5)
        mu = (mu1 * prec1 + mu2 * prec2) / (prec1 + prec2)
        return mu, std

    def forward(self, x, condition):

        # upward

        h, l = self.u1(x, condition)
        um1, uv1 = self.iso_gauss_param(l)
        h, l = self.u2(h, condition)
        um2, uv2 = self.iso_gauss_param(l)
        h, l = self.u3(h, condition)
        um3, uv3 = self.iso_gauss_param(l)
        h, l = self.u4(h, condition)
        um4, uv4 = self.iso_gauss_param(l)
        h, l = self.u5(h, condition)
        um5, uv5 = self.iso_gauss_param(l)

        # downward
        l = self.d5(condition)
        l = l[:, :, None, None]
        pm5, pv5 = self.iso_gauss_param(l)
        qm5, qv5 = self.prec_weighted_com(um5, uv5, pm5, pv5)

        z5, n5 = self.iso_gauss_sample(qm5, qv5, None)
        _, l = self.d4(z5, condition)
        pm4, pv4 = self.iso_gauss_param(l)
        qm4, qv4 = self.prec_weighted_com(um4, uv4, pm4, pv4)

        z4, n4 = self.iso_gauss_sample(qm4, qv4, None)
        _, l = self.d3(z4, condition)
        pm3, pv3 = self.iso_gauss_param(l)
        qm3, qv3 = self.prec_weighted_com(um3, uv3, pm3, pv3)

        z3, n3 = self.iso_gauss_sample(qm3, qv3, None)
        _, l = self.d2(z3, condition)
        pm2, pv2 = self.iso_gauss_param(l)
        qm2, qv2 = self.prec_weighted_com(um2, uv2, pm2, pv2)

        z2, n2 = self.iso_gauss_sample(qm2, qv2, None)
        _, l = self.d1(z2, condition)
        pm1, pv1 = self.iso_gauss_param(l)
        qm1, qv1 = self.prec_weighted_com(um1, uv1, pm1, pv1)

        z1, n1 = self.iso_gauss_sample(qm1, qv1, None)
        _, logits = self.d0(z1, condition)

        params = {
            'q': [(qm1, qv1), (qm2, qv2), (qm3, qv3), (qm4, qv4), (qm5, qv5)],
            'p': [(pm1, pv1), (pm2, pv2), (pm3, pv3), (pm4, pv4), (pm5, pv5)],
            'z': [z1, z2, z3, z4, z5],
            'noise': [n1, n2, n3, n4, n5]
        }

        return logits, params

    def nelbo(self, x, logits, params, beta):
        elbo_nll = nll(x, logits, self.config['nll_type'])
        kl5 = kl_gaussian(*params['q'][4], *params['p'][4])
        kl4 = kl_gaussian(*params['q'][3], *params['p'][3])
        kl3 = kl_gaussian(*params['q'][2], *params['p'][2])
        kl2 = kl_gaussian(*params['q'][1], *params['p'][1])
        kl1 = kl_gaussian(*params['q'][0], *params['p'][0])
        kl = kl1 + kl2 + kl3 + kl4 + kl5
        nelbo = elbo_nll + beta * kl
        return nelbo, elbo_nll, kl1, kl2, kl3, kl4, kl5

    def generate_condition(self, size, continual_step):

        condition = float_type(self.config['cuda'])(size, 2).zero_()

        condition[:, continual_step] = 1
        return condition

    def generate_synthetic_samples(self,
                                   size,
                                   continual_step,
                                   noise_list=None):

        if noise_list is not None:
            n1, n2, n3, n4, n5 = noise_list
        else:
            n1 = n2 = n3 = n4 = n5 = None

        condition = self.generate_condition(size, continual_step)

        l = self.d5(condition)
        l = l[:, :, None, None]
        pm5, pv5 = self.iso_gauss_param(l)

        z5, n5 = self.iso_gauss_sample(pm5, pv5, n5)
        _, l = self.d4(z5, condition)
        pm4, pv4 = self.iso_gauss_param(l)

        z4, n4 = self.iso_gauss_sample(pm4, pv4, n4)
        _, l = self.d3(z4, condition)
        pm3, pv3 = self.iso_gauss_param(l)

        z3, n3 = self.iso_gauss_sample(pm3, pv3, n3)
        _, l = self.d2(z3, condition)
        pm2, pv2 = self.iso_gauss_param(l)

        z2, n2 = self.iso_gauss_sample(pm2, pv2, n2)
        _, l = self.d1(z2, condition)
        pm1, pv1 = self.iso_gauss_param(l)

        z1, n1 = self.iso_gauss_sample(pm1, pv1, n1)
        _, logits = self.d0(z1, condition)

        params = {
            'p': [(pm1, pv1), (pm2, pv2), (pm3, pv3), (pm4, pv4), (pm5, pv5)],
            'z': [z1, z2, z3, z4, z5],
            'noise': [n1, n2, n3, n4, n5]
        }

        return logits, nll_activation(logits, self.config['nll_type']), params

    def train(self, mode: bool = True):
        res = super().train(mode=mode)
        for name, m in self.named_modules():
            if 'bn' in name:
                m.train(True)
        return res


class ConvRMLP(nn.Module):
    def __init__(self,
                 in_dim=1,
                 out_dim=8,
                 hidden_dim=32,
                 stride=2,
                 num_block=1):
        super(ConvRMLP, self).__init__()
        blocks = []

        # downsample are treated at beggining
        blocks.append(ConvBasicBlock_Cond(in_dim, hidden_dim, stride))
        for _ in range(1, num_block):
            blocks.append(ConvBasicBlock(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*blocks)

        self.out = conv1x1(hidden_dim, out_dim)

    def forward(self, x, cond_class):

        h = self.net((x, cond_class))
        out = self.out(h)

        return h, out


class ConvRMLP_multiDS(nn.Module):
    def __init__(self,
                 in_dim=1,
                 out_dim=8,
                 hidden_dim=32,
                 stride=2,
                 num_block=2,
                 num_downsample=2):
        super(ConvRMLP_multiDS, self).__init__()
        blocks = []

        # downsample are treated at beggining
        blocks.append(ConvBasicBlock_Cond(in_dim, hidden_dim, stride))
        for _ in range(1, num_block):
            blocks.append(ConvBasicBlock(hidden_dim, hidden_dim))
        for index_ds in range(1, num_downsample):
            blocks.append(ConvBasicBlock(hidden_dim, hidden_dim, stride))
            for _ in range(1, num_block):
                blocks.append(ConvBasicBlock(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*blocks)

        self.out = conv1x1(hidden_dim, out_dim)

    def forward(self, x, cond_class):

        h = self.net((x, cond_class))
        out = self.out(h)

        return h, out


class ConvTRMLP(nn.Module):
    def __init__(self,
                 in_dim=4,
                 out_dim=1,
                 hidden_dim=32,
                 stride=2,
                 num_block=1):
        super(ConvTRMLP, self).__init__()
        blocks = []

        # upsample are treated at beggining
        blocks.append(ConvTBasicBlock_Cond(in_dim, hidden_dim, stride))
        for _ in range(1, num_block):
            blocks.append(ConvTBasicBlock(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*blocks)

        self.out = conv1x1(hidden_dim, out_dim)

    def forward(self, x, cond_class):

        h = self.net((x, cond_class))
        out = self.out(h)

        return h, out


class ConvTRMLP_multiDS(nn.Module):
    def __init__(self,
                 in_dim=4,
                 out_dim=1,
                 hidden_dim=32,
                 stride=2,
                 num_block=2,
                 num_upsample=2):
        super(ConvTRMLP_multiDS, self).__init__()
        blocks = []

        # upsample are treated at beggining
        blocks.append(ConvTBasicBlock_Cond(in_dim, hidden_dim, stride))
        for _ in range(1, num_block):
            blocks.append(ConvTBasicBlock(hidden_dim, hidden_dim))
        for index_ds in range(1, num_upsample):
            blocks.append(ConvTBasicBlock(hidden_dim, hidden_dim, stride))
            for _ in range(1, num_block):
                blocks.append(ConvTBasicBlock(hidden_dim, hidden_dim))

        self.net = nn.Sequential(*blocks)

        self.out = conv1x1(hidden_dim, out_dim)

    def forward(self, x, cond_class):

        h = self.net((x, cond_class))
        out = self.out(h)

        return h, out


class ConvBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ConvBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = conv1x1(inplanes, planes, stride)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

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
        out = self.relu(out)
        return out


class ConvTBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ConvTBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.unpool = None
        if stride != 1:
            self.unpool = nn.UpsamplingNearest2d(scale_factor=stride)
        self.upsample = None
        if stride != 1 or inplanes != planes:
            self.upsample = conv1x1(inplanes, planes)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        if self.unpool is not None:
            x = self.unpool(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out


class ConvBasicBlock_Cond(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ConvBasicBlock_Cond, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        inplanes = inplanes + 2
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = conv1x1(inplanes, planes, stride)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x_cond_class):
        x, cond_class = x_cond_class
        cond_class_ext = cond_class[:, :, None, None].repeat(
            (1, 1, x.shape[2], x.shape[3]))
        x = torch.cat([x, cond_class_ext], dim=1)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ConvTBasicBlock_Cond(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ConvTBasicBlock_Cond, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        inplanes = inplanes + 2
        self.unpool = None
        if stride != 1:
            self.unpool = nn.UpsamplingNearest2d(scale_factor=stride)
        self.upsample = None
        if stride != 1 or inplanes != planes:
            self.upsample = conv1x1(inplanes, planes)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x_cond_class):
        x, cond_class = x_cond_class
        cond_class_ext = cond_class[:, :, None, None].repeat(
            (1, 1, x.shape[2], x.shape[3]))
        x = torch.cat([x, cond_class_ext], dim=1)
        if self.unpool is not None:
            x = self.unpool(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)
