import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import create_conv2d, drop_path, make_divisible
from .layers.activations import sigmoid

__all__ = [
    'SqueezeExcite', 'ConvBnAct', 'DepthwiseSeparableConv', 'InvertedResidual', 'CondConvResidual', 'EdgeResidual']


class SqueezeExcite(nn.Module):
    def __init__(
            self, in_chs, se_ratio=0.25, act_layer=nn.ReLU, gate_fn=sigmoid,
            block_in_chs=None, reduce_from_block=True, force_act_layer=None, divisor=1):
        super(SqueezeExcite, self).__init__()
        reduced_chs = (block_in_chs or in_chs) if reduce_from_block else in_chs
        reduced_chs = make_divisible(reduced_chs * se_ratio, divisor)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)


class ConvBnAct(nn.Module):
    def __init__(
            self, in_chs, out_chs, kernel_size, stride=1, dilation=1, pad_type='',
            skip=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop_path_rate=0.):
        super(ConvBnAct, self).__init__()
        self.has_residual = skip and stride == 1 and in_chs == out_chs
        self.drop_path_rate = drop_path_rate
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(out_chs)
        self.act1 = act_layer(inplace=True)

    def feature_info(self, location):
        if location == 'expansion':  
            info = dict(module='act1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels)
        return info

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='',
            noskip=False, pw_kernel_size=1, pw_act=False, se_ratio=0.,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_layer=None, drop_path_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        has_se = se_layer is not None and se_ratio > 0.
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  
        self.drop_path_rate = drop_path_rate

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, depthwise=True)
        self.bn1 = norm_layer(in_chs)
        self.act1 = act_layer(inplace=True)

        
        self.se = se_layer(in_chs, se_ratio=se_ratio, act_layer=act_layer) if has_se else nn.Identity()

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs)
        self.act2 = act_layer(inplace=True) if self.has_pw_act else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  
            info = dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  
            info = dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class InvertedResidual(nn.Module):
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, se_ratio=0.,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_layer=None, conv_kwargs=None, drop_path_rate=0.):
        super(InvertedResidual, self).__init__()
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_layer is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_layer(mid_chs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, depthwise=True, **conv_kwargs)
        self.bn2 = norm_layer(mid_chs)
        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation
        self.se = se_layer(
            mid_chs, se_ratio=se_ratio, act_layer=act_layer, block_in_chs=in_chs) if has_se else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_layer(out_chs)

    def feature_info(self, location):
        if location == 'expansion': 
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else: 
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut

        return x


class CondConvResidual(InvertedResidual):
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, se_ratio=0.,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_layer=None, num_experts=0, drop_path_rate=0.):

        self.num_experts = num_experts
        conv_kwargs = dict(num_experts=self.num_experts)

        super(CondConvResidual, self).__init__(
            in_chs, out_chs, dw_kernel_size=dw_kernel_size, stride=stride, dilation=dilation, pad_type=pad_type,
            act_layer=act_layer, noskip=noskip, exp_ratio=exp_ratio, exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size, se_ratio=se_ratio, se_layer=se_layer,
            norm_layer=norm_layer, conv_kwargs=conv_kwargs, drop_path_rate=drop_path_rate)

        self.routing_fn = nn.Linear(in_chs, self.num_experts)

    def forward(self, x):
        shortcut = x

        # CondConv routing
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))

        # Point-wise expansion
        x = self.conv_pw(x, routing_weights)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x, routing_weights)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x, routing_weights)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class EdgeResidual(nn.Module):

    def __init__(
            self, in_chs, out_chs, exp_kernel_size=3, stride=1, dilation=1, pad_type='',
            force_in_chs=0, noskip=False, exp_ratio=1.0, pw_kernel_size=1, se_ratio=0.,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_layer=None, drop_path_rate=0.):
        super(EdgeResidual, self).__init__()
        if force_in_chs > 0:
            mid_chs = make_divisible(force_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_layer is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        # Expansion convolution
        self.conv_exp = create_conv2d(
            in_chs, mid_chs, exp_kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(mid_chs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        self.se = SqueezeExcite(
            mid_chs, se_ratio=se_ratio, act_layer=act_layer, block_in_chs=in_chs) if has_se else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs)

    def feature_info(self, location):
        if location == 'expansion':  
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else: 
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def forward(self, x):
        shortcut = x

        # Expansion convolution
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut

        return x
