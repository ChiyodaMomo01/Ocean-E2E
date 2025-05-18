from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile

class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
            
        elif self.type == 'conv1x1-Krisch-1':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 5.0
                self.mask[i, 0, 0, 1] = 5.0
                self.mask[i, 0, 0, 2] = 5.0
                self.mask[i, 0, 1, 0] = -3.0
                self.mask[i, 0, 1, 1] = 0.0
                self.mask[i, 0, 1, 2] = -3.0
                self.mask[i, 0, 2, 0] = -3.0
                self.mask[i, 0, 2, 1] = -3.0
                self.mask[i, 0, 2, 2] = -3.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            
        elif self.type == 'conv1x1-Krisch-2':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = -3.0
                self.mask[i, 0, 0, 1] = 5.0
                self.mask[i, 0, 0, 2] = 5.0
                self.mask[i, 0, 1, 0] = -3.0
                self.mask[i, 0, 1, 1] = 0.0
                self.mask[i, 0, 1, 2] = 5.0
                self.mask[i, 0, 2, 0] = -3.0
                self.mask[i, 0, 2, 1] = -3.0
                self.mask[i, 0, 2, 2] = -3.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            
        elif self.type == 'conv1x1-Krisch-3':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = -3.0
                self.mask[i, 0, 0, 1] = -3.0
                self.mask[i, 0, 0, 2] = 5.0
                self.mask[i, 0, 1, 0] = -3.0
                self.mask[i, 0, 1, 1] = 0.0
                self.mask[i, 0, 1, 2] = 5.0
                self.mask[i, 0, 2, 0] = -3.0
                self.mask[i, 0, 2, 1] = -3.0
                self.mask[i, 0, 2, 2] = 5.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            
        elif self.type == 'conv1x1-Krisch-4':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = -3.0
                self.mask[i, 0, 0, 1] = -3.0
                self.mask[i, 0, 0, 2] = -3.0
                self.mask[i, 0, 1, 0] = -3.0
                self.mask[i, 0, 1, 1] = 0.0
                self.mask[i, 0, 1, 2] = 5.0
                self.mask[i, 0, 2, 0] = -3.0
                self.mask[i, 0, 2, 1] = 5.0
                self.mask[i, 0, 2, 2] = 5.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            
        elif self.type == 'conv1x1-Krisch-5':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = -3.0
                self.mask[i, 0, 0, 1] = -3.0
                self.mask[i, 0, 0, 2] = -3.0
                self.mask[i, 0, 1, 0] = -3.0
                self.mask[i, 0, 1, 1] = 0.0
                self.mask[i, 0, 1, 2] = -3.0
                self.mask[i, 0, 2, 0] = 5.0
                self.mask[i, 0, 2, 1] = 5.0
                self.mask[i, 0, 2, 2] = 5.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            
        elif self.type == 'conv1x1-Krisch-6':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = -3.0
                self.mask[i, 0, 0, 1] = -3.0
                self.mask[i, 0, 0, 2] = -3.0
                self.mask[i, 0, 1, 0] = 5.0
                self.mask[i, 0, 1, 1] = 0.0
                self.mask[i, 0, 1, 2] = -3.0
                self.mask[i, 0, 2, 0] = 5.0
                self.mask[i, 0, 2, 1] = 5.0
                self.mask[i, 0, 2, 2] = -3.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            
        elif self.type == 'conv1x1-Krisch-7':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 5.0
                self.mask[i, 0, 0, 1] = -3.0
                self.mask[i, 0, 0, 2] = -3.0
                self.mask[i, 0, 1, 0] = 5.0
                self.mask[i, 0, 1, 1] = 0.0
                self.mask[i, 0, 1, 2] = -3.0
                self.mask[i, 0, 2, 0] = 5.0
                self.mask[i, 0, 2, 1] = -3.0
                self.mask[i, 0, 2, 2] = -3.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
            
        elif self.type == 'conv1x1-Krisch-8':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 5.0
                self.mask[i, 0, 0, 1] = 5.0
                self.mask[i, 0, 0, 2] = -3.0
                self.mask[i, 0, 1, 0] = 5.0
                self.mask[i, 0, 1, 1] = 0.0
                self.mask[i, 0, 1, 2] = -3.0
                self.mask[i, 0, 2, 0] = -3.0
                self.mask[i, 0, 2, 1] = -3.0
                self.mask[i, 0, 2, 2] = -3.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
  

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1
    
    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3':
            # re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1,) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1,) + b1
        return RK, RB



class ECB(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu'):
        super(ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
        self.conv1x1_k1 = SeqConv3x3('conv1x1-Krisch-1', self.inp_planes, self.out_planes, -1)
        self.conv1x1_k2 = SeqConv3x3('conv1x1-Krisch-2', self.inp_planes, self.out_planes, -1)
        self.conv1x1_k3 = SeqConv3x3('conv1x1-Krisch-3', self.inp_planes, self.out_planes, -1)
        self.conv1x1_k4 = SeqConv3x3('conv1x1-Krisch-4', self.inp_planes, self.out_planes, -1)
        self.conv1x1_k5 = SeqConv3x3('conv1x1-Krisch-5', self.inp_planes, self.out_planes, -1)
        self.conv1x1_k6 = SeqConv3x3('conv1x1-Krisch-6', self.inp_planes, self.out_planes, -1)
        self.conv1x1_k7 = SeqConv3x3('conv1x1-Krisch-7', self.inp_planes, self.out_planes, -1)
        self.conv1x1_k8 = SeqConv3x3('conv1x1-Krisch-8', self.inp_planes, self.out_planes, -1)
        

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.training:
            y = self.conv3x3(x)     + \
                self.conv1x1_3x3(x) + \
                self.conv1x1_k1(x) + \
                self.conv1x1_k2(x) + \
                self.conv1x1_k3(x) + \
                self.conv1x1_k4(x) + \
                self.conv1x1_k5(x) + \
                self.conv1x1_k6(x) + \
                self.conv1x1_k7(x) + \
                self.conv1x1_k8(x)

        else:

            RK, RB = self.rep_params()
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1) 

        if self.act_type != 'linear':
            y = self.act(y)
        return y

    def rep_params(self):
     
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_k1.rep_params()
        K3, B3 = self.conv1x1_k2.rep_params()
        K4, B4 = self.conv1x1_k3.rep_params()
        K5, B5 = self.conv1x1_k4.rep_params()
        K6, B6 = self.conv1x1_k5.rep_params()
        K7, B7 = self.conv1x1_k6.rep_params()
        K8, B8 = self.conv1x1_k7.rep_params()
        K9, B9 = self.conv1x1_k8.rep_params()  
            
        RK, RB = (K0+K1+K2+K3+K4+K5+K6+K7+K8+K9), (B0+B1+B2+B3+B4+B5+B6+B7+B8+B9)

        return RK, RB

class KRM(nn.Module):    
	def __init__(self,channel):                                
		super(KRM,self).__init__()

		self.conv_in = nn.Conv2d(channel,channel//4,kernel_size=1,stride=1,padding=0,bias=False)
		self.ecbb_t1 = ECB(channel//4, channel//4, depth_multiplier=2.0)
		self.conv_out = nn.Conv2d(channel//4,channel,kernel_size=1,stride=1,padding=0,bias=False)
        
	def forward(self,x):
        
		x_t = self.conv_out(self.ecbb_t1(self.conv_in(x)))
		
		return	x_t

class Block(nn.Module):
    def __init__(self,channel):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel,kernel_size=3,padding=1,bias=True)
        self.act = nn.PReLU(channel)
        self.conv2= nn.Conv2d(channel,channel,kernel_size=3,padding=1,bias=True)
        self.conv3= nn.Conv2d(channel,channel,kernel_size=3,padding=1,bias=True)
        self.conv4= nn.Conv2d(channel,channel,kernel_size=3,padding=1,bias=True)
        self.cbam = CBAMLayer(channel)
        
        self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)


    def forward(self, x):
             
        res1 = self.act(self.norm(self.conv1(x)))
        res2 = self.act(self.norm(self.conv2(res1)))  
        cbam = self.cbam(res2)
        res3 = self.act(self.norm(self.conv3(cbam)))          
        res4 = self.act(self.norm(self.conv4(res3)) + x)
        
        return res4

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=8, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(

            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        return x
        
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x


class BottleNect(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 63
        pad = ker // 2
        self.in_conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.ReLU()

        ### sca ###
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        ### fca ###
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)

        ### fca ###
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2,-1), norm='backward')
        x_fca = torch.abs(x_fca)

        ### fca ###
        ### sca ###
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        ### sca ###
        x_sca = self.fgm(x_sca)

        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)

class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # res = x.clone()
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


class Assimilation(nn.Module): 
	def __init__(self,
                 # args,
                 in_c = 3,
                 out_c = 1,
                 channel = 16):
		super(Assimilation,self).__init__()

		self.Haze_E = Encoder(channel)#channel = 16

		self.Share  = ShareNet(channel)#channel = 16

		self.Haze_D = Decoder(channel)#channel = 16

		self.Haze_in1 = nn.Conv2d(in_c,channel,kernel_size=1,stride=1,padding=0,bias=False)#3 16 对由多个输入平面组成的输入信号进行二维卷积
		# self.Haze_in2 = nn.Conv2d(2,channel,kernel_size=1,stride=1,padding=0,bias=False)#3 16 对由多个输入平面组成的输入信号进行二维卷积  
		# self.Haze_in3 = nn.Conv2d(3,channel,kernel_size=1,stride=1,padding=0,bias=False)#3 16 对由多个输入平面组成的输入信号进行二维卷积
		# self.Haze_in4 = nn.Conv2d(4,channel,kernel_size=1,stride=1,padding=0,bias=False)#3 16 对由多个输入平面组成的输入信号进行二维卷积                             
		self.Haze_out = nn.Conv2d(channel,out_c,kernel_size=1,stride=1,padding=0,bias=False)#16 3

	def forward(self,x):
		# B, H, W, T = x.shape
		# x = x.permute(0, 3, 1, 2)
        
		x_in = self.Haze_in1(x)#3 16
		L,M,S,SS = self.Haze_E(x_in)
		Share = self.Share(SS)
		x_out = self.Haze_D(Share,SS,S,M,L)
		out = self.Haze_out(x_out) #B T, H, W

		# out = out.permute(0, 2, 3, 1)

		return out

class Encoder(nn.Module):
	def __init__(self,channel):
		super(Encoder,self).__init__()    

		self.el = ResidualBlock(channel)#16
		self.em = ResidualBlock(channel*2)#32
		self.es = ResidualBlock(channel*4)#64
		self.ess = ResidualBlock(channel*8)#128
		self.esss = ResidualBlock(channel*16)#256
        
		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.conv_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)#16 32
		self.conv_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)#32 64
		self.conv_estess = nn.Conv2d(4*channel,8*channel,kernel_size=1,stride=1,padding=0,bias=False)#64 128
		self.conv_esstesss = nn.Conv2d(8*channel,16*channel,kernel_size=1,stride=1,padding=0,bias=False)#128 256
        
	def forward(self,x):
        
		elout = self.el(x)#16
		x_emin = self.conv_eltem(self.maxpool(elout))#32
		emout = self.em(x_emin)
		x_esin = self.conv_emtes(self.maxpool(emout))        
		esout = self.es(x_esin)
		x_esin = self.conv_estess(self.maxpool(esout))        
		essout = self.ess(x_esin)#128

		return elout,emout,esout,essout#,esssout

class ShareNet(nn.Module):
	def __init__(self,channel):
		super(ShareNet,self).__init__()    

		self.s1 = MTRB(channel*8)#128
		self.s2 = MTRB(channel*8)#128

	def forward(self,x):
		share1 = self.s1(x)
		share2 = self.s2(share1+x)

		return share2

class Decoder(nn.Module):
	def __init__(self,channel):
		super(Decoder,self).__init__()    

		self.dss = ResidualBlock(channel*8)#128
		self.ds = ResidualBlock(channel*4)#64
		self.dm = ResidualBlock(channel*2)#32
		self.dl = ResidualBlock(channel)#16
        
		self.conv_dssstdss = nn.Conv2d(16*channel,8*channel,kernel_size=1,stride=1,padding=0,bias=False)#256 128
		self.conv_dsstds = nn.Conv2d(8*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)#128 64
		self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)#64 32
		self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)#32 16
        
	def _upsample(self,x,y):
		_,_,H0,W0 = y.size()
		return F.interpolate(x,size=(H0,W0),mode='bilinear')
    
	def forward(self,x,ss,s,m,l):

		dssout = self.dss(x+ss)
		x_dsin = self.conv_dsstds(self._upsample(dssout, s))        
		dsout = self.ds(x_dsin+s)
		x_dmin = self.conv_dstdm(self._upsample(dsout, m))
		dmout = self.dm(x_dmin+m)
		x_dlin = self.conv_dmtdl(self._upsample(dmout, l))
		dlout = self.dl(x_dlin+l)
        
		return dlout


class MTRB(nn.Module):# Edge-oriented Residual Convolution Block
	def __init__(self,channel,norm=False):                                
		super(MTRB,self).__init__()

		self.conv_1_1 = nn.Conv2d(channel*1,  channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_2_1 = nn.Conv2d(channel*1,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2_2 = nn.Conv2d(channel*2,channel,kernel_size=3,stride=1,padding=1,bias=False)

        
		self.conv_3_1 = nn.Conv2d(channel*1,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_3_2 = nn.Conv2d(channel*4,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_3_3 = nn.Conv2d(channel*1,channel,kernel_size=3,stride=1,padding=1,bias=False)

 
		self.conv_4_1 = nn.Conv2d(channel*1,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_4_2 = nn.Conv2d(channel*3,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_4_3 = nn.Conv2d(channel*4,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_4_4 = nn.Conv2d(channel*1,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_5_1 = nn.Conv2d(channel*2,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_5_2 = nn.Conv2d(channel*4,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_5_3 = nn.Conv2d(channel*2,channel,kernel_size=3,stride=1,padding=1,bias=False)


		self.conv_6_1 = nn.Conv2d(channel*2,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_6_2 = nn.Conv2d(channel*3,channel,kernel_size=3,stride=1,padding=1,bias=False)

		self.conv_7_1 = nn.Conv2d(channel*2,channel,kernel_size=3,stride=1,padding=1,bias=False)

        
		self.conv_out = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.act = nn.PReLU(channel)
		self.sig = nn.Sigmoid()

		self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#
   
	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')


	def forward(self,x):
        
		x_1_1 = self.act(self.norm(self.conv_1_1(x)))

		x_2_1 = self.act(self.norm(self.conv_2_1(x_1_1)))
		x_2_2 = self.act(self.norm(self.conv_2_2(torch.cat((x_2_1 , x_1_1),1))))

        
		x_3_1 = self.act(self.norm(self.conv_3_1(x_2_1)))
		x_3_3 = self.act(self.norm(self.conv_3_3(x_2_2)))
		x_3_2 = self.act(self.norm(self.conv_3_2(torch.cat((x_3_1 , x_3_3 , x_2_1 , x_2_2),1))))
        
        
		x_4_1 = self.act(self.norm(self.conv_4_1(x_3_1)))
		x_4_4 = self.act(self.norm(self.conv_4_4(x_3_3)))
		x_4_2 = self.act(self.norm(self.conv_4_2(torch.cat((x_4_1 , x_3_1 , x_3_2),1))))
		x_4_3 = self.act(self.norm(self.conv_4_3(torch.cat((x_4_2 , x_4_4 , x_3_2 , x_3_3),1))))
 
    
		x_5_1 = self.act(self.norm(self.conv_5_1(torch.cat((x_4_1 , x_4_2),1))))
		x_5_3 = self.act(self.norm(self.conv_5_3(torch.cat((x_4_3 , x_4_4),1))))
		x_5_2 = self.act(self.norm(self.conv_5_2(torch.cat((x_5_1 , x_5_3 , x_4_2 , x_4_3),1))))
     
        
		x_6_1 = self.act(self.norm(self.conv_6_1(torch.cat((x_5_1 , x_5_2),1))))
		x_6_2 = self.act(self.norm(self.conv_6_2(torch.cat((x_6_1 , x_5_2 , x_5_3),1))))


		x_7_1 = self.act(self.norm(self.conv_7_1(torch.cat((x_6_1 , x_6_2),1))))

        
		x_out = self.act(self.norm(self.conv_out(x_7_1)) + x)


		return	x_out


class ResidualBlock(nn.Module):  # Edge-oriented Residual Convolution Block 面向边缘的残差网络块 解决梯度消失的问题
	def __init__(self, channel, norm=False):
		super(ResidualBlock, self).__init__()

		self.conv_1_1 = BottleNect(channel)  # 16 16
		self.conv_2_1 = BottleNect(channel)  # 16 16
		self.conv_3_1 = KRM(channel)
		self.conv_4_1 = Block(channel) 
		# self.conv_out = DeformConv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)#16 16
		self.act = nn.PReLU(channel)  # 16
		self.norm = nn.GroupNorm(num_channels=channel, num_groups=1)  # nn.InstanceNorm2d(channel)#

	def _upsample(self, x, y):  # 上采样->放大图片
		_, _, H, W = y.size()
		return F.upsample(x, size=(H, W), mode='bilinear')

	def forward(self, x):
		x_1 = self.act(self.norm(self.conv_1_1(x)))
		x_2 = self.act(self.norm(self.conv_2_1(x_1))) + x
		x_3 =  self.act(self.norm(self.conv_3_1(x_2))) + x
		x_4 =  self.act(self.norm(self.conv_4_1(x_3))) + x
		# x_out = self.act(self.norm(self.conv_out(x_2)) + x)

		return x_4


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    net = Assimilation(in_c=7, out_c=2, channel = 32).to(device)

    input = torch.randn(1, 7, 360, 720).to(device)
    output = net(input)

    print(output.shape)
