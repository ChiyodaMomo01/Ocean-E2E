

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


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



class ERANet(nn.Module):
	def __init__(self,channel=32):
		super(ERANet,self).__init__()

		self.conv_in = nn.Conv2d(3,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_out = nn.Conv2d(channel,3,kernel_size=3,stride=1,padding=1,bias=False)

		self.repb1 = Block(channel)
		self.repb2 = Block(channel)
		self.repb3 = Block(channel)
		self.repb4 = Block(channel)
		self.repb5 = Block(channel)

		self.krm = KRM(channel)
        

	def forward(self,x):
        
		x_in = self.conv_in(x)
           
		x_1 = self.repb1(self.krm(x_in) + x_in)
		x_2 = self.repb2(self.krm(x_1) + x_1)        
		x_3 = self.repb3(self.krm(x_2) + x_2)
		x_4 = self.repb4(self.krm(x_3) + x_3)
		x_5 = self.repb5(self.krm(x_4) + x_4)
        
		x_out = self.conv_out(x_5)
        
		return x_out


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
    
    
class KRM(nn.Module):    
	def __init__(self,channel):                                
		super(KRM,self).__init__()

		self.conv_in = nn.Conv2d(channel,channel//4,kernel_size=1,stride=1,padding=0,bias=False)
		self.ecbb_t1 = ECB(channel//4, channel//4, depth_multiplier=2.0)
		self.conv_out = nn.Conv2d(channel//4,channel,kernel_size=1,stride=1,padding=0,bias=False)
        
	def forward(self,x):
        
		x_t = self.conv_out(self.ecbb_t1(self.conv_in(x)))
		
		return	x_t
    

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


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    net = ERANet().to(device)

    input = torch.randn(1, 3, 240, 240).to(device)
    output = net(input)

    print(input.shape)
    print(output.shape)
