# Codes for section: Results on Navier Stocks Equation (2D)
# https://github.com/ashiq24/UNO/tree/main
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import matplotlib.pyplot as plt
# from UNO_utils import *
import operator
from functools import reduce
from functools import partial

from timeit import default_timer

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1,modes1 = None):
        super(SpectralConv1d_Uno, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        Ratio of grid size of the input and the output implecitely 
        set the expansion or contraction farctor along each dimension of the domain.
        modes1 = Number of fourier modes to consider for the integral operator.
                Number of modes must be compatibale with the input grid size 
                and desired output grid size.
                i.e., modes1 <= min( dim1/2, input_dim1/2). 
                Here "input_dim1" is the grid size along x axis (or first dimension) of the input domain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension
        """
        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1 #output dimensions
        if modes1 is not None:
            self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        else:
            self.modes1 = dim1//2

        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, dim1 = None):
        """
        input shape = (batch, in_codim, input_dim1)
        output shape = (batch, out_codim, dim1)
        """
        if dim1 is not None:
            self.dim1 = dim1
        batchsize = x.shape[0]

        x_ft = torch.fft.rfft(x, norm = 'forward')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=self.dim1, norm = 'forward')
        return x

class pointwise_op_1D(nn.Module):
    """
    All variables are consistent with the SpectralConv1d_Uno class.
    """
    def __init__(self, in_codim, out_codim,dim1):
        super(pointwise_op_1D,self).__init__()
        self.conv = nn.Conv1d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)

    def forward(self,x, dim1 = None):
        if dim1 is None:
            dim1 = self.dim1
        x_out = self.conv(x)

        #x_out = torch.nn.functional.interpolate(x_out, size = dim1,mode = 'linear',align_corners=True, antialias= True)
        x_out = torch.nn.functional.interpolate(x_out, size=(dim1, dim2), mode='bilinear', align_corners=True)

        return x_out

class OperatorBlock_1D(nn.Module):
    """
    Normalize = if true performs InstanceNorm1d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv1d_Uno class.
    """
    def __init__(self, in_codim, out_codim,dim1,modes1, Normalize = True,Non_Lin = True):
        super(OperatorBlock_1D,self).__init__()
        self.conv = SpectralConv1d_Uno(in_codim, out_codim, dim1,modes1)
        self.w = pointwise_op_1D(in_codim, out_codim, dim1)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm1d(int(out_codim),affine=True)


    def forward(self,x, dim1 = None):
        """
        input shape = (batch, in_codim, input_dim1)
        output shape = (batch, out_codim, dim1)
        """
        x1_out = self.conv(x,dim1)
        x2_out = self.w(x,dim1)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out


class SpectralConv2d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim, dim1, dim2,modes1 = None, modes2 = None):
        super(SpectralConv2d_Uno, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        dim2 = Default output grid size along y ( or 2nd dimension of output domain)
        Ratio of grid size of the input and the output implecitely 
        set the expansion or contraction farctor along each dimension.
        modes1, modes2 = Number of fourier modes to consider for the ontegral operator
                        Number of modes must be compatibale with the input grid size 
                        and desired output grid size.
                        i.e., modes1 <= min( dim1/2, input_dim1/2). 
                        Here "input_dim1" is the grid size along x axis (or first dimension) of the input domain.
                        Other modes also the have same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension
        """

        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1 
        self.dim2 = dim2
        if modes1 is not None:
            self.modes1 = modes1 
            self.modes2 = modes2
        else:
            self.modes1 = dim1//2-1 
            self.modes2 = dim2//2 
        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * (torch.randn(in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.randn(in_codim, out_codim, self.modes1, self.modes2, dtype=torch.cfloat)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):

        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, dim1 = None,dim2 = None):
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, norm = 'forward')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1, self.dim2//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(self.dim1, self.dim2),norm = 'forward')
        return x

class pointwise_op_2D(nn.Module):
    """ 
    dim1 = Default output grid size along x (or 1st dimension) 
    dim2 = Default output grid size along y ( or 2nd dimension)
    in_codim = Input co-domian dimension
    out_codim = output co-domain dimension
    """
    def __init__(self, in_codim, out_codim,dim1, dim2):
        super(pointwise_op_2D,self).__init__()
        self.conv = nn.Conv2d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)

    def forward(self,x, dim1 = None, dim2 = None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
        x_out = self.conv(x)

        #ft = torch.fft.rfft2(x_out)
        #ft_u = torch.zeros_like(ft)
        #ft_u[:dim1//2-1,:dim2//2-1] = ft[:dim1//2-1,:dim2//2-1]
        #ft_u[-(dim1//2-1):,:dim2//2-1] = ft[-(dim1//2-1):,:dim2//2-1]
        #x_out = torch.fft.irfft2(ft_u)
        
        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2),mode = 'bicubic',align_corners=True)
        return x_out

class OperatorBlock_2D(nn.Module):
    """
    Normalize = if true performs InstanceNorm2d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv2d_Uno class.
    """
    def __init__(self, in_codim, out_codim,dim1, dim2,modes1,modes2, Normalize = False, Non_Lin = True):
        super(OperatorBlock_2D,self).__init__()
        self.conv = SpectralConv2d_Uno(in_codim, out_codim, dim1,dim2,modes1,modes2)
        self.w = pointwise_op_2D(in_codim, out_codim, dim1,dim2)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm2d(int(out_codim),affine=True)


    def forward(self,x, dim1 = None, dim2 = None):
        """
        input shape = (batch, in_codim, input_dim1,input_dim2)
        output shape = (batch, out_codim, dim1,dim2)
        """
        x1_out = self.conv(x,dim1,dim2)
        x2_out = self.w(x,dim1,dim2)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out

class SpectralConv3d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim,dim1,dim2,dim3, modes1=None, modes2=None, modes3=None):
        super(SpectralConv3d_Uno, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        dim2 = Default output grid size along y ( or 2nd dimension of output domain)
        dim3 = Default output grid size along time t ( or 3rd dimension of output domain)
        Ratio of grid size of the input and output grid size (dim1,dim2,dim3) implecitely 
        set the expansion or contraction farctor along each dimension.
        modes1, modes2, modes3 = Number of fourier modes to consider for the ontegral operator
                                Number of modes must be compatibale with the input grid size 
                                and desired output grid size.
                                i.e., modes1 <= min( dim1/2, input_dim1/2).
                                      modes2 <= min( dim2/2, input_dim2/2)
                                Here input_dim1, input_dim2 are respectively the grid size along 
                                x axis and y axis (or first dimension and second dimension) of the input domain.
                                Other modes also have the same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension   
        """
        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        if modes1 is not None:
            self.modes1 = modes1 
            self.modes2 = modes2
            self.modes3 = modes3 
        else:
            self.modes1 = dim1 
            self.modes2 = dim2
            self.modes3 = dim3//2+1

        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):

        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, dim1 = None,dim2=None,dim3=None):
        """
        dim1,dim2,dim3 are the output grid size along (x,y,t)
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
            self.dim3 = dim3   

        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1], norm = 'forward')

        out_ft = torch.zeros(batchsize, self.out_channels, self.dim1, self.dim2, self.dim3//2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(self.dim1, self.dim2, self.dim3), norm = 'forward')
        return x

class pointwise_op_3D(nn.Module):
    def __init__(self, in_codim, out_codim,dim1, dim2,dim3):
        super(pointwise_op_3D,self).__init__()
        self.conv = nn.Conv3d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)

    def forward(self,x, dim1 = None, dim2 = None, dim3 = None):
        """
        dim1,dim2,dim3 are the output dimensions (x,y,t)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
            dim3 = self.dim3
        x_out = self.conv(x)

        ft = torch.fft.rfftn(x_out,dim=[-3,-2,-1])
        ft_u = torch.zeros_like(ft)
        ft_u[:, :, :(dim1//2), :(dim2//2), :(dim3//2)] = ft[:, :, :(dim1//2), :(dim2//2), :(dim3//2)]
        ft_u[:, :, -(dim1//2):, :(dim2//2), :(dim3//2)] = ft[:, :, -(dim1//2):, :(dim2//2), :(dim3//2)]
        ft_u[:, :, :(dim1//2), -(dim2//2):, :(dim3//2)] = ft[:, :, :(dim1//2), -(dim2//2):, :(dim3//2)]
        ft_u[:, :, -(dim1//2):, -(dim2//2):, :(dim3//2)] = ft[:, :, -(dim1//2):, -(dim2//2):, :(dim3//2)]
        
        x_out = torch.fft.irfftn(ft_u, s=(dim1, dim2, dim3))

        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2,dim3),mode = 'trilinear')
        return x_out

class OperatorBlock_3D(nn.Module):
    """
    Normalize = if true performs InstanceNorm3d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv3d_Uno class.
    """
    def __init__(self, in_codim, out_codim,dim1, dim2,dim3,modes1,modes2,modes3, Normalize = False,Non_Lin = True):
        super(OperatorBlock_3D,self).__init__()
        self.conv = SpectralConv3d_Uno(in_codim, out_codim, dim1,dim2,dim3,modes1,modes2,modes3)
        self.w = pointwise_op_3D(in_codim, out_codim, dim1,dim2,dim3)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm3d(int(out_codim),affine=True)


    def forward(self,x, dim1 = None, dim2 = None, dim3 = None):
        """
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """
        x1_out = self.conv(x,dim1,dim2,dim3)
        x2_out = self.w(x,dim1,dim2,dim3)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out


torch.manual_seed(0)
np.random.seed(0)


# UNO model more aggressive domian contraction and expansion (factor of 1/2)
class UNO_P(nn.Module):
    def __init__(self,in_width, width,pad = 0, factor = 1):
        super(UNO_P, self).__init__()

        """
        The overall network. It contains 7 integral operator.
        1. Lift the input to the desire channel dimension by  self.fc, self.fc0 .
        2. 7 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        input: the solution of the first 10 timesteps (u(1), ..., u(10)).
        input shape: (batchsize, x=S, y=S, t=10)
        output: the solution of the next timesteps
        output shape: (batchsize, x=S, y=S, t=1)
        Here SxS is the spatial resolution
        in_width = 12 (10 input time steps + (x,y) location)
        with = uplifting dimension
        pad = padding the domian for non-periodic input
        factor = factor for scaling up/down the co-domain dimension at each integral operator
        """
        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_2D(self.width, 2*factor*self.width,32, 32, 14, 14)

        self.L1 = OperatorBlock_2D(2*factor*self.width, 4*factor*self.width, 16, 16, 6,6)

        self.L2 = OperatorBlock_2D(4*factor*self.width, 8*factor*self.width, 8, 8,3,3)
        
        self.L3 = OperatorBlock_2D(8*factor*self.width, 8*factor*self.width, 8, 8,3,3)
        
        self.L4 = OperatorBlock_2D(8*factor*self.width, 4*factor*self.width, 16, 16,3,3)

        self.L5 = OperatorBlock_2D(8*factor*self.width, 2*factor*self.width, 32, 32,6,6)

        self.L6 = OperatorBlock_2D(4*factor*self.width, self.width, 64, 64,14,14) # will be reshaped


        self.fc1 = nn.Linear(2*self.width, 3*self.width)
        self.fc2 = nn.Linear(3*self.width + self.width//2, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        x_fc0 = F.pad(x_fc0, [self.padding,self.padding, self.padding,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.L0(x_fc0,D1//2,D2//2)

        x_c1 = self.L1(x_c0,D1//4,D2//4)


        x_c2 = self.L2(x_c1,D1//8,D2//8)

        
        x_c3 = self.L3(x_c2,D1//8,D2//8)


        x_c4 = self.L4(x_c3 ,D1//4,D2//4)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)

        x_c5 = self.L5(x_c4 ,D1//2,D2//2)
        x_c5 = torch.cat([x_c5, x_c0], dim=1)

        x_c6 = self.L6(x_c5,D1,D2)
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        if self.padding!=0:
            x_c6 = x_c6[..., self.padding:-self.padding, self.padding:-self.padding]

        x_c6 = x_c6.permute(0, 2, 3, 1)

        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)

        x_fc1 = torch.cat([x_fc1, x_fc], dim=3)
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 2*np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 2*np.pi, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((torch.sin(gridx),torch.sin(gridy),torch.cos(gridx),torch.cos(gridy)), dim=-1).to(device)

#####
# UNO model 
# it has less aggressive scaling factors for domains and co-domains.
# ####    
class UNO(nn.Module):
    def __init__(self,in_width, width,pad = 0, factor = 3/4):
        super(UNO, self).__init__()


        self.in_width = in_width # input channel
        self.width = width 
        self.factor = factor
        self.padding = pad  

        self.fc = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_2D(self.width, 2*factor*self.width,48, 48, 22, 22)

        self.L1 = OperatorBlock_2D(2*factor*self.width, 4*factor*self.width, 32, 32, 14,14)

        self.L2 = OperatorBlock_2D(4*factor*self.width, 8*factor*self.width, 16, 16,6,6)
        
        self.L3 = OperatorBlock_2D(8*factor*self.width, 8*factor*self.width, 16, 16,6,6)
        
        self.L4 = OperatorBlock_2D(8*factor*self.width, 4*factor*self.width, 32, 32,6,6)

        self.L5 = OperatorBlock_2D(8*factor*self.width, 2*factor*self.width, 48, 48,14,14)

        self.L6 = OperatorBlock_2D(4*factor*self.width, self.width, 64, 64,22,22) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
         
        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        
        
        x_fc0 = F.pad(x_fc0, [self.padding,self.padding, self.padding,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        
        x_c0 = self.L0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x_c1 = self.L1(x_c0 ,D1//2,D2//2)

        x_c2 = self.L2(x_c1 ,D1//4,D2//4)        
        x_c3 = self.L3(x_c2,D1//4,D2//4)
        x_c4 = self.L4(x_c3,D1//2,D2//2)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)
        x_c5 = self.L5(x_c4,int(D1*self.factor),int(D2*self.factor))
        x_c5 = torch.cat([x_c5, x_c0], dim=1)
        x_c6 = self.L6(x_c5,D1,D2)
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        if self.padding!=0:
            x_c6 = x_c6[..., :-self.padding, :-self.padding]

        x_c6 = x_c6.permute(0, 2, 3, 1)
        
        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)
        
        x_out = self.fc2(x_fc1)
        
        return x_out
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 2*np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 2*np.pi, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((torch.sin(gridx),torch.sin(gridy),torch.cos(gridx),torch.cos(gridy)), dim=-1).to(device)

###
# UNO for high resolution (256x256) navier stocks simulations
###

class UNO_S256(nn.Module):
    def __init__(self, in_width, width,pad = 0, factor = 1):
        super(UNO_S256, self).__init__()
        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc = nn.Linear(self.in_width, 16)

        self.fc0 = nn.Linear(16, self.width) # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_2D(self.width, 2*factor*self.width,64, 64, 32, 33)

        self.L1 = OperatorBlock_2D(2*factor*self.width, 4*factor*self.width, 16, 16, 8,9)

        self.L2 = OperatorBlock_2D(4*factor*self.width, 8*factor*self.width, 8, 8,4,5)
        
        self.L3 = OperatorBlock_2D(8*factor*self.width, 8*factor*self.width, 8, 8,4,5)
        
        self.L4 = OperatorBlock_2D(8*factor*self.width, 4*factor*self.width, 16, 16,4,5)

        self.L5 = OperatorBlock_2D(8*factor*self.width, 2*factor*self.width, 64, 64,8,9)

        self.L6 = OperatorBlock_2D(4*factor*self.width, self.width, 256, 256,32,32) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 3*self.width)
        self.fc2 = nn.Linear(3*self.width + 16, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        x_fc0 = F.pad(x_fc0, [self.padding,self.padding, self.padding,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.L0(x_fc0,D1//4,D2//4)
        x_c1 = self.L1(x_c0,D1//16,D2//16)
        x_c2 = self.L2(x_c1,D1//32,D2//32)        
        x_c3 = self.L3(x_c2,D1//32,D2//32)
        x_c4 = self.L4(x_c3 ,D1//16,D2//16)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)
        x_c5 = self.L5(x_c4 ,D1//4,D2//4)
        x_c5 = torch.cat([x_c5, x_c0], dim=1)

        x_c6 = self.L6(x_c5,D1,D2)
        #print(x.shape)
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        if self.padding!=0:
            x_c6 = x_c6[..., self.padding:-self.padding, self.padding:-self.padding]

        x_c6 = x_c6.permute(0, 2, 3, 1)

        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)

        x_fc1 = torch.cat([x_fc1, x_fc], dim=3)
        x_out = self.fc2(x_fc1)
        
        return x_out
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 2*np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 2*np.pi, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((torch.sin(gridx),torch.sin(gridy),torch.cos(gridx),torch.cos(gridy)), dim=-1).to(device)
