import torch
import torch.nn as nn

class encoder_conv2d(nn.Module):
    def __init__(self, input_channels, op_size):
        super(encoder_conv2d, self).__init__()
        self.layer = nn.Conv2d(input_channels, op_size, kernel_size=1)
        
    def forward(self, x):
        x = x.squeeze(1) 
        x = self.layer(x)
        x = x.unsqueeze(1)  
        return x

class decoder_conv2d(nn.Module):
    def __init__(self, output_channels, op_size):
        super(decoder_conv2d, self).__init__()
        self.layer = nn.Conv2d(op_size, output_channels, kernel_size=1)
        
    def forward(self, x):
        # [batch, 1, op_size, height, width]
        x = x.squeeze(1)  #  -> [batch, op_size, height, width]
        x = self.layer(x)
        x = x.unsqueeze(1)  #  -> [batch, 1, channels, height, width]
        return x

class Koopman_Operator2D(nn.Module):
    def __init__(self, op_size, modes_x, modes_y):
        super(Koopman_Operator2D, self).__init__()
        self.op_size = op_size
        self.scale = (1 / (op_size * op_size))
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.koopman_matrix = nn.Parameter(self.scale * torch.rand(op_size, op_size, self.modes_x, self.modes_y, dtype=torch.cfloat))

    def time_marching(self, input, weights):
        return torch.einsum("btxy,tfxy->bfxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes_x, :self.modes_y] = self.time_marching(x_ft[:, :, :self.modes_x, :self.modes_y], self.koopman_matrix)
        out_ft[:, :, -self.modes_x:, :self.modes_y] = self.time_marching(x_ft[:, :, -self.modes_x:, :self.modes_y], self.koopman_matrix)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class KNO2d(nn.Module):
    def __init__(self, input_channels, output_channels, op_size, modes_x=10, modes_y=10, decompose=6, linear_type=True, normalization=False):
        super(KNO2d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.op_size = op_size
        self.decompose = decompose
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.enc = encoder_conv2d(self.input_channels, op_size)
        self.dec = decoder_conv2d(self.output_channels, op_size)
        self.koopman_layer = Koopman_Operator2D(self.op_size, self.modes_x, self.modes_y)
        self.w0 = nn.Conv2d(op_size, op_size, 1)
        self.linear_type = linear_type
        self.normalization = normalization
        if self.normalization:
            self.norm_layer = torch.nn.BatchNorm2d(op_size)
            
    def forward(self, x):        
        x_reconstruct = self.enc(x)  # [B, 1, op_size, H, W]
        x_reconstruct = torch.tanh(x_reconstruct)
        x_reconstruct = self.dec(x_reconstruct)  # [B, 1, C, H, W]
        
        x = self.enc(x)  # [B, 1, op_size, H, W]
        x = torch.tanh(x)
        
        x = x.squeeze(1)  # [B, op_size, H, W]
        x_w = x
        
        for i in range(self.decompose):
            x1 = self.koopman_layer(x)
            if self.linear_type:
                x = x + x1
            else:
                x = torch.tanh(x + x1)
                
        if self.normalization:
            x = torch.tanh(self.norm_layer(self.w0(x_w)) + x)
        else:
            x = torch.tanh(self.w0(x_w) + x)
            
        x = x.unsqueeze(1)  # [B, 1, op_size, H, W]
        x = self.dec(x)  # [B, 1, C, H, W]
        
        return x, x_reconstruct