from torch import nn
import torch
from torch import nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm=act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y

    


def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
    
    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1


class Decoder(nn.Module):
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y

class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class SimVP(nn.Module):
    def __init__(self, shape_in, shape_out, hid_S=32, hid_T=128, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(SimVP, self).__init__()
        T, C, H, W = shape_in
        _, C_o, _, _ = shape_out
        self.C_o = C_o
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)


    def forward(self, x_raw):
        # x_raw = x_raw.unsqueeze(1)
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        term = Y.shape
        Y = Y.reshape(B, T, C, H, W)
        # Y = Y.squeeze(1)
        Y = Y[:,:,0:self.C_o]
        return Y



# import torch
# from torch import nn
# import math
# from einops import rearrange


# class BasicConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
#         super(BasicConv2d, self).__init__()
#         self.act_norm=act_norm
#         if not transpose:
#             self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
#         else:
#             self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
#         #self.norm = nn.GroupNorm(2, out_channels)
#         self.norm = nn.BatchNorm2d(out_channels)
#         self.act = nn.LeakyReLU(0.2, inplace=True)

#     def forward(self, x):
#         y = self.conv(x)
#         if self.act_norm:
#             y = self.act(self.norm(y))
#         return y


# class ConvSC(nn.Module):
#     def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
#         super(ConvSC, self).__init__()
#         if stride == 1:
#             transpose = False
#         self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
#                                 padding=1, transpose=transpose, act_norm=act_norm)

#     def forward(self, x):
#         y = self.conv(x)
#         return y


# class GroupConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
#         super(GroupConv2d, self).__init__()
#         self.act_norm = act_norm
#         if in_channels % groups != 0:
#             groups = 1
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
#         self.norm = nn.GroupNorm(groups,out_channels)
#         self.activate = nn.LeakyReLU(0.2, inplace=True)
    
#     def forward(self, x):
#         y = self.conv(x)
#         if self.act_norm:
#             y = self.activate(self.norm(y))
#         return y


# class Inception(nn.Module):
#     def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
#         super(Inception, self).__init__()
#         self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
#         layers = []
#         for ker in incep_ker:
#             layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
#         self.layers = nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         y = 0
#         for layer in self.layers:
#             y += layer(x)
#         return y
    
# def stride_generator(N, reverse=False):
#     strides = [1, 2]*10
#     if reverse: return list(reversed(strides[:N]))
#     else: return strides[:N]

# class Encoder(nn.Module):
#     def __init__(self,C_in, C_hid, N_S):
#         super(Encoder,self).__init__()
#         strides = stride_generator(N_S)
#         self.enc = nn.Sequential(
#             ConvSC(C_in, C_hid, stride=strides[0]),
#             *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
#         )
    
#     def forward(self,x):# B*4, 3, 128, 128
#         #print('x'+str(x.shape))
#         enc1 = self.enc[0](x)
#         latent = enc1
#         #print('enc'+str(enc1.shape))
#         for i in range(1,len(self.enc)):
#             latent = self.enc[i](latent)
#             #print('enc'+str(enc1.shape))
#         return latent,enc1


# class Decoder(nn.Module):
#     def __init__(self,C_hid, C_out, N_S):
#         super(Decoder,self).__init__()
#         strides = stride_generator(N_S, reverse=True)
#         self.dec = nn.Sequential(
#             *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
#             ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
#         )
#         self.readout = nn.Conv2d(C_hid, C_out, 1)
    
#     def forward(self, hid, enc1=None):
#         for i in range(0,len(self.dec)-1):
#             hid = self.dec[i](hid)
#         Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
#         Y = self.readout(Y)
#         return Y

# class Mid_Xnet(nn.Module):
#     def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
#         super(Mid_Xnet, self).__init__()

#         self.N_T = N_T
#         enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
#         for i in range(1, N_T-1):
#             enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
#         enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

#         dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
#         for i in range(1, N_T-1):
#             dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
#         dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

#         self.enc = nn.Sequential(*enc_layers)
#         self.dec = nn.Sequential(*dec_layers)

#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         x = x.reshape(B, T*C, H, W)

#         # encoder
#         skips = []
#         z = x
#         for i in range(self.N_T):
#             z = self.enc[i](z)
#             if i < self.N_T - 1:
#                 skips.append(z)

#         # decoder
#         z = self.dec[0](z)
#         for i in range(1, self.N_T):
#             z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

#         y = z.reshape(B, T, C, H, W)
#         return y

# # CBAM
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // ratio, in_channels)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * self.sigmoid(y)


# class SpatialAttention(nn.Module):
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         max_pool = torch.max(x, dim=1, keepdim=True)[0]
#         avg_pool = torch.mean(x, dim=1, keepdim=True)
#         y = torch.cat([max_pool, avg_pool], dim=1)
#         y = self.conv(y)
#         return x * self.sigmoid(y)

# class CBAM(nn.Module):
#     def __init__(self, in_channels, ratio=6):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(in_channels, ratio)
#         self.spatial_attention = SpatialAttention()

#     def forward(self, x):
#         x = self.channel_attention(x)
#         x = self.spatial_attention(x)
#         return x

# '''
# class Temporal_evo(nn.Module):
#     def __init__(self, channel_in, channel_hid, N_T, h, w, incep_ker=[3, 5, 7, 11], groups=8):
#         super(Temporal_evo, self).__init__()

#         self.N_T = N_T
#         enc_layers = [Inception(channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
#         for i in range(1, N_T - 1):
#             enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
#         enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))

#         dec_layers = [Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
#         for i in range(1, N_T - 1):
#             dec_layers.append(
#                 Inception(2 * channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
#         dec_layers.append(Inception(2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups))
#         norm_layer = partial(nn.LayerNorm, eps=1e-6)
#         self.norm = norm_layer(channel_hid)

#         self.enc = nn.Sequential(*enc_layers)
#         dpr = [x.item() for x in torch.linspace(0, 0, 12)]
#         self.h = h
#         self.w = w
#         self.blocks = nn.ModuleList([FourierNetBlock(
#             dim=channel_hid,
#             mlp_ratio=4,
#             drop=0.,
#             drop_path=dpr[i],
#             act_layer=nn.GELU,
#             norm_layer=norm_layer,
#             h = self.h,
#             w = self.w)
#             for i in range(12)
#         ])
#         self.dec = nn.Sequential(*dec_layers)

#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         bias = x
#         x = x.reshape(B, T * C, H, W)

#         # downsampling
#         skips = []
#         z = x
#         for i in range(self.N_T):
#             z = self.enc[i](z)
#             if i < self.N_T - 1:
#                 skips.append(z)
        
#         # Spectral Domain
#         B, D, H, W = z.shape
#         N = H * W
#         z = z.permute(0, 2, 3, 1)
#         z = z.view(B, N, D)
#         for blk in self.blocks:
#              z = blk(z)
#         z = self.norm(z).permute(0, 2, 1)

#         z = z.reshape(B, D, H, W)

#         # upsampling
#         z = self.dec[0](z)
#         for i in range(1, self.N_T):
#             z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

#         y = z.reshape(B, T, C, H, W)
#         return y + bias
# '''

# class SimVP(nn.Module):
#     def __init__(self, shape_in, shape_out, channel_attention, ratio=1, hid_S=64, 
#                  hid_T=128, N_S=6, N_T=4, incep_ker=[3,5,7,11], groups=8,
#                  in_time_seq_length=8, out_time_seq_length=7):
#         super(SimVP, self).__init__()
#         T_in, C_in, H, W = shape_in
#         T_out, C_out, _, _ = shape_out
#         self.H1 = int(H / 2 ** (N_S / 2)) + 1 if H % 3 == 0 else int(H / 2 ** (N_S / 2))
#         self.W1 = int(W / 2 ** (N_S / 2))
#         self.in_time_seq_length = in_time_seq_length
#         self.out_time_seq_length = out_time_seq_length
#         self.enc = Encoder(C_in, hid_S, N_S)
#         self.hid = Mid_Xnet(T_in*hid_S, hid_T, N_T, incep_ker, groups)
#         #self.hid = Temporal_evo(T_in*hid_S, hid_T, N_T, self.H1, self.W1, incep_ker, groups) 
#         self.dec = Decoder(hid_S, C_out, N_S)
#         self.cbam = CBAM(C_in, ratio)
#         self.C_out = C_out
#         self.T_out = T_out
#         self.channel_attention = channel_attention
#         self.time_reshape = nn.Conv2d(T_in*C_out, T_out*C_out, 1)


#     def forward_(self, x_raw):
#         B, T_in, C_in, H, W = x_raw.shape
#         # Channel Attention
#         x_raw = x_raw.reshape(B*T_in, C_in, H, W)
#         if not self.channel_attention:
#             x_pre = x_raw
#         else:
#             x_pre = self.cbam(x_raw)
#         x_pre = x_pre.reshape(B, T_in, C_in, H, W) 
          
#         x = x_pre.view(B*T_in, C_in, H, W)
        
#         # SimVP model
#         #print(x.shape)
#         embed, skip = self.enc(x)
#         _, C_, H_, W_ = embed.shape

#         z = embed.view(B, T_in, C_, H_, W_)
#         hid = self.hid(z)
#         hid = hid.reshape(B*T_in, C_, H_, W_)

#         Y = self.dec(hid, skip)
#         Y = Y.reshape(B, T_in, self.C_out, H, W)
#         Y = rearrange(Y,'B T C H W -> B (T C) H W')
#         Y = self.time_reshape(Y)
#         Y = rearrange(Y,'B (T C) H W -> B T C H W', T=self.T_out, C = self.C_out)
#         return Y
    
#     def forward(self, xx):
#         yy = self.forward_(xx)
#         in_time_seq_length, out_time_seq_length = self.in_time_seq_length, self.out_time_seq_length
#         if out_time_seq_length == in_time_seq_length:
#             #print('out_time_seq_length == in_time_seq_length')
#             y_pred = yy
#         if out_time_seq_length < in_time_seq_length:
#             #print('out_time_seq_length < in_time_seq_length')
#             y_pred = yy[:, :out_time_seq_length]
#         elif out_time_seq_length > in_time_seq_length:
#             #print('out_time_seq_length > in_time_seq_length')
#             y_pred = [yy]
#             d = out_time_seq_length // in_time_seq_length
#             m = out_time_seq_length % in_time_seq_length
            
#             for _ in range(1, d):
#                 cur_seq = self.forward_(y_pred[-1])
#                 y_pred.append(cur_seq)
            
#             if m != 0:
#                 cur_seq = self.forward_(y_pred[-1])
#                 y_pred.append(cur_seq[:, :m])
            
#             y_pred = torch.cat(y_pred, dim=1)
#         return y_pred
    
if __name__ == '__main__':
     v = SimVP(shape_in=[1, 6, 360, 720], shape_out=[1,3,360,720])

     img = torch.randn(1, 6, 360, 720)
     preds = v(img)
     print(preds.shape)