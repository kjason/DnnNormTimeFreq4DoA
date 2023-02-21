"""
Created on Sat Mar 5 2022
@author: Kuan-Lin Chen
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class U_BasicBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, bias=False, bn=True):
        super(U_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias)
        if bn is True:
            self.bn1 = nn.BatchNorm2d(mid_planes)
            self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y) if hasattr(self,'bn1') else y
        y = F.relu(y)
        y = self.conv2(y)
        y = self.bn2(y) if hasattr(self,'bn2') else y
        out = F.relu(y)
        return out

class U_Encoder(nn.Module):
    def __init__(self, block, num_planes, bias, bn):
        super(U_Encoder, self).__init__()
        self.bias = bias
        self.bn = bn
        self.num_planes = num_planes
        self.num_channels = len(num_planes) - 1
        self.stage = nn.ModuleList()
        for j in range(self.num_channels):
            self.stage.append(block(num_planes[j],num_planes[j+1],num_planes[j+1],bias,bn))
            
    def forward(self, x):
        out = []
        num_down = self.num_channels - 1
        for j in range(num_down):
            y = self.stage[j](x)
            out.append(y)
            x = F.max_pool2d(y,2)
        y = self.stage[-1](x)
        out.append(y)
        return out

class U_Decoder(nn.Module):
    def __init__(self, block, en_num_planes, num_planes, up_num_planes, bias, bn):
        super(U_Decoder, self).__init__()
        self.bias = bias
        self.bn = bn
        self.num_planes = num_planes
        self.num_channels = len(num_planes)
        self.stage = nn.ModuleList()
        self.up = nn.ModuleList()
        for j in range(self.num_channels):
            self.stage.append(block(en_num_planes[-j-1]+up_num_planes[2*j+1],num_planes[j],num_planes[j],bias,bn))
            self.up.append(nn.ConvTranspose2d(up_num_planes[2*j],up_num_planes[2*j+1],kernel_size=2,stride=2))

    def forward(self, x):
        y = x[-1]
        for j in range(self.num_channels):
            x_up = self.up[j](y)
            diff_r = x[-j-2].size()[2] - x_up.size()[2]
            diff_c = x[-j-2].size()[3] - x_up.size()[3]
            x_up = F.pad(x_up, [diff_c // 2, diff_c - diff_c // 2, diff_r // 2, diff_r - diff_r // 2]) 
            x_cat = torch.cat([x[-j-2],x_up],dim=1)
            y = self.stage[j](x_cat)
        return y

class U_Net(nn.Module):
    def __init__(self, block, en_num_planes, de_num_planes, up_num_planes, num_net_out, bias, bn):
        super(U_Net, self).__init__()
        self.bias = bias
        self.bn = bn
        self.encoder = U_Encoder(block,en_num_planes,bias,bn)
        self.decoder = U_Decoder(block,en_num_planes[1:-1],de_num_planes,up_num_planes,bias,bn)
        self.conv = nn.Conv2d(de_num_planes[-1], num_net_out, kernel_size=1)

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        y = self.conv(y)
        return y

class Sigmoid_U_Net(U_Net):
    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)
        y = self.conv(y)
        y = torch.sigmoid(y)
        return y

# models
def COMPLEX_IRM_Sigmoid_U_Net_Standard_Small(): return Sigmoid_U_Net(U_BasicBlock,[2,16,32,64,128,256],[128,64,32,16],[256,128,128,64,64,32,32,16],1,False,True)
def COMPLEX_IRM_Sigmoid_U_Net_Standard_Tiny(): return Sigmoid_U_Net(U_BasicBlock,[2,8,16,32,64,128],[64,32,16,8],[128,64,64,32,32,16,16,8],1,False,True)
def COMPLEX_IRM_Sigmoid_U_Net_Small(): return Sigmoid_U_Net(U_BasicBlock,[2,64,64,64,64,128],[128,128,128,128],[128,64,128,64,128,64,128,64],1,False,True)
def COMPLEX_IRM_Sigmoid_U_Net_Tiny(): return Sigmoid_U_Net(U_BasicBlock,[2,32,32,32,32,64],[64,64,64,64],[64,32,64,32,64,32,64,32],1,False,True)
def COMPLEX_IRM_Sigmoid_U_Net_Expand_Tiny(): return Sigmoid_U_Net(U_BasicBlock,[2,128,32,32,32,64],[64,64,64,64],[64,32,64,32,64,32,64,32],1,False,True)

model_dict = {
                'COMPLEX_IRM_Sigmoid_U_Net_Standard_Small':COMPLEX_IRM_Sigmoid_U_Net_Standard_Small,
                'COMPLEX_IRM_Sigmoid_U_Net_Standard_Tiny':COMPLEX_IRM_Sigmoid_U_Net_Standard_Tiny,
                'COMPLEX_IRM_Sigmoid_U_Net_Small':COMPLEX_IRM_Sigmoid_U_Net_Small,
                'COMPLEX_IRM_Sigmoid_U_Net_Tiny':COMPLEX_IRM_Sigmoid_U_Net_Tiny,
                'COMPLEX_IRM_Sigmoid_U_Net_Expand_Tiny':COMPLEX_IRM_Sigmoid_U_Net_Expand_Tiny
             }
