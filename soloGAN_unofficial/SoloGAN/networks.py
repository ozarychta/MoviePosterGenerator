"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import math
import functools
from torch.nn import init
from cbin import CBINorm2d
from cbbn import CBBNorm2d
from torch.nn.utils.spectral_norm import spectral_norm
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass


##################################################################################
# Discriminator
##################################################################################
class Discreminator(nn.Module):
    def __init__(self, opt): #input_nc=3, ndf=64, block_num=4, nd=4, norm_type='instance'):
        super(Discreminator, self).__init__()
        nl_layer = get_nl_layer('lrelu')
        input_nc = opt.input_nc
        ndf = opt.ndf
        block_num = opt.de_blocks
        nd =opt.num_domains
        norm_type = opt.norm
        block = [Conv2dBlock(input_nc, ndf, kernel_size=4,stride=2,padding=1,bias=False,nl_layer=nl_layer)]
        dim_in=ndf
        for n in range(0, block_num):
            dim_out = dim_in*2
            block += [Conv2dBlock(dim_in, dim_out, kernel_size=4, stride=2, padding=1,bias=False,nl_layer=nl_layer)]
            dim_in = dim_out
        self.conv1 = nn.Sequential(*block) #(1,1024,8,8)
        self.gap = nn.AdaptiveAvgPool2d(1) #(1,1024,1,1)
        dim_out = dim_in
        self.fc1 =nn.Linear(dim_out,1)
        self.embed = nn.Embedding(nd,dim_out)
        dim_out = dim_in*2
        block = [Conv2dBlock(dim_in,dim_out,kernel_size=4,stride=2,padding=1,nl_layer=nl_layer)] #(1,2048,4,4)
        block +=[Conv2dBlock(dim_out,nd,kernel_size=4,stride=1,padding=0)]
        self.conv2 = nn.Sequential(*block)

    def forward(self, x,y):
        y = y.squeeze()
        b =list(x.shape)[0]
        out = self.conv1(x)
        h = self.gap(out)               # (4,1024,1,1)
        #out1 = h.view(h.size(0), -1)     # (4,1024)
        out1 = h.view(-1)     # (4,1024)
        d = self.fc1(out1)               # (4,1)
        word = self.embed(y.long())        # (4,1024)
        dis =torch.dot(word,out1)
        # dis = torch.bmm(word.view(word.size(0),1,word.size(1)), h.view(h.size(0), h.size(1), 1))        # word:(4,1,1024) h:(4,1024,1) batch inner product
        dis = dis.view(1, -1) + d     # (4,1)
        cls = self.conv2(out).view(1,-1)       #(4,4,1,1)
        return dis,cls

##################################################################################
# Generator
##################################################################################

class Generator(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, opt):
        super(Generator, self).__init__()
        input_nc = opt.input_nc #3
        output_nc=opt.output_nc #3
        nef = opt.nef #64
        ngf = opt.ngf #256
        nd = opt.num_domains #4
        se_blocks = opt.se_blocks #3
        ce_blocks= opt.ce_blocks #4
        ge_blocks = opt.ge_blocks #5
        style_dim= opt.style_dim #8
        content_dim = opt.content_dim #256
        nc=nd+style_dim # add style and label to feed in CBIN in generator 12
        norm_type = opt.norm
        # Style encoder
        self.enc_style = Style_encoder(input_nc= input_nc, output_nc=style_dim, nef= nef, nd=nd, n_blocks= se_blocks, norm_type=norm_type)
        # Content encoder
        self.enc_content = Content_encoder(input_nc=input_nc,output_nc=content_dim, nef=nef,n_blocks=ce_blocks,norm_type=norm_type)
        # Decoder #########################
        self.decoder = decoder(input_nc=content_dim,output_nc=output_nc,ngf=ngf,nc=nc,e_blocks=ge_blocks,norm_type=norm_type,up_type='Trp')

    def forward(self, images, labels):
        # reconstruct an image
        style_fake = self.enc_style(images,labels)
        content = self.enc_content(images)
        images_recon = self.decode(content, style_fake,labels)
        return images_recon

    def encode(self, images,labels):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images,labels)
        content = self.enc_content(images)
        return content, style_fake

    def encode_style(self, images,labels):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images,labels)
        return style_fake

    def encode_content(self, images):
        content = self.enc_content(images)
        return content

    def decode(self, content, style,labels):
        # decode content and style codes to an image
        style = torch.normal(0, 1, size=style.size()).cuda()
        images = self.decoder(content,style,labels)
        return images


##################################################################################
# Encoder and Decoders
##################################################################################
# Style encoder
class Style_encoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=8, nef=64, nd=4, n_blocks=3, norm_type='instance'):
        # img 128*128 -> n_blocks=4 // img 256*256 -> n_blocks=5
        super(Style_encoder, self).__init__()
        _, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=nd)
        max_ndf = 4
        nl_layer = get_nl_layer(layer_type='relu')
        self.entry = Conv2dBlock(input_nc, nef, kernel_size=4, stride=2, padding=1, bias=True)  #(1,64,128,128)
        conv_layers =[]

        for n in range(0, n_blocks):
            input_ndf = min(256, nef)
            output_ndf = min(256, input_ndf*2)
            conv_layers += [CDResBlock(input_ndf, output_ndf, c_norm_layer, nl_layer)]
            nef = nef*2

        self.middle = nn.Sequential(*conv_layers) #(1,256,16,16)
        self.cnorm2 = c_norm_layer(output_ndf)
        self.exit = nn.Sequential(*[nl_layer(), nn.AdaptiveAvgPool2d(1)]) #(1,256,1,1)
        # self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fc = nn.Linear(output_ndf,output_nc)


    def forward(self, x, d):
        x_conv = self.exit(self.cnorm2(self.middle([self.entry(x),d])[0],d))
        b = x_conv.size(0)
        x_conv = x_conv.view(b, -1)
        out = self.fc(x_conv)
        return out #.squeeze()  # out should be 1*8


# Content encoder
class Content_encoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=256, nef=64, nd=4, n_blocks=4, norm_type='instance'):
        # img 128*128 -> n_blocks=4 // img 256*256 -> n_blocks=5
        super(Content_encoder, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=nd)
        max_ndf = 4
        nl_layer = get_nl_layer(layer_type='relu')
        self.conv1 = Conv2dBlock(input_nc, nef, kernel_size=7, stride=1, padding=3, bias=True,norm_layer=norm_layer,nl_layer=nl_layer) #(1,64,256,256)
        self.conv2 = Conv2dBlock(nef,nef*2,kernel_size=4, stride=2, padding=1, bias=True, norm_layer=norm_layer,nl_layer=nl_layer) #(1,128,128,128)
        nef=nef*2
        self.conv3 = Conv2dBlock(nef,nef*2,kernel_size=4, stride=2, padding=1, bias=True,norm_layer=norm_layer,nl_layer=nl_layer) #(1,256,64,64)
        nef=nef*2
        conv_layers =[]
        for n in range(0, n_blocks):
            conv_layers += [RResBlock(nef, norm_layer, nl_layer)]
        self.middle = nn.Sequential(*conv_layers)
        # conv_layers = [RResBlock(nef, norm_layer, nl_layer)]
        # self.middle = nn.Sequential(*conv_layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = self.middle(x3)
        return out  # out should be (1,256,64,64)

# decoder
class decoder(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, ngf=256, nc=4, e_blocks=5, norm_type='instance',up_type='Trp'):
        super(decoder, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=nc)
        nl_layer = get_nl_layer(layer_type='relu')
        self.c1 = Conv2dBlock(input_nc, ngf, kernel_size=3, stride=1, padding=1, pad_type='reflect', bias=False)
        self.n1 = c_norm_layer(ngf)
        self.a1 = nl_layer()
        block = []
        for i in range(e_blocks):
            block.append(CResidualBlock(ngf, c_norm_layer=c_norm_layer,nl_layer=nl_layer))
        self.resBlocks =  nn.Sequential(*block)
        self.n2 = c_norm_layer(ngf) #(1,256,64,64)
        self.a2 = nl_layer()
        self.IC1 = TrConv2dBlock(in_planes=ngf,out_planes=(ngf//2),kernel_size=4,stride=2,padding=1)
        # self.IC1 = TrConv2dBlock(in_planes=ngf,(ngf/2),kernel_size=4,stride=2,padding=1)
        ngf = ngf//2
        self.n3 = c_norm_layer(ngf) #(1,128,128,128)
        self.a3 = nl_layer()
        self.IC2 = TrConv2dBlock(ngf,(ngf//2),kernel_size=4,stride=2,padding=1)
        ngf = ngf//2
        self.n4 = c_norm_layer(ngf) #(1,64,256,256)
        self.a4 =nl_layer()
        self.c2 = Conv2dBlock(ngf, output_nc, kernel_size=7, stride=1, padding=3, pad_type='reflect', bias=False,nl_layer=nn.Tanh) #(1,3,256,256)

    def forward(self, x,z, c):
        c_new =torch.cat((z,c),dim=1)
        x = self.a1(self.n1(self.c1(x),c_new))
        x1 = self.a2(self.n2(self.resBlocks([x,c_new])[0],c_new))
        x2 = self.a3(self.n3(self.IC1(x1),c_new))
        x3 = self.a4(self.n4(self.IC2(x2),c_new))
        y = self.c2(x3)
        return y

############################################################
#==================== Residual blocks ======================
############################################################
# CDResidual block---Style encoder=============
class CDResBlock(nn.Module):
    def __init__(self, inplanes, outplanes, c_norm_layer=None, nl_layer=None):
        super(CDResBlock, self).__init__()
        self.cnorm1 = c_norm_layer(inplanes)            #CBIN (take image x and label d)
        self.nl1 = nl_layer()                           #activation relu
        self.conv1 = conv3x3(inplanes, inplanes)        #conv3*3 ( kernal 3*3, stride 1, padding 1)
        self.cnorm2 = c_norm_layer(inplanes)            # CBIN (output of conv1 and label d)
        self.nl2 = nl_layer()                           # Relu
        self.cmp = meanpoolConv(inplanes, outplanes)    # con3*3 avgpooling 2,2 1st output
        self.shortcut = convMeanpool(inplanes, outplanes) # Avg2,2, conv1*1, 2nd output

    def forward(self, input):
        x, d = input
        out = self.cmp(self.nl2(self.cnorm2(self.conv1(self.nl1(self.cnorm1(x,d))),d)))
        out = out + self.shortcut(x)
        return [out,d]

# Regular ResBlock for content encoder
class RResBlock(nn.Module):
    """Residual Block."""
    def __init__(self, h_dim, norm_layer=None, nl_layer=None):
        super(RResBlock, self).__init__()
        self.c1 = conv3x3(h_dim,h_dim,norm_layer=norm_layer,nl_layer=nl_layer)
        self.c2 = conv3x3(h_dim,h_dim,norm_layer=norm_layer,nl_layer=nl_layer)

    def forward(self, x):
        # x, d = input
        y = self.c1(x)
        y = self.c2(y)
        out = y+ x
        return out

    # Conditional residual block for Generator
class CResidualBlock(nn.Module):
    def __init__(self, h_dim, c_norm_layer=None, nl_layer=None):
        super(CResidualBlock, self).__init__()
        self.c1 = c_norm_layer(h_dim)
        self.n1 = nl_layer()
        self.conv1 = conv3x3(h_dim,h_dim) #,norm_layer=norm_layer,nl_layer=nl_layer)
        self.c2 = c_norm_layer(h_dim)
        self.n2 = nl_layer()
        self.conv2 = conv3x3(h_dim,h_dim)

    def forward(self, input):
        x, c = input[0], input[1]
        y = self.conv1(self.n1(self.c1(x,c)))
        y = self.conv2(self.n2(self.c2(y,c)))
        return [x + y,  c]

##################################################################################
# Basic Blocks
##################################################################################
class Conv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, pad_type='reflect', bias=True, norm_layer=None, nl_layer=None):
        super(Conv2dBlock, self).__init__()
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        self.conv = spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                            padding=0, bias=bias))
        if norm_layer is not None:
            self.norm = norm_layer(out_planes)
        else:
            self.norm = lambda x: x

        if nl_layer is not None:
            self.activation = nl_layer()
        else:
            self.activation = lambda x: x

    def forward(self, x):
        return self.activation(self.norm(self.conv(self.pad(x))))

class TrConv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True, dilation=1, norm_layer=None, nl_layer=None):
        super(TrConv2dBlock, self).__init__()
        self.trConv = spectral_norm(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                                       padding=padding, bias=bias, dilation=dilation))
        if norm_layer is not None:
            self.norm = norm_layer(out_planes)
        else:
            self.norm = lambda x: x

        if nl_layer is not None:
            self.activation = nl_layer()
        else:
            self.activation = lambda x: x

    def forward(self, x):
        return self.activation(self.norm(self.trConv(x)))

def conv3x3(in_planes, out_planes, norm_layer=None, nl_layer=None):
    "3x3 convolution with padding"
    return Conv2dBlock(in_planes, out_planes, kernel_size=3, stride=1,
                       padding=1, pad_type='reflect', bias=False, norm_layer=norm_layer, nl_layer=nl_layer)

def conv1x1(in_planes, out_planes, norm_layer=None, nl_layer=None):
    "1x1 convolution with no padding"
    return Conv2dBlock(in_planes, out_planes, kernel_size=1, stride=1,
                       padding=0, pad_type='reflect', bias=False, norm_layer=norm_layer, nl_layer=nl_layer)

def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [Conv2dBlock(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)

def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)



## nd =  number of domains used in cbin nd to num_feature
# normalization layer and activation layer

def get_norm_layer(layer_type='instance', num_con=4):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        c_norm_layer = functools.partial(CBBNorm2d, affine=True, num_con=num_con)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        c_norm_layer = functools.partial(CBINorm2d, affine=True, num_con=num_con)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer, c_norm_layer

def get_nl_layer(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'sigmoid':
        nl_layer = nn.Sigmoid
    elif layer_type == 'tanh':
        nl_layer = nn.Tanh
    else:
        raise NotImplementedError('nl_layer layer [%s] is not found' % layer_type)
    return nl_layer


# class Upsampling2dBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, type='Trp', norm_layer=None, nl_layer=None):
#         super(Upsampling2dBlock, self).__init__()
#         if type=='Trp':
#             self.upsample = TrConv2dBlock(in_planes,out_planes,kernel_size=4,stride=2,padding=1,bias=False,norm_layer=norm_layer,nl_layer=nl_layer)
#         elif type=='Ner':
#             self.upsample = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='nearest'),
#                 Conv2dBlock(in_planes,out_planes,kernel_size=4, stride=1, padding=1, pad_type='reflect', bias=False,norm_layer=norm_layer,nl_layer=nl_layer)
#             )
#         else:
#             raise('None Upsampling type {}'.format(type))
#     def forward(self, x):
#         return self.upsample(x)




