# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import torch
import model.norm as mynn
from torchvision import models
import torch.nn as nn
from model.resnet import resnet34
# from resnet import resnet34
# import resnet
from functools import partial
from torch.nn import functional as F
# import torchsummary
from torch.nn import init
up_kwargs = {'mode': 'bilinear', 'align_corners': True}
nonlinearity = partial(F.relu, inplace=True)
class S_V1_Net(nn.Module):
    def __init__(self,out_planes=1,ccm=True,norm_layer=nn.BatchNorm2d,is_training=True,expansion=2,base_channel=32):
        super(S_V1_Net,self).__init__()
        
        self.backbone =resnet34(pretrained=False)
        self.expansion=expansion
        self.base_channel=base_channel
        if self.expansion==4 and self.base_channel==64:
            expan=[512,1024,2048]
            spatial_ch=[128,256]
        elif self.expansion==4 and self.base_channel==32:
            expan=[256,512,1024]
            spatial_ch=[32,128]
            conv_channel_up=[256,384,512]
        elif self.expansion==2 and self.base_channel==32:
            expan=[128,256,512]
            spatial_ch=[64,64]
            conv_channel_up=[128,256,512]    
        conv_channel = expan[0] 
        
        self.is_training = is_training
        # self.sap=SAPblock(expan[-1])
        self.sap=SSCM(expan[-1])

        self.decoder5=DecoderBlock(expan[-1],expan[-2],relu=False,last=True) #256
        self.decoder4=DecoderBlock(expan[-2],expan[-3],relu=False) #128
        self.decoder3=DecoderBlock(expan[-3],spatial_ch[-1],relu=False) #64
        self.decoder2=DecoderBlock(spatial_ch[-1],spatial_ch[-2]) #32

      
        # self.mce_2=GPG_2([spatial_ch[-1],expan[0], expan[1], expan[2]],width=spatial_ch[-1], up_kwargs=up_kwargs)
        # self.mce_3=GPG_3([expan[0], expan[1], expan[2]],width=expan[0], up_kwargs=up_kwargs)
        # self.mce_4=GPG_4([expan[1], expan[2]],width=expan[1], up_kwargs=up_kwargs)
        self.mce_2=M2F2(spatial_ch[-1],spatial_ch[-1])
        self.mce_3=M2F2(expan[0],expan[0])
        self.mce_4=M2F2(expan[1],expan[1])
        self.main_head= BaseNetHead(spatial_ch[0], out_planes, 2,
                             is_aux=False, norm_layer=norm_layer)
       
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.backbone.conv1(x) #    
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)#1/2  64   
        
        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)#1/4   64  
        c3 = self.backbone.layer2(c2)#1/8   128  
        c4 = self.backbone.layer3(c3)#1/16   256  
        c5 = self.backbone.layer4(c4)#1/32   512  
        m2=self.mce_2(c2)
        m3=self.mce_3(c3)
        m4=self.mce_4(c4)
        # d_bottom=self.bottom(c5)
        c5=self.sap(c5)

        # d5=d_bottom+c5           #512

        d4=self.relu(self.decoder5(c5)+m4)  #256
        d3=self.relu(self.decoder4(d4)+m3)  #128
        d2=self.relu(self.decoder3(d3)+m2) #64
        d1=self.decoder2(d2)+c1 #32
        main_out=self.main_head(d1)
        main_out=F.log_softmax(main_out,dim=1)

            
        return main_out
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
#        return F.logsigmoid(main_out,dim=1)


class BaseNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BaseNetHead, self).__init__()
        if is_aux:
            self.conv_1x1_3x3=nn.Sequential(
                ConvBnRelu(in_planes, 64, 1, 1, 0,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False),
                ConvBnRelu(64, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False))
        else:
            self.conv_1x1_3x3=nn.Sequential(
                ConvBnRelu(in_planes, 32, 1, 1, 0,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False),
                ConvBnRelu(32, 32, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False))
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1_2 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1_2 = nn.Conv2d(32, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)
        fm = self.conv_1x1_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1_2(fm)
        return output
class SAPblock(nn.Module):
    def __init__(self, in_channels):
        super(SAPblock, self).__init__()
        self.conv3x3=nn.Conv2d(in_channels=in_channels, out_channels=in_channels,dilation=1,kernel_size=3, padding=1)
        
        self.bn=nn.ModuleList([nn.BatchNorm2d(in_channels),nn.BatchNorm2d(in_channels),nn.BatchNorm2d(in_channels)]) 
        self.conv1x1=nn.ModuleList([nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0),
                                    nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels,dilation=1,kernel_size=1, padding=0)])
        self.conv3x3_1=nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,dilation=1,kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,dilation=1,kernel_size=3, padding=1)])
        self.conv3x3_2=nn.ModuleList([nn.Conv2d(in_channels=in_channels//2, out_channels=2,dilation=1,kernel_size=3, padding=1),
                                      nn.Conv2d(in_channels=in_channels//2, out_channels=2,dilation=1,kernel_size=3, padding=1)])
        self.conv_last=ConvBnRelu(in_planes=in_channels,out_planes=in_channels,ksize=1,stride=1,pad=0,dilation=1)



        self.gamma = nn.Parameter(torch.zeros(1))
    
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):

        x_size= x.size()

        branches_1=self.conv3x3(x)
        branches_1=self.bn[0](branches_1)

        branches_2=F.conv2d(x,self.conv3x3.weight,padding=2,dilation=2)#share weight
        branches_2=self.bn[1](branches_2)

        branches_3=F.conv2d(x,self.conv3x3.weight,padding=4,dilation=4)#share weight
        branches_3=self.bn[2](branches_3)

        feat=torch.cat([branches_1,branches_2],dim=1)
        # feat=feat_cat.detach()
        feat=self.relu(self.conv1x1[0](feat))
        feat=self.relu(self.conv3x3_1[0](feat))
        att=self.conv3x3_2[0](feat)
        att = F.softmax(att, dim=1)
        
        att_1=att[:,0,:,:].unsqueeze(1)
        att_2=att[:,1,:,:].unsqueeze(1)

        fusion_1_2=att_1*branches_1+att_2*branches_2



        feat1=torch.cat([fusion_1_2,branches_3],dim=1)
        # feat=feat_cat.detach()
        feat1=self.relu(self.conv1x1[0](feat1))
        feat1=self.relu(self.conv3x3_1[0](feat1))
        att1=self.conv3x3_2[0](feat1)
        att1 = F.softmax(att1, dim=1)
        
        att_1_2=att1[:,0,:,:].unsqueeze(1)
        att_3=att1[:,1,:,:].unsqueeze(1)


        ax=self.relu(self.gamma*(att_1_2*fusion_1_2+att_3*branches_3)+(1-self.gamma)*x)
        ax=self.conv_last(ax)

        return ax
class DecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d,scale=2,relu=True,last=False):
        super(DecoderBlock, self).__init__()
       

        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
       
        self.sap=SAPblock(in_planes)
        self.scale=scale
        self.last=last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.last==False:
            x = self.conv_3x3(x)
            # x=self.sap(x)
        if self.scale>1:
            x=F.interpolate(x,scale_factor=self.scale,mode='bilinear',align_corners=True)
        x=self.conv_1x1(x)
        return x

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class GatedSpatialConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')
        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels+1),
            nn.Conv2d(in_channels+1, in_channels+1, 1),
            nn.ReLU(), 
            nn.Conv2d(in_channels+1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        """

        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))
        input_features = (input_features * (alphas + 1)) 
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
     
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x
    
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)

        return inputs

class HSBlock_rfb(nn.Module):
    '''
    替代3x3卷积
    '''
    def __init__(self, in_ch, s=4, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        '''
        特征大小不改变
        :param in_ch: 输入通道
        :param s: 分组数
        '''
        super(HSBlock_rfb, self).__init__()
        self.s = s
        self.module_list = nn.ModuleList()
        # 避免无法整除通道数
        in_ch, in_ch_last = (in_ch // s, in_ch // s) if in_ch % s == 0 else (in_ch // s + 1, in_ch % s)
        self.module_list.append(nn.Sequential())
        acc_channels = 0
        for i in range(1,self.s):
            if i == 1:
                channels=in_ch
                acc_channels=channels//2
            elif i == s - 1:
                channels = in_ch_last + acc_channels
            else:
                channels=in_ch+acc_channels
                acc_channels=channels//2
            self.module_list.append(self.conv_bn_relu(in_ch=channels, out_ch=channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
        self.initialize_weights()

    def conv_bn_relu(self, in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return conv_bn_relu

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1))
        for i in range(1, len(self.module_list)):
            y = self.module_list[i](x[i])
            if i == len(self.module_list) - 1:
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
        return x[0]
class BasicConv_hs(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv_hs, self).__init__()
        self.out_channels = out_planes
        self.conv = HSBlock_rfb(in_planes, s=4, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialMeanMaxsoftAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialMeanMaxsoftAttention, self).__init__()
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.conv=nn.Sequential(nn.Conv2d(2,1,3,1,1,1),nn.BatchNorm2d(1))
    def forward(self, x):#low,high x torch.Size([1, 128, 56, 56])
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        ###spatial attention
        out2=self.value(x)
        out3=out2.transpose(1,3)
        max_pool = F.max_pool2d(out3,(1,out2.size(1)),(1,out2.size(1)))
        avg_pool = F.avg_pool2d(out3, (1,out2.size(1)),(1,out2.size(1)))
        # sf_pool_f = SoftPooling2D((1,out2.size(1)),(1,out2.size(1)))
        # sf_pool = sf_pool_f(out3)
        out2=torch.cat([max_pool.transpose(1,3),avg_pool.transpose(1,3)],dim=1) #torch.Size([1, 2, 56, 56])
        out2=self.conv(out2)
        out2=self.softmax(out2) #torch.Size([1, 1, 56, 56])
        outspatial=torch.matmul(x,out2)#torch.Size([1, 128, 56, 56])

        return outspatial
class M2F2(nn.Module):#RFB_hs

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(M2F2, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv_hs(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv_hs(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),# fig 4(a)中5 x 5 conv拆分成两个3 x 3 conv
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv_hs(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )
                

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1) # concate
        out = self.ConvLinear(out) # 1 x 1 conv
        short = self.shortcut(x) # shortcut
        out = out*self.scale + short # 结合fig 4(a)很容易理解
        out = self.relu(out) # 最后做一个relu

        return out
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//8,kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        # self.up=nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=2)
        # self.conv=nn.Sequential(nn.Conv2d(3,1,3,1,1,1),nn.BatchNorm2d(1))
    def forward(self, x):#low,high x torch.Size([1, 128, 56, 56])
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        # y=self.up(y)    self attention
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)#HW*HW
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        outself = self.gamma * weights + x 
        ###channel attention
        # outChannel=ChannelMeanMaxsoftAttention(x)
        # output=outself+outspatial+outChannel
        return outself
class ChannelMeanMaxsoftAttention(nn.Module):
    def __init__(self, num_channels):
        super(ChannelMeanMaxsoftAttention, self).__init__()
        num_channels_reduced = num_channels // 2
        # self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias = True)
        self.conv1_1=nn.Conv2d(num_channels,num_channels_reduced,1)
        # self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias = True)#使用1*1卷积代替全连接
        self.conv1_2=nn.Conv2d(num_channels_reduced,num_channels,1)
        self.relu = nonlinearity
        # self.softpool=SoftPooling2D(2,2)
    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()#(1,256,28,28)
        squeeze_tensor_mean = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        fc_out_1_mean = self.relu(self.conv1_1(squeeze_tensor_mean.unsqueeze(dim=2).unsqueeze(dim=3)))#torch.Size([1, 256])
        fc_out_2_mean = self.conv1_2(fc_out_1_mean) #torch.Size([1, 256])
        
        squeeze_tensor_max = input_tensor.view(batch_size, num_channels, -1).max(dim=2)[0]
        fc_out_1_max = self.relu(self.conv1_1(squeeze_tensor_max.unsqueeze(dim=2).unsqueeze(dim=3)))
        fc_out_2_max = self.conv1_2(fc_out_1_max) #torch.Size([1, 256])

        # squeeze_tensor_softpool=SoftPooling2D((input_tensor.size(2), input_tensor.size(3)), (input_tensor.size(2), input_tensor.size(3)))
        # squeeze_tensor_soft=squeeze_tensor_softpool(input_tensor).view(batch_size,-1) #torch.Size([1, 256, 1, 1])
        # fc_out_1_softpool = self.relu(self.conv1_1(squeeze_tensor_soft.unsqueeze(dim=2).unsqueeze(dim=3)))
        # fc_out_2_softpool = self.conv1_2(fc_out_1_softpool)#torch.Size([1, 256])

        a, b = squeeze_tensor_mean.size()
        result = torch.Tensor(a,b)
        # result = torch.add(fc_out_2_mean, fc_out_2_max)
        # result=torch.add(result,fc_out_2_softpool)
        result=fc_out_2_mean+fc_out_2_max
        fc_out_2 = torch.sigmoid(result)#torch.Size([1, 256])
        output_tensor = torch.mul(input_tensor, fc_out_2)
        return output_tensor

class SSCM(nn.Module):
    def __init__(self,channel) -> None:
        super(SSCM,self).__init__()
        self.selfA=SelfAttention(channel)
        self.channelA=ChannelMeanMaxsoftAttention(channel)
        self.spatialA=SpatialMeanMaxsoftAttention(channel)
    def forward(self,x):
        selfA=self.selfA(x)
        channelA=self.channelA(x)
        spatialA=self.spatialA(x)
        out=selfA+channelA+spatialA
        return out




# if __name__ == '__main__':
   

#    model = BaseNet()
#    torchsummary.summary(model, (3, 512, 512))
# torch.Size([1, 1, 224, 224])
# net total parameters: 34870232