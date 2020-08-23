import torch
import torch.nn as nn
from .common import *
import ipdb

from NAS import operations
from NAS import gen_upsample_layer



class OutputBlock(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 bias,
                 pad,
                 need_sigmoid):

        super(OutputBlock, self).__init__()

        if need_sigmoid:
            self.op = nn.Sequential(
                conv(in_f=in_channel,
                     out_f=out_channel,
                     kernel_size=kernel_size,
                     bias=bias,
                     pad=pad
                ),
                bn(num_features=out_channel),
                act(act_fun=act_fun)
                nn.Sigmoid(),
            )

        else:
            self.op = nn.Sequential(
                conv(in_f=in_channel,
                     out_f=out_channel,
                     kernel_size=kernel_size,
                     bias=bias,
                     pad=pad
                ),
            )

    def forward(self, data):
        return self.op(data)


class UpsampleBlock(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 model_index):
        
        super(UpsampleBlock, self).__init__()

        self.op = gen_upsample_layer.gen_layer(
            C_in=in_channel,                            
            C_out=out_channel,
            model_index=model_index
        )
        
    def forward(self, data):
        return self.op(data)


class SkipBlock(nn.Module):

    def __init__(self, 
                 in_channel, 
                 out_channel, 
                 kernel_size, 
                 bias,
                 pad,
                 act_fun):

        super(SkipBlock, self).__init__()
        
        self.op = nn.Sequential(
            conv(in_f=in_channel, 
                 out_f=out_channel, 
                 kernel_size=kernel_size, 
                 bias=bias, 
                 pad=pad),
            bn(num_features=out_channel),
            act(act_fun=act_fun)
        )

    def forward(self, data):
        return self.op(data)


class EncoderBlock(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 bias,
                 pad,
                 act_fun,
                 downsample_mode):

        super(EncoderBlock, self).__init__()

        self.op = nn.Sequential(
            conv(in_f=in_channel, out_f=out_channel, kernel_size=kernel_size, stride=2, bias=bias, pad=pad, downsample_mode=downsample_mode),
            bn(num_features=out_channel),
            act(act_fun=act_fun),
            conv(in_f=out_channel, out_f=out_channel, kernel_size=kernel_size, bias=bias, pad=pad),
            bn(num_features=out_channel),
            act(act_fun=act_fun),
        )

    def forward(self, data):
        return self.op(data)


class DecoderBlock(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 bias,
                 pad,
                 act_fun,
                 need1x1_up):

        super(DecoderBlock, self).__init__()

        if need1x1_up:
            self.op = nn.Sequential(
                bn(num_features=out_channel),
                conv(in_f=in_channel, out_f=out_channel, kernel_size=kernel_size, stride=1, bias=bias, pad=pad),
                bn(num_features=out_channel),
                act(act_fun=act_fun),
                conv(in_f=out_channel, out_f=out_channel, kernel_size=kernel_size, bias=bias, pad=pad),
                bn(num_features=out_channel),
                act(act_fun=act_fun),
                conv(in_f=out_channel, out_f=out_channel, kernel_size=1, bias=bias, pad=pad),
                bn(num_features=out_channel),
                act(act_fun=act_fun),
            )

        else:
            self.op = nn.Sequential(
                bn(num_features=out_channel),
                conv(in_f=in_channel, out_f=out_channel, kernel_size=kernel_size, stride=1, bias=bias, pad=pad),
                bn(num_features=out_channel),
                act(act_fun=act_fun),
                conv(in_f=out_channel, out_f=out_channel, kernel_size=kernel_size, bias=bias, pad=pad),
                bn(num_features=out_channel),
                act(act_fun=act_fun),
            )

    def forward(self, data):
        return self.op(data)


class skip(nn.Module):

    def __init__(self,
                 model_index,
                 num_input_channels=2,
                 num_output_channels=3,
                 num_channels_down=[16, 32, 64, 128, 128],
                 num_channels_up=[16, 32, 64, 128, 128],
                 num_channels_skip=[4, 4, 4, 4, 4],
                 filter_size_down=3,
                 filter_size_up=3,
                 filter_skip_size=1,
                 need_sigmoid=True,
                 need_bias=True,
                 pad='zero',
                 upsample_mode='nearest',
                 downsample_mode='stride',
                 act_fun='LeakyReLU',
                 need1x1_up=True,
                 skip_conv=True):

        super(skip, self).__init__()

        self.skip_conv = skip_conv

        self.enc1 = EncoderBlock(in_channel=num_input_channels, out_channel=num_channels_down[0], kernel_size=filter_size_down,
                                 bias=need_bias, pad=pad, act_fun=act_fun, downsample_mode=downsample_mode)

        self.enc2 = EncoderBlock(in_channel=num_channels_down[0], out_channel=num_channels_down[1], kernel_size=filter_size_down,
                                 bias=need_bias, pad=pad, act_fun=act_fun, downsample_mode=downsample_mode)

        self.enc3 = EncoderBlock(in_channel=num_channels_down[1], out_channel=num_channels_down[2], kernel_size=filter_size_down,
                                 bias=need_bias, pad=pad, act_fun=act_fun, downsample_mode=downsample_mode)

        self.enc4 = EncoderBlock(in_channel=num_channels_down[2], out_channel=num_channels_down[3], kernel_size=filter_size_down,
                                 bias=need_bias, pad=pad, act_fun=act_fun, downsample_mode=downsample_mode)

        self.enc5 = EncoderBlock(in_channel=num_channels_down[3], out_channel=num_channels_down[4], kernel_size=filter_size_down,
                                 bias=need_bias, pad=pad, act_fun=act_fun, downsample_mode=downsample_mode)


        if self.skip_conv:
            self.upsample5 = UpsampleBlock(in_channel=num_channels_up[4], 
                                           out_channel=num_channels_up[4], model_index=model_index)

            self.upsample4 = UpsampleBlock(in_channel=num_channels_up[3], 
                                           out_channel=num_channels_up[3], model_index=model_index)

            self.upsample3 = UpsampleBlock(in_channel=num_channels_up[2], 
                                           out_channel=num_channels_up[2], model_index=model_index)

            self.upsample2 = UpsampleBlock(in_channel=num_channels_up[1], 
                                           out_channel=num_channels_up[1], model_index=model_index)

            self.upsample1 = UpsampleBlock(in_channel=num_channels_up[0], 
                                           out_channel=num_channels_up[0], model_index=model_index)


        self.dec5 = DecoderBlock(in_channel=num_channels_down[4], out_channel=num_channels_up[4], kernel_size=filter_size_up, 
                                 bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=False)
                                 #bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)

        self.dec4 = DecoderBlock(in_channel=num_channels_up[3], out_channel=num_channels_up[3], kernel_size=filter_size_up, 
                                 bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=False)
                                 #bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)

        self.dec3 = DecoderBlock(in_channel=num_channels_up[2], out_channel=num_channels_up[2], kernel_size=filter_size_up, 
                                 bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=False)
                                 #bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)

        self.dec2 = DecoderBlock(in_channel=num_channels_up[1], out_channel=num_channels_up[1], kernel_size=filter_size_up, 
                                 bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=False)
                                 #bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)

        self.dec1 = DecoderBlock(in_channel=num_channels_up[0], out_channel=num_channels_up[0], kernel_size=filter_size_up, 
                                 bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=False)
                                 #bias=need_bias, pad=pad, act_fun=act_fun, need1x1_up=need1x1_up)


        self.output = OutputBlock(in_channel=num_channels_up[0], out_channel=num_output_channels,
                                  kernel_size=1, bias=need_bias, pad=pad, need_sigmoid=need_sigmoid)


    def forward(self, data):

        enc1 = self.enc1(data)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        
        dec5 = self.dec5(enc5)
        if self.skip_conv:
            enc5 = self.skip5(enc5)
        up5 = self.upsample5(dec5+enc5)

        dec4 = self.dec4(up5)
        if self.skip_conv:
            enc4 = self.skip4(enc4)
        up4 = self.upsample4(dec4+enc4)

        dec3 = self.dec3(up4)
        if self.skip_conv:
            enc3 = self.skip3(enc3)
        up3 = self.upsample3(dec3+enc3)

        dec2 = self.dec2(up3)
        if self.skip_conv:
            enc2 = self.skip2(enc2)
        up2 = self.upsample2(dec2+enc2)

        dec1 = self.dec1(up2)
        if self.skip_conv:
            enc1 = self.skip5(enc1)
        up1 = self.upsample1(dec1+enc1)

        out = self.output(up1)

        return out
