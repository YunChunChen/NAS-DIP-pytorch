import torch
import torch.nn as nn
from .common import *

import models.gen_upsample_layer

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

        self.op = models.gen_upsample_layer.gen_layer(
            C_in=in_channel,                            
            C_out=out_channel,
            model_index=model_index
        )
        
    def forward(self, data):
        return self.op(data)


class DownsampleBlock(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 bias,
                 pad,
                 act_fun,
                 downsample_mode):

        super(DownsampleBlock, self).__init__()

        self.op = nn.Sequential(
            conv(in_f=in_channel, out_f=out_channel, kernel_size=kernel_size, stride=2, bias=bias, pad=pad, downsample_mode=downsample_mode),
            bn(num_features=out_channel),
            act(act_fun=act_fun)
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


class Model(nn.Module):

    def __init__(self,
                 model_index=119,
                 num_input_channels=32,
                 num_output_channels=3,
                 num_channels_down=[128, 128, 128, 128, 128],
                 num_channels_up=[128, 128, 128, 128, 128],
                 num_channels_skip=[4, 4, 4, 4, 4],
                 filter_size_down=3,
                 filter_size_up=3,
                 filter_skip_size=1,
                 need_sigmoid=True,
                 need_bias=True,
                 pad='reflection',
                 upsample_mode='nearest',
                 downsample_mode='stride',
                 act_fun='LeakyReLU',
                 need1x1_up=True):

        super(Model, self).__init__()

        self.enc1 = EncoderBlock(in_channel=num_input_channels,                      
                                 out_channel=num_channels_down[0], 
                                 kernel_size=filter_size_down,
                                 bias=need_bias, 
                                 pad=pad, 
                                 act_fun=act_fun, 
                                 downsample_mode=downsample_mode)

        self.enc2 = EncoderBlock(in_channel=num_channels_down[0], 
                                 out_channel=num_channels_down[1], 
                                 kernel_size=filter_size_down,
                                 bias=need_bias, 
                                 pad=pad, 
                                 act_fun=act_fun, 
                                 downsample_mode=downsample_mode)

        self.enc3 = EncoderBlock(in_channel=num_channels_down[1], 
                                 out_channel=num_channels_down[2], 
                                 kernel_size=filter_size_down,
                                 bias=need_bias, 
                                 pad=pad, 
                                 act_fun=act_fun, 
                                 downsample_mode=downsample_mode)

        self.enc4 = EncoderBlock(in_channel=num_channels_down[2], 
                                 out_channel=num_channels_down[3], 
                                 kernel_size=filter_size_down,
                                 bias=need_bias, 
                                 pad=pad, 
                                 act_fun=act_fun, 
                                 downsample_mode=downsample_mode)

        self.enc5 = EncoderBlock(in_channel=num_channels_down[3], 
                                 out_channel=num_channels_down[4], 
                                 kernel_size=filter_size_down,
                                 bias=need_bias, 
                                 pad=pad, 
                                 act_fun=act_fun, 
                                 downsample_mode=downsample_mode)

        self.skip1 = SkipBlock(in_channel=num_input_channels, 
                               out_channel=num_channels_up[0], 
                               kernel_size=1, 
                               bias=need_bias, 
                               pad=pad, 
                               act_fun=act_fun)

        self.skip2 = SkipBlock(in_channel=num_channels_down[0], 
                               out_channel=num_channels_up[1], 
                               kernel_size=1, 
                               bias=need_bias, 
                               pad=pad, 
                               act_fun=act_fun)

        self.skip3 = SkipBlock(in_channel=num_channels_down[1], 
                               out_channel=num_channels_up[2], 
                               kernel_size=1, 
                               bias=need_bias, 
                               pad=pad, 
                               act_fun=act_fun)

        self.skip4 = SkipBlock(in_channel=num_channels_down[2], 
                               out_channel=num_channels_up[3], 
                               kernel_size=1, 
                               bias=need_bias, 
                               pad=pad, 
                               act_fun=act_fun)

        self.skip5 = SkipBlock(in_channel=num_channels_down[3], 
                               out_channel=num_channels_up[4], 
                               kernel_size=1, 
                               bias=need_bias, 
                               pad=pad, 
                               act_fun=act_fun)

        self.skip_up_5_4 = UpsampleBlock(in_channel=num_channels_down[4], 
                                         out_channel=num_channels_up[3], 
                                         model_index=model_index)

        self.skip_up_4_3 = UpsampleBlock(in_channel=num_channels_down[3], 
                                         out_channel=num_channels_up[2], 
                                         model_index=model_index)

        self.skip_up_3_2 = UpsampleBlock(in_channel=num_channels_down[2], 
                                         out_channel=num_channels_up[1], 
                                         model_index=model_index)

        self.skip_up_2_1 = UpsampleBlock(in_channel=num_channels_down[1], 
                                         out_channel=num_channels_up[0], 
                                         model_index=model_index)

        self.skip_down_1_2 = DownsampleBlock(in_channel=num_input_channels, 
                                             out_channel=num_channels_up[0], 
                                             kernel_size=filter_size_down,
                                             bias=need_bias, 
                                             pad=pad, 
                                             act_fun=act_fun, 
                                             downsample_mode=downsample_mode)
 
        self.skip_down_2_3 = DownsampleBlock(in_channel=num_channels_down[0], 
                                             out_channel=num_channels_up[1], 
                                             kernel_size=filter_size_down,
                                             bias=need_bias, 
                                             pad=pad, 
                                             act_fun=act_fun, 
                                             downsample_mode=downsample_mode)
 
        self.skip_down_3_4 = DownsampleBlock(in_channel=num_channels_down[1], 
                                             out_channel=num_channels_up[2], 
                                             kernel_size=filter_size_down,
                                             bias=need_bias, 
                                             pad=pad, 
                                             act_fun=act_fun, 
                                             downsample_mode=downsample_mode)
 
        self.skip_down_4_5 = DownsampleBlock(in_channel=num_channels_down[2], 
                                             out_channel=num_channels_up[3], 
                                             kernel_size=filter_size_down,
                                             bias=need_bias, 
                                             pad=pad, 
                                             act_fun=act_fun, 
                                             downsample_mode=downsample_mode)
 
        self.up5 = UpsampleBlock(in_channel=num_channels_up[4], 
                                 out_channel=num_channels_up[4], 
                                 model_index=model_index)

        self.up4 = UpsampleBlock(in_channel=num_channels_up[3], 
                                 out_channel=num_channels_up[3], 
                                 model_index=model_index)

        self.up3 = UpsampleBlock(in_channel=num_channels_up[2], 
                                 out_channel=num_channels_up[2], 
                                 model_index=model_index)

        self.up2 = UpsampleBlock(in_channel=num_channels_up[1], 
                                 out_channel=num_channels_up[1], 
                                 model_index=model_index)

        self.up1 = UpsampleBlock(in_channel=num_channels_up[0], 
                                 out_channel=num_channels_up[0], 
                                 model_index=model_index)

        self.dec5 = DecoderBlock(in_channel=num_channels_down[4], 
                                 out_channel=num_channels_up[4], 
                                 kernel_size=filter_size_up, 
                                 bias=need_bias, 
                                 pad=pad, 
                                 act_fun=act_fun, 
                                 need1x1_up=need1x1_up)

        self.dec4 = DecoderBlock(in_channel=num_channels_up[3], 
                                 out_channel=num_channels_up[3], 
                                 kernel_size=filter_size_up, 
                                 bias=need_bias, 
                                 pad=pad, 
                                 act_fun=act_fun, 
                                 need1x1_up=need1x1_up)

        self.dec3 = DecoderBlock(in_channel=num_channels_up[2], 
                                 out_channel=num_channels_up[2], 
                                 kernel_size=filter_size_up, 
                                 bias=need_bias, 
                                 pad=pad, 
                                 act_fun=act_fun, 
                                 need1x1_up=need1x1_up)

        self.dec2 = DecoderBlock(in_channel=num_channels_up[1], 
                                 out_channel=num_channels_up[1], 
                                 kernel_size=filter_size_up, 
                                 bias=need_bias, 
                                 pad=pad, 
                                 act_fun=act_fun, 
                                 need1x1_up=need1x1_up)

        self.dec1 = DecoderBlock(in_channel=num_channels_up[0], 
                                 out_channel=num_channels_up[0], 
                                 kernel_size=filter_size_up, 
                                 bias=need_bias, 
                                 pad=pad, 
                                 act_fun=act_fun, 
                                 need1x1_up=need1x1_up)

        self.output = OutputBlock(in_channel=num_channels_up[0], 
                                  out_channel=num_output_channels, 
                                  kernel_size=1, 
                                  bias=need_bias, 
                                  pad=pad, 
                                  need_sigmoid=need_sigmoid)

    def forward(self, data):

        enc1 = self.enc1(data)   # H/2  x W/2  x 128
        enc2 = self.enc2(enc1)   # H/4  x W/4  x 128
        enc3 = self.enc3(enc2)   # H/8  x W/8  x 128
        enc4 = self.enc4(enc3)   # H/16 x W/16 x 128
        enc5 = self.enc5(enc4)   # H/32 x W/32 x 128

        add5 = self.up5(enc5) + self.skip_down_4_5(enc3) + self.skip5(enc4)
        dec5 = self.dec5(add5)

        add4 = self.up4(dec5) + self.skip_down_3_4(enc2) + self.skip4(enc3)
        dec4 = self.dec4(add4)

        add3 = self.up3(dec4) + self.skip_down_2_3(enc1) + self.skip3(enc2)
        dec3 = self.dec3(add3)

        add2 = self.up2(dec3) + self.skip_down_1_2(data) + self.skip2(enc1) + self.skip_up_3_2(self.skip_up_4_3(self.skip_up_5_4(enc4)))
        dec2 = self.dec2(add2)

        add1 = self.up1(dec2) + self.skip1(data) + self.skip_up_2_1(self.skip_up_3_2(self.skip_up_4_3(enc3)))
        dec1 = self.dec1(add1)

        out = self.output(dec1)

        return out
