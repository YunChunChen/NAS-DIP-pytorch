import torch
import torch.nn as nn



CONV_OPS = {
  'sep_conv_3x3' : lambda C_in, C_out, affine: SepConv(C_in=C_in, C_out=C_out, kernel_size=3, padding=1, affine=affine),
  'sep_conv_5x5' : lambda C_in, C_out, affine: SepConv(C_in=C_in, C_out=C_out, kernel_size=5, padding=2, affine=affine),
  'sep_conv_7x7' : lambda C_in, C_out, affine: SepConv(C_in=C_in, C_out=C_out, kernel_size=7, padding=3, affine=affine),
  'dil_conv_3x3' : lambda C_in, C_out, affine: DilConv(C_in=C_in, C_out=C_out, kernel_size=3, padding=1, affine=affine),
  'dil_conv_5x5' : lambda C_in, C_out, affine: DilConv(C_in=C_in, C_out=C_out, kernel_size=5, padding=2, affine=affine),
  'dil_conv_7x7' : lambda C_in, C_out, affine: DilConv(C_in=C_in, C_out=C_out, kernel_size=7, padding=3, affine=affine),
}



UPSAMPLE_OPS = {
  'bilinear_conv_3x3': lambda C_in, stride: BilinearConv(C_in=C_in, kernel_size=3, stride=stride, padding=1),
  'bilinear_conv_5x5': lambda C_in, stride: BilinearConv(C_in=C_in, kernel_size=5, stride=stride, padding=2),
  'bilinear_conv_7x7': lambda C_in, stride: BilinearConv(C_in=C_in, kernel_size=7, stride=stride, padding=3),
  'trans_conv_3x3': lambda C_in, stride: TransConv(C_in=C_in, kernel_size=3, stride=stride, padding=1),
  'trans_conv_5x5': lambda C_in, stride: TransConv(C_in=C_in, kernel_size=5, stride=stride, padding=2),
  'trans_conv_7x7': lambda C_in, stride: TransConv(C_in=C_in, kernel_size=7, stride=stride, padding=3),
  'bilinear_additive_3x3': lambda C_in, stride: BilinearAdditive(C_in=C_in, kernel_size=3, stride=stride, padding=1),
  'bilinear_additive_5x5': lambda C_in, stride: BilinearAdditive(C_in=C_in, kernel_size=5, stride=stride, padding=2),
  'bilinear_additive_7x7': lambda C_in, stride: BilinearAdditive(C_in=C_in, kernel_size=7, stride=stride, padding=3),
  'depth_to_space_3x3': lambda C_in, stride: DepthToSpace(C_in=C_in, kernel_size=3, stride=stride, padding=1),
  'depth_to_space_5x5': lambda C_in, stride: DepthToSpace(C_in=C_in, kernel_size=5, stride=stride, padding=2),
  'depth_to_space_7x7': lambda C_in, stride: DepthToSpace(C_in=C_in, kernel_size=7, stride=stride, padding=3),
  'factorized_reduce': lambda C_in, stride: Identity() if stride == 1 else FactorizedReduce(C_in=C_in, C_out=C_in, op_type='upsample'),
  'none': lambda C_in, stride: Zero(stride=stride, op_type='upsample'),
}



DOWNSAMPLE_OPS = {
  'conv_downsample_3x3_dilation_1': lambda C_in, stride: ConvDownSample(C_in=C_in, kernel_size=3, stride=stride, padding=1, dilation=1),
  'conv_downsample_3x3_dilation_2': lambda C_in, stride: ConvDownSample(C_in=C_in, kernel_size=3, stride=stride, padding=2, dilation=2),
  'conv_downsample_5x5_dilation_1': lambda C_in, stride: ConvDownSample(C_in=C_in, kernel_size=5, stride=stride, padding=2, dilation=1),
  'conv_downsample_5x5_dilation_2': lambda C_in, stride: ConvDownSample(C_in=C_in, kernel_size=5, stride=stride, padding=4, dilation=2),
  'conv_downsample_7x7_dilation_1': lambda C_in, stride: ConvDownSample(C_in=C_in, kernel_size=7, stride=stride, padding=3, dilation=1),
  'conv_downsample_7x7_dilation_2': lambda C_in, stride: ConvDownSample(C_in=C_in, kernel_size=7, stride=stride, padding=6, dilation=2),
  'avg_pool_3x3': lambda C_in, stride: nn.AvgPool2d(kernel_size=3, stride=stride, padding=1, count_include_pad=False),
  'avg_pool_5x5': lambda C_in, stride: nn.AvgPool2d(kernel_size=5, stride=stride, padding=2, count_include_pad=False),
  'avg_pool_7x7': lambda C_in, stride: nn.AvgPool2d(kernel_size=7, stride=stride, padding=3, count_include_pad=False),
  'max_pool_3x3': lambda C_in, stride: nn.MaxPool2d(3, stride=stride, padding=1),
  'max_pool_5x5': lambda C_in, stride: nn.MaxPool2d(5, stride=stride, padding=2),
  'max_pool_7x7': lambda C_in, stride: nn.MaxPool2d(7, stride=stride, padding=3),
  'factorized_reduce': lambda C_in, stride: Identity() if stride == 1 else FactorizedReduce(C_in=C_in, C_out=C_in, op_type='downsample'),
  'none': lambda C_in, stride: Zero(stride=stride, op_type='downsample'),
}



class ReLUConvBN(nn.Module):

  def __init__(self, 
               C_in, 
               C_out, 
               kernel_size, 
               stride, 
               padding, 
               affine=True):

    super(ReLUConvBN, self).__init__()

    self.op = nn.Sequential(
      nn.LeakyReLU(0.2, inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
      out = self.op(x)
      return out


class ConvDownSample(nn.Module):
    
  def __init__(self, 
               C_in, 
               kernel_size, 
               stride, 
               padding, 
               dilation):

    super(ConvDownSample, self).__init__()
    
    self.op = nn.Sequential(
        nn.LeakyReLU(0.2, inplace=False),
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, 
                  padding=padding, dilation=dilation, groups=C_in, bias=False),
    )

  def forward(self, x):
      return self.op(x)


class DilConv(nn.Module):
    
  def __init__(self, 
               C_in, 
               C_out, 
               kernel_size,
               padding,
               affine=True): 

    super(DilConv, self).__init__()
    
    self.op = nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
    )

  def forward(self, x):
      return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, 
               C_in, 
               C_out, 
               kernel_size, 
               padding, 
               affine=True):

    super(SepConv, self).__init__()

    self.op = nn.Sequential(
        nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_in, affine=affine),  
        nn.LeakyReLU(0.2, inplace=False),
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
    )

  def forward(self, x):
      return self.op(x)


class BilinearConv(nn.Module):
    
  def __init__(self, 
               C_in, 
               kernel_size,
               stride,
               padding): 

    super(BilinearConv, self).__init__()

    self.op = nn.Sequential(
        nn.LeakyReLU(0.2, inplace=False),
        nn.Upsample(scale_factor=stride, mode='bilinear'),
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, padding=padding, bias=False),
        nn.LeakyReLU(0.2, inplace=False),
    )

  def forward(self, x):
      return self.op(x)


class TransConv(nn.Module):
    
  def __init__(self, 
               C_in, 
               kernel_size, 
               stride, 
               padding, 
               affine=True):

    super(TransConv, self).__init__()

    self.op = nn.Sequential(
        nn.LeakyReLU(0.2, inplace=False),
        nn.ConvTranspose2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=stride-1),
        nn.LeakyReLU(0.2, inplace=False),
    )

  def forward(self, x):
      return self.op(x)


class BilinearAdditive(nn.Module):
    
  def __init__(self, 
               C_in, 
               kernel_size,
               stride,
               padding,
               split=4): 

    super(BilinearAdditive, self).__init__()

    self.chuck_size = int(C_in/split)

    self.op1 = nn.Sequential(
        nn.LeakyReLU(0.2, inplace=False),
        nn.Upsample(scale_factor=stride, mode='bilinear'),
    )

    self.op2 = nn.Sequential(
        nn.Conv2d(int(C_in/split), C_in, kernel_size=kernel_size, padding=padding, bias=False),
        nn.LeakyReLU(0.2, inplace=False),
    )

  def forward(self, x):
      out = self.op1(x)
      split = torch.split(out, self.chuck_size, dim=1) # the resulting number of channels will be 1/4 of the number of input channels
      split_tensor =  torch.stack(split, dim=1)
      out = torch.sum(split_tensor, dim=1)
      out = self.op2(out)
      return out


class DepthToSpace(nn.Module):
    
  def __init__(self, 
               C_in,
               kernel_size,
               stride,
               padding):

    super(DepthToSpace, self).__init__()

    self.op = nn.Sequential(
        nn.LeakyReLU(0.2, inplace=False),
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, padding=padding, bias=False),
        nn.PixelShuffle(stride),
        nn.Conv2d(int(C_in/(stride**2)), C_in, kernel_size=kernel_size, padding=padding, bias=False),
        nn.LeakyReLU(0.2, inplace=False),
    )

  def forward(self, x):
      return self.op(x)


class Zero(nn.Module):

    def __init__(self, 
                 stride,
                 op_type='downsample'):

        super(Zero, self).__init__()

        self.stride = stride
        self.op_type = op_type

        if self.op_type == 'upsample':
            self.op = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)

        if self.op_type == 'downsample':
            return x[:,:,::self.stride,::self.stride].mul(0.)

        else:
            return self.op(x).mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, 
               C_in, 
               C_out, 
               affine=True,
               op_type='downsample'):

    super(FactorizedReduce, self).__init__()

    self.relu = nn.LeakyReLU(0.2, inplace=False)
    
    self.op_type = op_type

    if self.op_type == 'downsample':
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    else:
        self.op_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.op_2 = nn.Upsample(scale_factor=2, mode='bilinear')

    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):

    x = self.relu(x)

    if self.op_type == 'downsample':
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    else:
        out = self.op_1(x)

    out = self.bn(out)

    return out



""" Newly Added """

UPSAMPLE_PRIMITIVE_OPS = {
  'bilinear':      lambda C_in, C_out, kernel_size, use_act: BilinearOp(stride=2, upsample_mode='bilinear', use_act=use_act),
  'bicubic':       lambda C_in, C_out, kernel_size, use_act: BilinearOp(stride=2, upsample_mode='bicubic', use_act=use_act),
  'nearest':       lambda C_in, C_out, kernel_size, use_act: BilinearOp(stride=2, upsample_mode='nearest', use_act=use_act),
  'trans_conv':    lambda C_in, C_out, kernel_size, use_act: TransConvOp(C_in=C_in, C_out=C_out, kernel_size=KERNEL_SIZE_OPS[kernel_size], use_act=use_act, stride=2),
  'pixel_shuffle': lambda C_in, C_out, kernel_size, use_act: DepthToSpaceOp(kernel_size=KERNEL_SIZE_OPS[kernel_size], use_act=use_act, stride=2),
}


UPSAMPLE_CONV_OPS = {
  'conv':            lambda C_in, C_out, kernel_size, use_act: ConvOp(C_in=C_in, C_out=C_out, kernel_size=KERNEL_SIZE_OPS[kernel_size], use_act=use_act),

  'trans_conv':      lambda C_in, C_out, kernel_size, use_act: TransConvOp(C_in=C_in, C_out=C_out, kernel_size=KERNEL_SIZE_OPS[kernel_size], use_act=use_act, stride=1),
  'split_stack_sum': lambda C_in, C_out, kernel_size, use_act: SplitStackSum(C_in=C_in, C_out=C_out, kernel_size=KERNEL_SIZE_OPS[kernel_size], use_act=use_act),
  'sep_conv':        lambda C_in, C_out, kernel_size, use_act: SepConvOp(C_in=C_in, C_out=C_out, kernel_size=KERNEL_SIZE_OPS[kernel_size], use_act=use_act),
  'identity':        lambda C_in, C_out, kernel_size, use_act: Identity(),
}


KERNEL_SIZE_OPS = {
  '3x3': 3,
  '4x4': 4,
  '5x5': 5,
  '7x7': 7,
}


DILATION_RATE_OPS = {
  '1': 1,
  '2': 2,
  '3': 3,
}


class BilinearOp(nn.Module):
    
  def __init__(self, 
               stride,
               upsample_mode,
               use_act): 

    super(BilinearOp, self).__init__()

    if use_act:
      self.op = nn.Sequential(
        nn.Upsample(scale_factor=stride, mode=upsample_mode),
        nn.LeakyReLU(0.2, inplace=False),
      )

    else:
      self.op = nn.Sequential(
        nn.Upsample(scale_factor=stride, mode=upsample_mode),
      )

  def forward(self, x):
    return self.op(x)


class DepthToSpaceOp(nn.Module):
    
  def __init__(self, 
               kernel_size,
               stride,
               use_act,
               affine=True):

    super(DepthToSpaceOp, self).__init__()

    if kernel_size == 3:
      padding = 1
    elif kernel_size == 5:
      padding = 2
    elif kernel_size == 7:
      padding = 3

    if use_act:
      self.op = nn.Sequential(
        nn.PixelShuffle(stride),
        nn.LeakyReLU(0.2, inplace=False),
      )

    else:
      self.op = nn.Sequential(
        nn.PixelShuffle(stride),
      )


  def forward(self, x):
    return self.op(x)


class TransConvOp(nn.Module):
    
  def __init__(self, 
               C_in, 
               C_out, 
               kernel_size, 
               stride, 
               use_act,
               affine=True):

    super(TransConvOp, self).__init__()

    if kernel_size == 3:
      padding = 1
    elif kernel_size == 5:
      padding = 2
    elif kernel_size == 7:
      padding = 3

    if use_act:
        self.op = nn.Sequential(
          nn.ConvTranspose2d(C_in, C_out, kernel_size=kernel_size, 
                             stride=stride, padding=padding, output_padding=stride-1),
          nn.LeakyReLU(0.2, inplace=False),
        )

    else:
      self.op = nn.Sequential(
        nn.ConvTranspose2d(C_in, C_out, kernel_size=kernel_size, 
                           stride=stride, padding=padding, output_padding=stride-1),
      )

  def forward(self, x):
    return self.op(x)


class ConvOp(nn.Module):
    
  def __init__(self, 
               C_in, 
               C_out, 
               kernel_size,
               use_act,
               affine=True): 

    super(ConvOp, self).__init__()

    if kernel_size == 3:
      padding = 1
    elif kernel_size == 5:
      padding = 2
    elif kernel_size == 7:
      padding = 3

    if use_act:
      self.op = nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=padding, bias=False),
        nn.LeakyReLU(0.2, inplace=False),
      )
 
    else:
      self.op = nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=padding, bias=False),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):

    super(Identity, self).__init__()

  def forward(self, x):
    return x


class SplitStackSum(nn.Module):
    
  def __init__(self, 
               C_in, 
               C_out, 
               kernel_size,
               use_act,
               split=4,
               affine=True): 

    super(SplitStackSum, self).__init__()

    if kernel_size == 3:
      padding = 1
    elif kernel_size == 5:
      padding = 2
    elif kernel_size == 7:
      padding = 3

    self.chuck_size = int(C_in/split)

    if use_act:
      self.op = nn.Sequential(
        nn.Conv2d(int(C_in/split), C_out, kernel_size=kernel_size, padding=padding, bias=False),
        nn.LeakyReLU(0.2, inplace=False),
      )
    
    else:
      self.op = nn.Sequential(
        nn.Conv2d(int(C_in/split), C_out, kernel_size=kernel_size, padding=padding, bias=False),
      )

  def forward(self, x):
    split = torch.split(x, self.chuck_size, dim=1) # the resulting number of channels will be 1/4 of the number of input channels
    stack = torch.stack(split, dim=1)
    out = torch.sum(stack, dim=1)
    out = self.op(out)
    return out


class SepConvOp(nn.Module):
  
  def __init__(self, 
               C_in, 
               C_out, 
               kernel_size,
               use_act,
               affine=True): 

    super(SepConvOp, self).__init__()

    if kernel_size == 3:
      padding = 1
    elif kernel_size == 5:
      padding = 2
    elif kernel_size == 7:
      padding = 3

    if use_act:
      self.op = nn.Sequential(
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, padding=padding, groups=C_in, bias=False), # per chaneel conv
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), # pointwise conv (1x1 conv)
        nn.LeakyReLU(0.2, inplace=False),
      )
    
    else:
      self.op = nn.Sequential(
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, padding=padding, groups=C_in, bias=False), # per chaneel conv
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), # pointwise conv (1x1 conv)
      )

  def forward(self, x):
    return self.op(x)
