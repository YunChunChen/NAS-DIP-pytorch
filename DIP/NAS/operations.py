import torch
import torch.nn as nn


UPSAMPLE_PRIMITIVE_OPS = {
  'bilinear':      lambda C_in, C_out, kernel_size, act_op: BilinearOp(stride=2, upsample_mode='bilinear', act_op=act_op),
  'bicubic':       lambda C_in, C_out, kernel_size, act_op: BilinearOp(stride=2, upsample_mode='bicubic', act_op=act_op),
  'nearest':       lambda C_in, C_out, kernel_size, act_op: BilinearOp(stride=2, upsample_mode='nearest', act_op=act_op),
  'trans_conv':    lambda C_in, C_out, kernel_size, act_op: TransConvOp(C_in=C_in, C_out=C_out, kernel_size=kernel_size, act_op=act_op, stride=2),
  'pixel_shuffle': lambda C_in, C_out, kernel_size, act_op: DepthToSpaceOp(act_op=act_op, stride=2),
}


UPSAMPLE_CONV_OPS = {
  'conv':            lambda C_in, C_out, kernel_size, act_op: ConvOp(C_in=C_in, C_out=C_out, kernel_size=kernel_size, act_op=act_op),
  'trans_conv':      lambda C_in, C_out, kernel_size, act_op: TransConvOp(C_in=C_in, C_out=C_out, kernel_size=kernel_size, act_op=act_op, stride=1),
  'split_stack_sum': lambda C_in, C_out, kernel_size, act_op: SplitStackSum(C_in=C_in, C_out=C_out, kernel_size=kernel_size, act_op=act_op),
  'sep_conv':        lambda C_in, C_out, kernel_size, act_op: SepConvOp(C_in=C_in, C_out=C_out, kernel_size=kernel_size, act_op=act_op),
  'depth_wise_conv': lambda C_in, C_out, kernel_size, act_op: DepthWiseConvOp(C_in=C_in, C_out=C_out, kernel_size=kernel_size, act_op=act_op),
  'identity':        lambda C_in, C_out, kernel_size, act_op: Identity(),
}


KERNEL_SIZE_OPS = {
  '1x1': 1,
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


PADDING_OPS = {
  '1x1': 0,
  '3x3': 1,
  '5x5': 2,
  '7x7': 3,
}


ACTIVATION_OPS = {
  'none': None,
  'ReLU': nn.ReLU(),
  'LeakyReLU': nn.LeakyReLU(0.2, inplace=False),
}


class BilinearOp(nn.Module):
    
  def __init__(self, 
               stride,
               upsample_mode,
               act_op): 

    super(BilinearOp, self).__init__()

    activation = ACTIVATION_OPS[act_op]

    if not activation:
      self.op = nn.Sequential(
        nn.Upsample(scale_factor=stride, mode=upsample_mode),
      )

    else:
      self.op = nn.Sequential(
        nn.Upsample(scale_factor=stride, mode=upsample_mode),
        activation,
      )

  def forward(self, x):
    return self.op(x)


class DepthToSpaceOp(nn.Module):
    
  def __init__(self, 
               stride,
               act_op,
               affine=True):

    super(DepthToSpaceOp, self).__init__()

    activation = ACTIVATION_OPS[act_op]

    if not activation:
      self.op = nn.Sequential(
        nn.PixelShuffle(stride),
      )

    else:
      self.op = nn.Sequential(
        nn.PixelShuffle(stride),
        activation,
      )


  def forward(self, x):
    return self.op(x)


class TransConvOp(nn.Module):
    
  def __init__(self, 
               C_in, 
               C_out, 
               kernel_size, 
               stride, 
               act_op,
               affine=True):

    super(TransConvOp, self).__init__()

    padding = PADDING_OPS[kernel_size]
    kernel_size = KERNEL_SIZE_OPS[kernel_size]
    activation = ACTIVATION_OPS[act_op]

    if not activation:
      self.op = nn.Sequential(
        nn.ConvTranspose2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=stride-1),
      )

    else:
      self.op = nn.Sequential(
        nn.ConvTranspose2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=stride-1),
        activation,
      )

  def forward(self, x):
    return self.op(x)


class ConvOp(nn.Module):
    
  def __init__(self, 
               C_in, 
               C_out, 
               kernel_size,
               act_op,
               affine=True): 

    super(ConvOp, self).__init__()

    padding = PADDING_OPS[kernel_size]
    kernel_size = KERNEL_SIZE_OPS[kernel_size]
    activation = ACTIVATION_OPS[act_op]

    if not activation:
      self.op = nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=padding, bias=False),
      )

    else:
      self.op = nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=padding, bias=False),
        activation,
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
               act_op,
               split=4,
               affine=True): 

    super(SplitStackSum, self).__init__()

    padding = PADDING_OPS[kernel_size]
    kernel_size = KERNEL_SIZE_OPS[kernel_size]
    activation = ACTIVATION_OPS[act_op]

    self.chuck_size = int(C_in/split)

    if not activation:
      self.op = nn.Sequential(
        nn.Conv2d(int(C_in/split), C_out, kernel_size=kernel_size, padding=padding, bias=False),
      )

    else:
      self.op = nn.Sequential(
        nn.Conv2d(int(C_in/split), C_out, kernel_size=kernel_size, padding=padding, bias=False),
        activation,
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
               act_op,
               affine=True): 

    super(SepConvOp, self).__init__()

    padding = PADDING_OPS[kernel_size]
    kernel_size = KERNEL_SIZE_OPS[kernel_size]
    activation = ACTIVATION_OPS[act_op]

    if not activation:
      self.op = nn.Sequential(
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, padding=padding, groups=C_in, bias=False), # per chaneel conv
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), # pointwise conv (1x1 conv)
      )
    
    else:
      self.op = nn.Sequential(
        nn.Conv2d(C_in, C_in, kernel_size=kernel_size, padding=padding, groups=C_in, bias=False), # per chaneel conv
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), # pointwise conv (1x1 conv)
        activation,
      )

  def forward(self, x):
    return self.op(x)


class DepthWiseConvOp(nn.Module):
  
  def __init__(self, 
               C_in, 
               C_out, 
               kernel_size,
               act_op,
               affine=True): 

    super(DepthWiseConvOp, self).__init__()

    padding = PADDING_OPS[kernel_size]
    kernel_size = KERNEL_SIZE_OPS[kernel_size]
    activation = ACTIVATION_OPS[act_op]

    if not activation:
      self.op = nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=padding, groups=C_out, bias=False), # per chaneel conv
      )
    
    else:
      self.op = nn.Sequential(
        nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=padding, groups=C_out, bias=False), # per chaneel conv
        activation,
      )

  def forward(self, x):
    return self.op(x)
