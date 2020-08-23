from collections import namedtuple


""" 

Newly added

Description:
  - decompose the upsampling operations into two primitives

Upsample = upsample primitive + conv
  - upsample primitive: change the spatial size
  - conv: maintain the channel size
 
Goal:
  - search for the upsample operation
  - replace the upsampling operation in DIP's network with the searched operation

Note:
  - do not need a separate point-wise convolution as it is only a 1x1 conv
  
"""

UPSAMPLE_PRIMITIVE = [
  'bilinear',
  'bicubic',
  'nearest',
  'pixel_shuffle',
  'trans_conv', # stride = 1
]

UPSAMPLE_CONV = [
  'conv',
  'trans_conv', # stride = 2
  'split_stack_sum', # additive
  'sep_conv',
  'depth_wise_conv',
  'identity',
]


KERNEL_SIZE = [
  '1x1',
  '3x3',
  '5x5',
  '7x7',
]


DILATION_RATE = [
  '1',
  '2',
  '3',
]


ACTIVATION = [
  'none',
  'ReLU',
  'LeakyReLU',
]
