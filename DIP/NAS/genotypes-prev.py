from collections import namedtuple


#Genotype = namedtuple('Genotype', 'downsample_conv downsample_method downsample_concat upsample_conv upsample_method upsample_concat')
Genotype = namedtuple('Genotype', 'upsample_prim_method upsample_conv upsample_kernel upsample_concat')


PRIMITIVES = [
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'dil_conv_7x7',
]


UPSAMPLE_METHOD = [
    'bilinear_conv_3x3',
    'bilinear_conv_5x5',
    'bilinear_conv_7x7',
    'trans_conv_3x3',
    'trans_conv_5x5',
    'trans_conv_7x7',
    'bilinear_additive_3x3',
    'bilinear_additive_5x5',
    'bilinear_additive_7x7',
    'depth_to_space_3x3',
    'depth_to_space_5x5',
    'depth_to_space_7x7',
    'factorized_reduce',
    #'none',
]


DOWNSAMPLE_METHOD = [
    'conv_downsample_3x3_dilation_1',
    'conv_downsample_3x3_dilation_2',
    'conv_downsample_5x5_dilation_1',
    'conv_downsample_5x5_dilation_2',
    'conv_downsample_7x7_dilation_1',
    'conv_downsample_7x7_dilation_2',
    'avg_pool_3x3',
    'avg_pool_5x5',
    'avg_pool_7x7',
    'max_pool_3x3',
    'max_pool_5x5',
    'max_pool_7x7',
    'factorized_reduce',
    #'none',
]



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
  - experiments are finished
  
"""

UPSAMPLE_PRIMITIVE = [
  'bilinear',
  'bicubic',
  'nearest',
  'pixel_shuffle',
  'trans_conv',
]

UPSAMPLE_CONV = [
  'conv',
  'trans_conv',
  'split_stack_sum', # additive
  'sep_conv',
  'identity',
]


KERNEL_SIZE = [
  '3x3',
  #'4x4',
  #'5x5',
  #'7x7',
]


DILATION_RATE = [
  '1',
  '2',
  '3',
]
