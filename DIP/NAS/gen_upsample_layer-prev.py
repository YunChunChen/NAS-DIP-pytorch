import random
import torch
import torch.nn as nn

try:
    from NAS import genotypes
except ImportError:
    import genotypes

try:
    from NAS import model
except ImportError:
    import model

try:
    from NAS import operations
except ImportError:
    import operations

#random.seed(1)
#torch.manual_seed(1)


# def gen_layer(C_in, C_out, use_act, model_index):

#     prim_index = int(model_index / 5)
#     conv_index = model_index % 5
#     kernel_index = 0

#     if model_index >= 15 and model_index < 23:
#         prim_index = 3
#         conv_index = (model_index - 3) % 4

#     prim_op = genotypes.UPSAMPLE_PRIMITIVE[prim_index] # adjust the spatial size
#     conv_op = genotypes.UPSAMPLE_CONV[conv_index] # adjust the spatial size
#     kernel_size = genotypes.KERNEL_SIZE[kernel_index] # select the kernel size

#     if prim_op == 'trans_conv': # only one single layer
#       prim_op_layer = operations.UPSAMPLE_PRIMITIVE_OPS[prim_op](C_in=C_in, 
#                                                                  C_out=C_out, 
#                                                                  kernel_size=kernel_size, 
#                                                                  use_act=use_act)
#       return prim_op_layer

#     elif prim_op == 'pixel_shuffle':
#       if model_index >= 15 and model_index < 23:
#         prim_op_layer = operations.UPSAMPLE_PRIMITIVE_OPS[prim_op](C_in=C_in, 
#                                                                    C_out=C_out, 
#                                                                    kernel_size=kernel_size,
#                                                                    use_act=use_act)

#         conv_op_layer = operations.UPSAMPLE_CONV_OPS[conv_op](C_in=int(C_in/4), 
#                                                               C_out=C_out, 
#                                                               kernel_size=kernel_size,
#                                                               use_act=use_act)
#         layer = nn.Sequential(prim_op_layer, conv_op_layer)

#       else:
#         conv_op_layer = operations.UPSAMPLE_CONV_OPS[conv_op](C_in=C_in, 
#                                                               C_out=int(C_out*4), 
#                                                               kernel_size=kernel_size,
#                                                               use_act=use_act)

#         prim_op_layer = operations.UPSAMPLE_PRIMITIVE_OPS[prim_op](C_in=C_in, 
#                                                                    C_out=C_out, 
#                                                                    kernel_size=kernel_size,
#                                                                    use_act=use_act)
#         layer = nn.Sequential(conv_op_layer, prim_op_layer)

#     else:
#       prim_op_layer = operations.UPSAMPLE_PRIMITIVE_OPS[prim_op](C_in=C_in, 
#                                                                  C_out=C_out, 
#                                                                  kernel_size=kernel_size,
#                                                                  use_act=use_act)

#       conv_op_layer = operations.UPSAMPLE_CONV_OPS[conv_op](C_in=C_in, 
#                                                             C_out=C_out, 
#                                                             kernel_size=kernel_size,
#                                                             use_act=use_act)

#       layer = nn.Sequential(prim_op_layer, conv_op_layer)

#     return layer


def gen_layer(C_in, C_out, use_act, model_index):

    prim_index = int(model_index / 5)
    conv_index = model_index % 5
    kernel_index = 0

    prim_op = genotypes.UPSAMPLE_PRIMITIVE[prim_index] # adjust the spatial size
    conv_op = genotypes.UPSAMPLE_CONV[conv_index] # adjust the spatial size
    kernel_size = genotypes.KERNEL_SIZE[kernel_index] # select the kernel size
  
    prim_op_layer = operations.UPSAMPLE_PRIMITIVE_OPS[prim_op](C_in=C_in, 
                                                               C_out=C_out, 
                                                               kernel_size=kernel_size,
                                                               use_act=use_act)

    conv_op_layer = operations.UPSAMPLE_CONV_OPS[conv_op](C_in=C_in, 
                                                          C_out=C_out, 
                                                          kernel_size=kernel_size,
                                                          use_act=use_act)

    layer = nn.Sequential(conv_op_layer, prim_op_layer)

    return layer
