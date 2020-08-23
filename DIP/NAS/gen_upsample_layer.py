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

def gen_layer(C_in, C_out, model_index):
    
    swap = False

    """ Bilinear  """
    if model_index >= 0 and model_index <= 251:
      prim_index = model_index // 63
      model_index = model_index % 63
      conv_index = ((model_index // len(genotypes.ACTIVATION)) // len(genotypes.KERNEL_SIZE)) % len(genotypes.UPSAMPLE_PRIMITIVE)
      kernel_index = (model_index // len(genotypes.ACTIVATION)) % len(genotypes.KERNEL_SIZE)
      act_index = model_index % len(genotypes.ACTIVATION)

      if (model_index >= 60 and model_index <= 62):
        conv_index = 5

    """ DepthToSpace - Second """
    if model_index >= 252 and model_index <= 311:
      swap = True
      prim_index = (model_index - 63) // 63
      model_index = model_index % 63
      conv_index = ((model_index // len(genotypes.ACTIVATION)) // len(genotypes.KERNEL_SIZE)) % len(genotypes.UPSAMPLE_PRIMITIVE)
      kernel_index = (model_index // len(genotypes.ACTIVATION)) % len(genotypes.KERNEL_SIZE)
      act_index = model_index % len(genotypes.ACTIVATION)

    """ Transposed Convolution """
    if model_index >= 312 and model_index <= 323:
      prim_index = 4
      conv_index = 5
      kernel_index = (model_index // len(genotypes.ACTIVATION)) % len(genotypes.KERNEL_SIZE)
      act_index = model_index % len(genotypes.ACTIVATION)

    prim_op = genotypes.UPSAMPLE_PRIMITIVE[prim_index] # adjust the spatial size
    conv_op = genotypes.UPSAMPLE_CONV[conv_index] # adjust the spatial size
    kernel_size = genotypes.KERNEL_SIZE[kernel_index] # select the kernel size
    act_op = genotypes.ACTIVATION[act_index] # select the kernel size

    #print('prim op:', prim_op)
    #print('conv op:', conv_op)
    #print('kernel size:', kernel_size)
    #print('act op:', act_op)
    #return

    if prim_op == 'pixel_shuffle':
      if not swap:
        prim_op_layer = operations.UPSAMPLE_PRIMITIVE_OPS[prim_op](C_in=C_in, 
                                                                   C_out=C_out, 
                                                                   kernel_size=kernel_size,
                                                                   act_op=act_op)

        conv_op_layer = operations.UPSAMPLE_CONV_OPS[conv_op](C_in=int(C_in/4), 
                                                              C_out=C_out, 
                                                              kernel_size=kernel_size,
                                                              act_op=act_op)
        return nn.Sequential(prim_op_layer, conv_op_layer)

      else:
        conv_op_layer = operations.UPSAMPLE_CONV_OPS[conv_op](C_in=C_in, 
                                                              C_out=int(C_out*4), 
                                                              kernel_size=kernel_size,
                                                              act_op=act_op)

        prim_op_layer = operations.UPSAMPLE_PRIMITIVE_OPS[prim_op](C_in=C_in, 
                                                                   C_out=C_out, 
                                                                   kernel_size=kernel_size,
                                                                   act_op=act_op)
        return nn.Sequential(conv_op_layer, prim_op_layer)

    else:
      prim_op_layer = operations.UPSAMPLE_PRIMITIVE_OPS[prim_op](C_in=C_in, 
                                                                 C_out=C_out, 
                                                                 kernel_size=kernel_size,
                                                                 act_op=act_op)

      conv_op_layer = operations.UPSAMPLE_CONV_OPS[conv_op](C_in=C_in, 
                                                            C_out=C_out, 
                                                            kernel_size=kernel_size,
                                                            act_op=act_op)
      return nn.Sequential(prim_op_layer, conv_op_layer)

#for i in range(321):
#    gen_layer(0, 0, model_index=i)
#gen_layer(0, 0, model_index=189)
