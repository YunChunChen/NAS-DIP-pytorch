import random
import torch

try:
    from NAS import genotypes
except ImportError:
    import genotypes

try:
    from NAS import model
except ImportError:
    import model


#random.seed(1)
#torch.manual_seed(1)


def gen_ops(num_of_ops):
    
    prim_ops_list = []
    conv_list = []
    kernel_list = []
    ops_concat = list(range(2, 2+int(num_of_ops/2)))
 
    # number of operations for the upsampling
    for i in range(num_of_ops):

        prim_op_idx = random.randint(0, len(genotypes.UPSAMPLE_PRIMITIVE)-1)
        sampled_prim_op = genotypes.UPSAMPLE_PRIMITIVE[prim_op_idx] # adjust the spatial size

        conv_op_idx = random.randint(0, len(genotypes.UPSAMPLE_CONV)-1)
        sampled_conv_op = genotypes.UPSAMPLE_CONV[conv_op_idx] # adjust the num of channels

        kernel_idx = random.randint(0, len(genotypes.KERNEL_SIZE)-1)
        sampled_kernel = genotypes.KERNEL_SIZE[kernel_idx] # kernel size

        max_node_id = int((i+2)/2) # upper bound of the input id
        input_id = random.randint(0, max_node_id) # sample an input id

        prim_ops_list.append((sampled_prim_op, input_id))
        conv_list.append((sampled_conv_op, input_id))
        
        if i == 0:
            kernel_list.append(sampled_kernel)

        if input_id in ops_concat:
            ops_concat.remove(input_id)

    return prim_ops_list, conv_list, kernel_list, ops_concat



def random_search(num_of_ops):

    upsample_prim_method, upsample_conv, upsample_kernel, upsample_concat = gen_ops(num_of_ops)

    DIPGenotype = genotypes.Genotype(
        upsample_prim_method,
        upsample_conv,
        upsample_kernel,
        upsample_concat,
    )

    return DIPGenotype



def model_gen(search_type='random_search', num_input_channel=32, num_of_nodes=4):

    if search_type == 'random_search':
        num_of_ops = num_of_nodes * 2
        sampled_genotype = random_search(num_of_ops=num_of_ops) # a genotype

        net = model.NetworkDIP(genotype=sampled_genotype, 
                               num_input_channel=num_input_channel)
        
        return net, sampled_genotype
