import torch
import torch.nn as nn
from torch.autograd import Variable

try:
    from NAS import operations
except ImportError:
    import operations

try:
    from NAS import utils
except ImportError:
    import utils


class Cell(nn.Module):

  def __init__(self, 
               genotype, 
               C_prev, 
               C_curr,
               C_prev_prev=None, 
               op_type='downsample'):

    super(Cell, self).__init__()
    
    self.op_type = op_type
    
    if self.op_type == 'downsample':
        self.preprocess0 = operations.ReLUConvBN(C_prev, C_curr, 1, 1, 0)
        conv_op_names, indices = zip(*genotype.downsample_conv)
        op_names, _ = zip(*genotype.downsample_method)
        concat = genotype.downsample_concat
    else:
        self.preprocess0 = operations.ReLUConvBN(C_prev, C_curr, 1, 1, 0)
        if C_prev_prev is not None:
            self.preprocess1 = operations.ReLUConvBN(C_prev_prev, C_curr, 1, 1, 0)
        conv_op_names, indices = zip(*genotype.upsample_conv)
        op_names, _ = zip(*genotype.upsample_method)
        concat = genotype.upsample_concat

    self._compile(C_curr=C_curr, 
                  conv_op_names=conv_op_names, 
                  op_names=op_names, 
                  indices=indices, 
                  concat=concat)


  def _compile(self, C_curr, conv_op_names, op_names, indices, concat):
      assert len(op_names) == len(indices)
      assert len(conv_op_names) == len(indices)

      self.num_of_nodes = len(op_names) // 2
      self._concat = concat
      self.multiplier = len(concat)

      self._ops = nn.ModuleList()

      for index in range(len(op_names)):

          if self.op_type == 'downsample':
              downsample_name = op_names[index]
              conv_name = conv_op_names[index]

              #stride = 2 if index < 2 else 1
              stride = 2 if indices[index] < 2 else 1

              downsample_op = operations.DOWNSAMPLE_OPS[downsample_name](C_in=C_curr, stride=stride)
              conv_op = operations.CONV_OPS[conv_name](C_in=C_curr, C_out=C_curr, affine=True)

              #print('\n\n[Downsample Op]:', downsample_op)
              #print('\n[Conv Op]:', conv_op)

              op = nn.Sequential(downsample_op, conv_op)

              #print('\n[combined Op]:', op)

          else: # upsample
              upsample_name = op_names[index]
              conv_name = conv_op_names[index]

              #stride = 2 if index < 2 else 1
              stride = 2 if indices[index] < 2 else 1

              upsample_op = operations.UPSAMPLE_OPS[upsample_name](C_in=C_curr, stride=stride)
              conv_op = operations.CONV_OPS[conv_name](C_in=C_curr, C_out=C_curr, affine=True)

              #print('\n\n[Upsample Op]:', upsample_op)
              #print('\n[Conv Op]:', conv_op)

              op = nn.Sequential(upsample_op, conv_op)

              #print('\n[combined Op]:', op)

          self._ops += [op]

      self._indices = indices


  def forward(self, s0, drop_prob, s1=None):
      
      #print('[Cell] before s0 shape:', s0.shape)
      s0 = self.preprocess0(s0) # C_prev
      #print('[Cell] after s0 shape:', s0.shape, '\n\n')
      if s1 is None:
          s1 = s0
      else:
          s1 = self.preprocess1(s1)

      states = [s0, s1]
      for i in range(self.num_of_nodes):
          h1 = states[self._indices[2*i]]
          h2 = states[self._indices[2*i+1]]
          op1 = self._ops[2*i]
          op2 = self._ops[2*i+1]
          h1 = op1(h1)
          h2 = op2(h2)
          if self.training and drop_prob > 0.:
              if not isinstance(op1, operations.Identity):
                  h1 = utils.drop_path(h1, drop_prob)
              if not isinstance(op2, operations.Identity):
                  h2 = utils.drop_path(h2, drop_prob)
          s = h1 + h2
          states += [s]

      out = torch.cat([states[i] for i in self._concat], dim=1)

      #print('[Cell] out dim:', out.shape)

      return out



class NetworkDIP(nn.Module):

  def __init__(self, 
               genotype,
               num_input_channel=3,
               num_output_channel=3,
               concat_x=False,
               need_bias=True,
               norm_layer=nn.InstanceNorm2d,
               pad='zero',
               filters=[64, 128, 256, 512, 1024], 
               init_filters=3, # for the airplane case
               feature_scale=4,
               drop_path_prob=0.2):

    super(NetworkDIP, self).__init__()

    self._layers = len(filters)
    self.drop_path_prob = drop_path_prob

    filters = [x // feature_scale for x in filters]

    stem_output_channel = filters[0] if not concat_x else filters[0] - num_input_channel # stem's output channel

    self.stem = Stem(num_input_channel=num_input_channel,
                     num_output_channel=stem_output_channel,
                     norm_layer=norm_layer,
                     need_bias=need_bias,
                     pad=pad)

    self.cells = nn.ModuleList()

    """ Initializa downsample cells first """
    op_type = 'downsample'
    C_prev = stem_output_channel # same as stem's output channel
    for i in range(self._layers):
        C_curr = filters[i]
        cell = Cell(genotype, C_prev=C_prev, C_curr=C_curr, op_type=op_type)
        self.cells += [cell]
        C_prev = cell.multiplier * C_curr


    """ Initializa upsample cells first """
    op_type = 'upsample'
    up_mode = genotype.upsample_method
    C_prev_prev = None
    for i in range(self._layers-1, -1, -1):
        
        #print('[NetworkDIP] Upsample multiplier, filter:', cell.multiplier, filters[i])

        C_prev = cell.multiplier * filters[i]
        if i > 0:
            C_curr = filters[i-1]
        else:
            C_curr = C_prev // 2 # output channel of the NetworkDIP


        #print('[NetworkDIP] C_prev_prev, C_prev, C_curr:', C_prev_prev, C_prev, C_curr, '\n\n')
        cell = Cell(genotype, C_prev_prev=C_prev_prev, C_prev=C_prev, C_curr=C_curr, op_type=op_type)
        self.cells += [cell]
        C_prev_prev = self.cells[i-1].multiplier * filters[i-1]

    C_curr = cell.multiplier * C_curr

    self.last_layer = nn.Conv2d(in_channels=C_curr, out_channels=num_output_channel, kernel_size=1)
    self.last_activ = nn.Sigmoid()


  def forward(self, data):

      s0 = self.stem(data)

      output_list = []

      for i, cell in enumerate(self.cells):
          if i < len(self.cells) / 2 + 1: 
              # no skip connection (encoder part and the first cell in the decoder)
              #s0 = cell(s0, drop_prob=self.drop_path_prob)
              s0 = cell(s0, drop_prob=0)
              output_list.append(s0)
          else:
              s1 = output_list[len(self.cells) - 1 - i]
              #s0 = cell(s0=s0, s1=s1, drop_prob=self.drop_path_prob)
              s0 = cell(s0=s0, s1=s1, drop_prob=0)

      s0 = self.last_layer(s0)
      s0 = self.last_activ(s0)

      return s0



class Stem(nn.Module):

    def __init__(self, 
                 num_input_channel,
                 num_output_channel,
                 norm_layer, 
                 need_bias, 
                 pad):

        super(Stem, self).__init__()

        if norm_layer is not None:
            self.conv1= nn.Sequential(
                conv(num_input_channel, num_output_channel, 3, bias=need_bias, pad=pad),
                norm_layer(num_output_channel),
                nn.ReLU(),
            )

            self.conv2= nn.Sequential(
                conv(num_output_channel, num_output_channel, 3, bias=need_bias, pad=pad),
                norm_layer(num_output_channel),
                nn.ReLU(),
            )

        else:
            self.conv1= nn.Sequential(
                conv(num_input_channel, num_output_channel, 3, bias=need_bias, pad=pad),
                nn.ReLU(),
            )

            self.conv2= nn.Sequential(
                conv(num_output_channel, num_output_channel, 3, bias=need_bias, pad=pad),
                nn.ReLU(),
            )


    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):

    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)

        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)

        elif downsample_mode in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, 
                                      kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)

    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])

    return nn.Sequential(*layers)
