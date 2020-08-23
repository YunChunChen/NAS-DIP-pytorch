import torch
import torch.nn as nn


class SepConv(nn.Module):
    
  def __init__(self, 
               stride=2, 
               mode='bicubic'):

    super(SepConv, self).__init__()

    self.op = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bicubic'),
      nn.ReLU(inplace=False),
      )

  def forward(self, x):
    return self.op(x)



data = torch.rand((4, 32, 256, 256)).cuda()

net = SepConv().cuda()

out = net(data)

print('input dim:', data.shape)
print('output dim:', out.shape)
