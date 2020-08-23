import torch
from torch.autograd import Variable


def drop_path(x, drop_prob, use_cuda=True):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        #mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        if use_cuda:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)).cuda()
        else:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x
