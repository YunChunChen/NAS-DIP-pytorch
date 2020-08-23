from __future__ import print_function
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt

import os
import cv2
import argparse
import numpy as np
from skimage.measure import compare_psnr

from utils.inpainting_utils import *
from utils.timer import Timer

import torch
import torch.optim

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.enabled       = True
torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = True
dtype = torch.cuda.FloatTensor

def parse_args():
    parser = argparse.ArgumentParser(description='NAS-DIP Inpainting')

    parser.add_argument('--optimizer', dest='optimizer',default='adam', type=str)
    parser.add_argument('--num_iter', dest='num_iter', default=11000, type=int)
    parser.add_argument('--show_every', dest='show_every', default=50, type=int)
    parser.add_argument('--lr', dest='lr', default=0.001, type=float)
    parser.add_argument('--plot', dest='plot', default=False, type=bool)
    parser.add_argument('--noise_method', dest='noise_method',default='noise', type=str)
    parser.add_argument('--input_depth', dest='input_depth', default=32, type=int)
    parser.add_argument('--output_path', dest='output_path',default='results/inpainting', type=str)
    parser.add_argument('--reg_noise_std', dest='reg_noise_std', default=0.03, type=float)
    parser.add_argument('--image_name', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    img_path = 'data/inpainting/' + args.image_name

    img_pil, img_np = get_image(img_path, -1)
    img_pil         = np_to_pil(img_np)

    img_mask    = get_bernoulli_mask(img_pil, 0.50)
    img_mask_np = pil_to_np(img_mask)

    img_masked  = img_np * img_mask_np
    mask_var    = np_to_torch(img_mask_np).type(dtype)

    # Visualization
    if args.plot:
        plot_image_grid([img_np, img_mask_np, img_mask_np * img_np], 3, 11);

    from models.model_inpainting import Model
    net = Model()

    net = net.type(dtype)

    net_input = get_noise(args.input_depth, args.noise_method, img_np.shape[1:]).type(dtype).detach()

    mse = torch.nn.MSELoss().type(dtype)

    img_var = np_to_torch(img_np).type(dtype)

    net_input_saved = net_input.detach().clone()
    noise           = net_input.detach().clone()

    last_net         = None
    psrn_masked_last = 0
    psnr_gt_best     = 0

    # Main
    i  = 0
    PSNR_list = []

    _t = {'im_detect': Timer(), 'misc': Timer()}

    def closure():

        global i, psrn_masked_last, last_net, net_input, psnr_gt_best, PSNR_list

        _t['im_detect'].tic()

        # Add variation
        if args.reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * args.reg_noise_std)
        out = net(net_input)

        total_loss = mse(out * mask_var, img_var * mask_var)
        total_loss.backward()

        psrn_masked = compare_psnr(img_masked, out.detach().cpu().numpy()[0] * img_mask_np)
        psrn        = compare_psnr(img_np,     out.detach().cpu().numpy()[0])

        PSNR_list.append(psrn)

        if psrn > psnr_gt_best:
            psnr_gt_best = psrn

        _t['im_detect'].toc()

        print ('Iteration %05d    Loss %f   PSNR_masked %f PSNR %f    Time %.3f'  % (i, total_loss.item(), psrn_masked, psrn, _t['im_detect'].total_time), '\r', end='')

        # Backtracking
        if args.plot and i % args.show_every == 0:
            
            plot_image_grid([np.clip(out_np, 0, 1)], factor=4, nrow=1)

            out_np = torch_to_np(out)

            if psrn_masked - psrn_masked_last < -5:
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())

                return total_loss*0
            else:
                last_net = [x.cpu() for x in net.parameters()]
                psrn_masked_last = psrn_masked

        i += 1

        return total_loss

    p = get_params('net', net, net_input)
    optimize(args.optimizer, p, closure, args.lr, args.num_iter)

    print('Finish optimization\n')
