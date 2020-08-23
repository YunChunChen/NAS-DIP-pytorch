from __future__ import print_function
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import cv2
import ipdb
import random
import pickle
import argparse
import numpy as np
from skimage.measure import compare_psnr

from utils.denoising_utils import *
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
    parser = argparse.ArgumentParser(description='NAS-DIP Denoising')

    parser.add_argument('--optimizer', dest='optimizer',default='adam', type=str)
    parser.add_argument('--num_iter', dest='num_iter', default=3000, type=int)
    parser.add_argument('--show_every', dest='show_every', default=50, type=int)
    parser.add_argument('--lr', dest='lr', default=0.01, type=float)
    parser.add_argument('--plot', dest='plot', default=False, type=bool)
    parser.add_argument('--noise_method', dest='noise_method',default='noise', type=str)
    parser.add_argument('--input_depth', dest='input_depth', default=32, type=int)
    parser.add_argument('--output_path', dest='output_path',default='results/denoising', type=str)
    parser.add_argument('--batch_size', dest='batch_size',default=1, type=int)
    parser.add_argument('--reg_noise_std', dest='reg_noise_std', default=1./30., type=float)
    parser.add_argument('--sigma', dest='sigma', default=25, type=float)
    parser.add_argument('--save_png', dest='save_png', default=0, type=int)
    parser.add_argument('--exp_weight', dest='exp_weight', default=0.99, type=float)
    parser.add_argument('--image_name', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    img_path = 'data/denoising/' + args.image_name

    img_pil = crop_image(get_image(img_path, -1)[0], 32)
    img_np  = pil_to_np(img_pil)

    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, args.sigma / 255.)

    from models.model_denoising import Model
    net = Model()

    net = net.type(dtype)

    net_input = get_noise(args.input_depth, args.noise_method, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

    mse = torch.nn.MSELoss().type(dtype)

    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

    net_input_saved = net_input.detach().clone()
    noise           = net_input.detach().clone()
    out_avg         = None
    last_net        = None
    psnr_noisy_last = 0
    psnr_gt_best    = 0

    i  = 0
    PSNR_list = []

    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    def closure():

        global i, out_avg, psnr_noisy_last, last_net, net_input, psnr_gt_best, PSNR_list

        _t['im_detect'].tic()

        if args.reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * args.reg_noise_std)

        out = net(net_input)

        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * args.exp_weight + out.detach() * (1 - args.exp_weight)

        total_loss = mse(out, img_noisy_torch)
        total_loss.backward()

        psnr_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
        psnr_gt    = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 

        PSNR_list.append(psnr_gt)

        if psnr_gt > psnr_gt_best:
            psnr_gt_best = psnr_gt

        _t['im_detect'].toc()

        print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSNR_gt: %f    Time %.3f'  % (i, total_loss.item(), psnr_noisy, psnr_gt, _t['im_detect'].total_time), '\n', end='')

        if  i % args.show_every == 0:
            out_np = torch_to_np(out)
            if args.save_png == 1:
                cv2.imwrite(os.path.join(global_path, image_name, str(i) + '.png'),\
                np.clip(out_np, 0, 1).transpose(1, 2, 0)[:,:,::-1] * 255)

            if args.plot:
                plot_image_grid([np.clip(out_np, 0, 1)], factor=4, nrow=1)

        if i % args.show_every:
            if psnr_noisy - psnr_noisy_last < -5:
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())

                return total_loss*0
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psnr_noisy_last = psnr_noisy

        i += 1

        return total_loss

    p = get_params('net', net, net_input)
    optimize(args.optimizer, p, closure, args.lr, args.num_iter)

    PSNR_mat  = np.concatenate((PSNR_mat, np.array(PSNR_list).reshape(1,args.num_iter)), axis=0)
    pickle.dump( PSNR_mat, open( os.path.join(global_path, 'PSNR.pkl'), "wb" ) )

    psnr_gt_best_list.append(psnr_gt_best)

    print('Finish optimization\n')