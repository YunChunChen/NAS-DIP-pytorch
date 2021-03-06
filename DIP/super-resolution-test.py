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

from models.downsampler import Downsampler
from utils.sr_utils import *
from utils.timer import Timer

import torch
import torch.optim

def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr

def compare_psnr_y(x, y):
    return compare_psnr(rgb2ycbcr(x.transpose(1,2,0))[:,:,0], rgb2ycbcr(y.transpose(1,2,0))[:,:,0])

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.enabled       = True
torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = True
dtype = torch.cuda.FloatTensor

def parse_args():
    parser = argparse.ArgumentParser(description='NAS-DIP Super-resolution')

    parser.add_argument('--optimizer', dest='optimizer',default='adam', type=str)
    parser.add_argument('--num_iter', dest='num_iter', default=2000, type=int)
    parser.add_argument('--factor', dest='factor', default=4, type=int)
    parser.add_argument('--show_every', dest='show_every', default=100, type=int)
    parser.add_argument('--lr', dest='lr', default=0.01, type=float)
    parser.add_argument('--plot', dest='plot', default=False, type=bool)
    parser.add_argument('--noise_method', dest='noise_method',default='noise', type=str)
    parser.add_argument('--input_depth', dest='input_depth', default=32, type=int)
    parser.add_argument('--output_path', dest='output_path',default='results/sr', type=str)
    parser.add_argument('--random_seed', dest='random_seed',default=0, type=int)
    parser.add_argument('--net', dest='net',default='default', type=str)
    parser.add_argument('--reg_noise_std', dest='reg_noise_std', default=0.03, type=float)
    parser.add_argument('--i_NAS', dest='i_NAS', default=-1, type=int)
    parser.add_argument('--job_index', dest='job_index', default=1, type=int)
    parser.add_argument('--save_png', dest='save_png', default=0, type=int)
    parser.add_argument('--image_name', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    img_path = 'data/sr/' + args.image_name

    imgs = load_LR_HR_imgs_sr(img_path , -1, args.factor, 'CROP')

    from models.model_sr import Model
    net = Model()

    net = net.type(dtype)

    net_input = get_noise(args.input_depth, args.noise_method, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])).type(dtype).detach()

    mse = torch.nn.MSELoss().type(dtype)

    img_LR_var      = np_to_torch(imgs['LR_np']).type(dtype)
    downsampler     = Downsampler(n_planes=3, factor=args.factor, kernel_type='lanczos2', phase=0.5, preserve_size=True).type(dtype)

    psnr_gt_best    = 0

    i  = 0
    PSNR_list = []

    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    def closure():

        global i, net_input, psnr_gt_best, PSNR_list

        _t['im_detect'].tic()

        if args.reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * args.reg_noise_std)

        out_HR = net(net_input)      #torch.Size([1, 3, tH, tW]): x
        out_LR = downsampler(out_HR) #torch.Size([1, 3, H, W])

        total_loss = mse(out_LR, img_LR_var)
        total_loss.backward()

        q1 = torch_to_np(out_HR)[:3].sum(0)
        t1 = np.where(q1.sum(0) > 0)[0]
        t2 = np.where(q1.sum(1) > 0)[0]
        psnr_HR = compare_psnr_y(imgs['HR_np'][:3,t2[0] + 4:t2[-1]-4,t1[0] + 4:t1[-1] - 4], \
                           torch_to_np(out_HR)[:3,t2[0] + 4:t2[-1]-4,t1[0] + 4:t1[-1] - 4])
        PSNR_list.append(psnr_HR)

        if psnr_HR > psnr_gt_best:
            psnr_gt_best = psnr_HR

        _t['im_detect'].toc()

        print ('Iteration %05d    Loss %f   PSNR_HR %.3f    Time %.3f'  % (i, total_loss.item(), psnr_HR, _t['im_detect'].total_time), '\r', end='')
        if  i % args.show_every == 0:
            if args.save_png == 1:
                out_HR_np = torch_to_np(out_HR)
                cv2.imwrite(os.path.join(global_path, image_name, str(i) + '.png'),\
                np.clip(out_HR_np, 0, 1).transpose(1, 2, 0)[:,:,::-1] * 255)

            if args.plot:
                plot_image_grid([np.clip(out_HR_np, 0, 1)], factor=4, nrow=1)

        i += 1

        return total_loss

    net_input_saved = net_input.detach().clone()
    noise           = net_input.detach().clone()

    p = get_params('net', net, net_input)
    optimize(args.optimizer, p, closure, args.lr, args.num_iter)

    psnr_gt_best_list.append(psnr_gt_best)

    print('Finish optimization')