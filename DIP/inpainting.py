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
# from torchviz import make_dot, make_dot_from_trace
# from torchvision import transforms, utils
# from torch.utils.data import Dataset, DataLoader


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
    parser = argparse.ArgumentParser(description='NAS-DIP Denoising')

    parser.add_argument('--optimizer', dest='optimizer',default='adam', type=str)
    parser.add_argument('--num_iter', dest='num_iter', default=11000, type=int)
    parser.add_argument('--show_every', dest='show_every', default=50, type=int)
    parser.add_argument('--lr', dest='lr', default=0.001, type=float)
    parser.add_argument('--plot', dest='plot', default=False, type=bool)
    parser.add_argument('--noise_method', dest='noise_method',default='noise', type=str)
    parser.add_argument('--input_depth', dest='input_depth', default=32, type=int)
    parser.add_argument('--output_path', dest='output_path',default='results/restoration', type=str)
    parser.add_argument('--batch_size', dest='batch_size',default=1, type=int)
    parser.add_argument('--random_seed', dest='random_seed',default=0, type=int)
    parser.add_argument('--net', dest='net',default='default', type=str)
    parser.add_argument('--reg_noise_std', dest='reg_noise_std', default=0.03, type=float)
    parser.add_argument('--i_NAS', dest='i_NAS', default=-1, type=int)
    parser.add_argument('--save_png', dest='save_png', default=0, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    if args.net == 'default':
        global_path = args.output_path + '_' + args.net
    elif args.net == 'NAS':
        global_path = args.output_path + '_' + args.net + '_' + str(args.i_NAS)
    elif args.net == 'Multiscale':
        global_path = args.output_path + '_' + args.net + '_' + str(args.i_NAS)
    else:
        assert False, 'Please choose between default and NAS'

    # Creat the output_path if not exists
    if not os.path.exists(global_path):
        os.makedirs(global_path)

    # #batch x #iter
    PSNR_mat = np.empty((0, args.num_iter), dtype=np.float32)

    # Choose figure
    img_path_list = ['barbara', 'boat', 'house', 'Lena512', 'peppers256', 'Cameraman256', 'couple', 'fingerprint', 'hill', 'man', 'montage']
    psnr_gt_best_list = []

    for image_name in img_path_list:

        if args.save_png == 1 and not os.path.exists(os.path.join(global_path, image_name)):
            os.makedirs(os.path.join(global_path, image_name))

        # Choose figure
        img_path = 'data/inpainting/' + image_name + '_GT.png'

        # Load image
        img_pil, img_np = get_image(img_path, -1)
        img_np          = nn.ReflectionPad2d(1)(np_to_torch(img_np))[0].numpy()
        img_pil         = np_to_pil(img_np)

        img_mask    = get_bernoulli_mask(img_pil, 0.50)
        img_mask_np = pil_to_np(img_mask)

        img_masked  = img_np * img_mask_np
        mask_var    = np_to_torch(img_mask_np).type(dtype)

        # Visualization
        if args.plot:
            plot_image_grid([img_np, img_mask_np, img_mask_np * img_np], 3, 11);


        if args.net == 'default':
            from models.skip import skip
            net = skip(num_input_channels=args.input_depth,
                       num_output_channels=1,
                       num_channels_down=[128] * 5,
                       num_channels_up=[128] * 5,
                       num_channels_skip=[4] * 5,
                       upsample_mode='bilinear',
                       downsample_mode='stride',
                       need_sigmoid=True, 
                       need_bias=True, 
                       pad='reflection',
                       act_fun='LeakyReLU')

        elif args.net == 'NAS':
            from models.skip_search_up import skip
            if args.i_NAS in [249, 250, 251]:
                exit(1)
            net = skip(model_index=args.i_NAS,
                       num_input_channels=args.input_depth,
                       num_output_channels=1,
                       num_channels_down=[128] * 5,
                       num_channels_up=[128] * 5,
                       num_channels_skip=[4] * 5,
                       upsample_mode='bilinear',
                       downsample_mode='stride',
                       need_sigmoid=True, 
                       need_bias=True, 
                       pad='reflection',
                       act_fun='LeakyReLU')

        elif args.net == 'Multiscale':
            from models.cross_skip import skip
            from gen_skip_index import skip_index
            skip_connect = skip_index()
            net = skip(model_index=args.i_NAS,
                       skip_index=skip_connect,
                       num_input_channels=args.input_depth,
                       num_output_channels=1,
                       num_channels_down=[128] * 5,
                       num_channels_up=[128] * 5,
                       num_channels_skip=[4] * 5,
                       upsample_mode='bilinear',
                       downsample_mode='stride',
                       need_sigmoid=True, 
                       need_bias=True, 
                       pad='reflection',
                       act_fun='LeakyReLU')

        else:
            assert False, 'Please choose between default and NAS'

        net = net.type(dtype)

        # z torch.Size([1, 32, 512, 512])
        net_input = get_noise(args.input_depth, args.noise_method, img_np.shape[1:]).type(dtype).detach()


        # Loss
        mse = torch.nn.MSELoss().type(dtype)

        # x0
        img_var = np_to_torch(img_np).type(dtype)

        net_input_saved = net_input.detach().clone()
        noise           = net_input.detach().clone()

        last_net         = None
        psrn_masked_last = 0
        psnr_gt_best     = 0

        # Main
        i  = 0
        PSNR_list = []

        _t = {'im_detect' : Timer(), 'misc' : Timer()}

        def closure():

            global i, psrn_masked_last, last_net, net_input, psnr_gt_best, PSNR_list

            _t['im_detect'].tic()

            # Add variation
            if args.reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * args.reg_noise_std)
            #ipdb.set_trace()
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

            if i % args.show_every == 0:
                if args.save_png == 1:
                    out_np = torch_to_np(out)
                    cv2.imwrite(os.path.join(global_path, image_name, str(i) + '.png'),\
                    np.clip(out_np, 0, 1).transpose(1, 2, 0)[:,:,::-1] * 255)

                if args.plot:
                    plot_image_grid([np.clip(out_np, 0, 1)], factor=4, nrow=1)

            # Backtracking
            if args.plot and i % args.show_every == 0:
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

        PSNR_mat  = np.concatenate((PSNR_mat, np.array(PSNR_list).reshape(1,args.num_iter)), axis=0)
        pickle.dump( PSNR_mat, open( os.path.join(global_path, 'PSNR.pkl'), "wb" ) )

        psnr_gt_best_list.append(psnr_gt_best)

    print('Finish optimization\n')

    for idx, image_name in enumerate(img_path_list):
        print ('Image: %8s   PSNR: %.2f'  % (image_name, psnr_gt_best_list[idx]), '\n', end='')
    print ('Averaged PSNR: %.2f'  % (np.mean(psnr_gt_best_list)), '\n', end='')
