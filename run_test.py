# -*- coding: utf-8 -*-
import argparse
import torch

from test_S2M2_im import test

use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.cuda.current_device()
else:
    device = torch.device('cpu')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=1,
                            help='number of support examples per training class')
    parser.add_argument('--classifier', type=str, default='ADV-CE', help='use ADV-CE or ADV-SVM or ADV-CE-TIM')
    parser.add_argument('--tukey', type=int, default=0,
                            help='applye turkey transform to feature or not')
    parser.add_argument('--l2normalization', type=int, default=0,
                            help='apply unit norm normalization to feautre or not')
    parser.add_argument('--nnk', type=int, default=9,
                            help='maximal possible nodes to visit in random walk')
    parser.add_argument('--gamma', type=float, default=100,
                            help='gamma for kernel in ADV-SVM. (=sigma in RBF kernel in the paper)')
    parser.add_argument('--sigma_bias', type=float, default=0.0,
                            help='sigma bias in vicinal distribution. Added to All element in the Sigma matrix estimated by random walk')
    parser.add_argument('--lr', type=float, default=0.1,
                            help='inner learning rate in ADV-CE')
    parser.add_argument('--step', type=int, default=100,
                            help='inner step number in ADV-CE')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                            help='weigth decay in ADV-LR')
    # parser.add_argument('--sdiag', type=float, default=0.003,
    #                         help='sigma diag in vicinal LR')
    parser.add_argument('--proto_int', type=int, default=1,
                            help='use prototype as classifier weights initialization in ADV-CE')
    parser.add_argument('--rw_step', type=int, default=1,
                            help='num of steps in random walk')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='temperature in random walk steps')
    parser.add_argument('--tim_para', type=float, default=5.,
                        help='the hyper parameter of TIM loss')


    args = parser.parse_args()


    cls_cfg = {}
    if args.classifier=='ADV-CE':  ### CE + VRM
        cls_cfg = {'device': device, 'n_shot': args.shot, 'n_ways': 5, 'feat_dim': 640,
                   'tukey': bool(args.tukey), 'l2n': bool(args.l2normalization),
                   'n_step': args.step, 'inner_lr': args.lr, 'proto_init': args.proto_int, 'weight_decay': args.weight_decay,
                   'rw_step': args.rw_step, 'nn_k': args.nnk, 'tau': args.tau, 'sigma_bias': args.sigma_bias}

        print(cls_cfg)
    elif args.classifier=='ADV-CE-TIM':  ### CE + VRM + TIM loss
        cls_cfg = {'device': device, 'n_shot': args.shot, 'n_ways': 5, 'feat_dim': 640,
                   'tukey': bool(args.tukey), 'l2n': bool(args.l2normalization),
                   'n_step': args.step, 'inner_lr': args.lr, 'proto_init': args.proto_int, 'weight_decay': args.weight_decay,
                   'rw_step': args.rw_step, 'nn_k': args.nnk, 'tau': args.tau, 'sigma_bias': args.sigma_bias,
                   'tim_para': args.tim_para}

        print(cls_cfg)

    elif args.classifier=='ADV-SVM':  ### Kernel SVM + VRM
        cls_cfg = {'device': device, 'n_shot': args.shot, 'n_ways': 5, 'feat_dim': 640,
                   'tukey': bool(args.tukey), 'l2n': bool(args.l2normalization),
                   'rw_step': args.rw_step, 'nn_k': args.nnk, 'tau': args.tau, 'sigma_bias': args.sigma_bias,
                   'gamma': args.gamma}

    test(cls_cfg, args.classifier)



