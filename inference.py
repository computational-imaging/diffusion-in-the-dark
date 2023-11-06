# one model that takes different resolutions of inputs
import argparse
import torch
import os
import dnnlib
import pandas as pd
import numpy as np
import pickle
import time
import kornia.contrib as kc
import torch.nn.functional as F
import training.dataset as dataset
import matplotlib.pyplot as plt
import sys

import lpips
import skimage.io
from training.constants import *
from training.sample import resize, sampling_loop
import training.utils as utils
import training.resizer as resizer

torch.manual_seed(0)
device = torch.device('cuda')

f0 = '00068-lol-fullres-bos-ddpmpp-edm-linear--res32x32-noiseTrue-paintnormFalse-scalenormTrue-lpipsTrue-gpus1-batch148-fp32/network-snapshot-001394.pkl'

fpaths = {'lowlight': '../data/lowLightDataset256',
          'lol': '../data/LOL/eval15_256',
          }

subfolder_dict = {'lowlight': {'gt': 'gt/test', 'input': 'input/test/1'},
                  'lol': {'gt': 'high', 'input': 'low'}}

runs_dir = 'runs'

def t2n(img):
    return (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def make_batches(fnames, batch_size):
    batch_fnames = []
    temp = []
    for i, fname in enumerate(fnames):
        temp.append(fname)
        if (i+1) % batch_size == 0:
            batch_fnames.append(temp)
            temp = []
    if len(temp) != 0:
        batch_fnames.append(temp)
    return batch_fnames


def sampling_loop_ilvr(net, batch_size, cond_img, ilvr_img, loop_num, device=torch.device('cuda:0'),
                       num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
                      S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):

    def il_resize(img, out_shape):
        return resizer.resize(img, out_shape=out_shape)

    b, c, h, w = cond_img.shape
    ref = ilvr_img

    # Pick latents and labels.
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[
            torch.randint(net.label_dim, size=[batch_size], device=device)]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]

    x_arr = x_next.clone()[None, ...]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.

        denoised = net(x_hat, cond_img, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, cond_img, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # # # q sample with reference image we want low frequency information from
        if i < loop_num:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            ref_hat = ref + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(ref)
            lr_next = il_resize(il_resize(x_next, out_shape=(h//2, w//2)), out_shape=(h,w))
            lr_ref = il_resize(il_resize(ref_hat, out_shape=(h//2, w//2)), out_shape=(h,w))
            x_next = x_next - lr_next + lr_ref

    return x_next, x_arr


def scale_loop(net, diff_img, cond_img_fr, from_size, to_size, diff0, he, loop_num, patch_size=(32, 32), scale_norm=False, self_norm=False,
               sampling_fn=sampling_loop_ilvr, batch_size=4, device=device):
    dark_down = resize(cond_img_fr, from_size=(256, 256), to_size=to_size)
    clean_lr  = resize(diff_img, from_size=from_size, to_size=to_size)
    diff0     = resize(diff0, from_size=patch_size, to_size=to_size)
    he        = resize(he, from_size=(256, 256), to_size=(to_size))


    imgs = torch.cat([dark_down, clean_lr, diff0, he], dim=1)
    pad = kc.compute_padding(original_size=to_size, window_size=patch_size)
    assert sum(list(pad)) == 0
    img_patches = kc.extract_tensor_patches(imgs, window_size=patch_size, stride=patch_size, padding=pad)

    diff_patches = []
    for i in range(0, img_patches.shape[1]):
        if scale_norm:
            input_patches = img_patches[:, i, 0:9]
        else:
            input_patches = img_patches[:, i, 0:6]

        if self_norm:
            input_patches = torch.cat([input_patches, img_patches[:, i, 9:12]], dim=1)

        ilvr_img = img_patches[:, i, 3:6]
        x_next, x_arr = sampling_fn(net, batch_size, cond_img=input_patches, ilvr_img=ilvr_img, loop_num=loop_num, device=device)
        diff_patches.append(x_next.unsqueeze(1))

    diff_patches = torch.cat(diff_patches, dim=1)
    diff_img = kc.combine_tensor_patches(diff_patches, original_size=to_size, window_size=patch_size, stride=patch_size, unpadding=pad)
    return diff_img


def main(args):
    input_path = f'{fpaths[args.dataset]}/{subfolder_dict[args.dataset]["input"]}'
    gt_path = f'{fpaths[args.dataset]}/{subfolder_dict[args.dataset]["gt"]}'

    fnames = sorted([f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))])

    with dnnlib.util.open_url(f'{runs_dir}/{f0}') as f:
        f0net = pickle.load(f)['ema'].to(device)

    self_norm = ('henormTrue' in f0)
    scale_norm = ('scalenormTrue' in f0)

    net_name = f0.split('-')[0]

    mean = LOL_LIGHT_LINEAR_MEAN
    std = LOL_LIGHT_LINEAR_STD


    batch_fnames = make_batches(fnames, batch_size=args.batch_size)
    net_name += '_ilvr'

    pd_name = f'results/rebuttal/{args.dataset}/{net_name}_results'

    os.makedirs(f'{pd_name}_f0', exist_ok=True)
    os.makedirs(f'{pd_name}_f1', exist_ok=True)
    os.makedirs(f'{pd_name}_f2', exist_ok=True)
    os.makedirs(f'{pd_name}_f3', exist_ok=True)
    stats = []
    total_time = 0.0

    for i, batch_fname in enumerate(batch_fnames):
        print(f'Processing {batch_fname}')

        for j in range(len(batch_fname)):
            fname = batch_fname[j]
            input = skimage.io.imread(f'{input_path}/{fname}')
            gt    = skimage.io.imread(f'{gt_path}/{fname}')

            input_he = (utils.srgb_to_linear_np(input / 255.0) * 255.0).astype(np.uint8)
            input_he = utils.equalize_histogram(input_he)
            input = torch.from_numpy(input).permute(2, 0, 1)[None, ...] / 255.0
            input = utils.srgb_to_linear(input)

            gt    = torch.from_numpy(gt).permute(2, 0, 1)[None, ...] / 255.0

            input_he = torch.from_numpy(input_he).permute(2, 0, 1)[None, ...] / 255.0

            gt_img = gt if j == 0 else torch.cat([gt_img, gt], dim=0)
            input_img = input if j == 0 else torch.cat([input_img, input], dim=0)
            he_img = input_he if j == 0 else torch.cat([he_img, input_he], dim=0)

        gt_img    = gt_img.to(device)
        input_img = input_img.to(device)
        he_img = he_img.to(device)

        input_img = utils.normalize(input_img ** (1/4), means=LOL_DARK_LINEAR_MEAN, stds=LOL_DARK_LINEAR_STD) / 2.0
        he_img = utils.normalize(he_img, means=LOL_DARK_HE_LINEAR_MEAN, stds=LOL_DARK_HE_LINEAR_STD)

        batch_size = len(batch_fname)

        f0_input = resize(input_img, from_size=(256, 256), to_size=(32, 32))
        f0_input = torch.cat([f0_input, f0_input], dim=1) if not scale_norm else torch.cat([f0_input, f0_input, f0_input], dim=1)

        if self_norm:
            he_resize = resize(he_img, from_size=(256, 256), to_size=(32, 32))
            f0_input = torch.cat([f0_input, he_resize], dim=1)

        start_time = time.time()
        diff0, x_arr = sampling_loop(f0net, batch_size, f0_input, device=device)

        diff1 = scale_loop(f0net, diff0, input_img, from_size=(32, 32), to_size=(64, 64), diff0=diff0, he=he_img, loop_num=6,
                           scale_norm=scale_norm, self_norm=self_norm, batch_size=batch_size)
        diff2 = scale_loop(f0net, diff1, input_img, from_size=(64, 64), to_size=(128, 128), diff0=diff0, he=he_img, loop_num=6,
                           scale_norm=scale_norm, self_norm=self_norm, batch_size=batch_size)
        diff3 = scale_loop(f0net, diff2, input_img, from_size=(128, 128), to_size=(256, 256), diff0=diff0, he=he_img, loop_num=6,
                           scale_norm=scale_norm, self_norm=self_norm, batch_size=batch_size)
        total_time += time.time() - start_time


        diff0 = utils.unnormalize(diff0, means=mean, stds=std, times_two=True)
        diff1 = utils.unnormalize(diff1, means=mean, stds=std, times_two=True)
        diff2 = utils.unnormalize(diff2, means=mean, stds=std, times_two=True)
        diff3 = utils.unnormalize(diff3, means=mean, stds=std, times_two=True)

        diff0 = utils.linear_to_srgb(diff0)
        diff1 = utils.linear_to_srgb(diff1)
        diff2 = utils.linear_to_srgb(diff2)
        diff3 = utils.linear_to_srgb(diff3)

        for k in range(len(batch_fname)):
            fname = batch_fname[k]

            skimage.io.imsave(f'{pd_name}_f0/{fname}', t2n(diff0[k:k+1]))
            skimage.io.imsave(f'{pd_name}_f1/{fname}', t2n(diff1[k:k+1]))
            skimage.io.imsave(f'{pd_name}_f2/{fname}', t2n(diff2[k:k+1]))
            skimage.io.imsave(f'{pd_name}_f3/{fname}', t2n(diff3[k:k+1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['lowlight', 'lol'])
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()
    main(args)