from easydict import EasyDict
import torch
import torchvision
import training.utils as utils
import time
import kornia.contrib as kc
import numpy as np

import torch.nn.functional as F
import training.resizer as resizer


def resize(img, from_size, to_size):
    assert img.shape[-1] == from_size[-1] and img.shape[-2] == from_size[-2]
    assert isinstance(to_size, tuple)
    if from_size[0] == to_size[0]:
        return img
    return F.interpolate(img, size=to_size, mode='bilinear', align_corners=False, antialias=True)


def scale_loop(net, diff_img, cond_img_fr, from_size, to_size, patch_size, self_norm, scale_norm, border_norm, sampling_fn, batch_size, device, gt0, he):
    dark_down = resize(cond_img_fr, from_size=(cond_img_fr.shape[-2], cond_img_fr.shape[-1]), to_size=to_size)
    clean_lr = resize(diff_img, from_size=(diff_img.shape[-2], diff_img.shape[-1]), to_size=to_size)
    gt0 = resize(gt0, from_size=patch_size, to_size=to_size)
    hist = resize(he, from_size=(cond_img_fr.shape[-2], cond_img_fr.shape[-1]), to_size=to_size)

    imgs = torch.cat([dark_down, clean_lr, gt0, hist], dim=1)
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
            input_patches = torch.cat([input_patches, img_patches[:, i, -3:]], dim=1)
        if border_norm:  # patches dim B, N, C, H, W
            mask = utils.make_border_mask(img_patches.shape[0], patch_size=patch_size[0], pick=0).to(device)
            input = img_patches[:, i, 3:6]
            borders = mask * input + (1. - mask) * torch.randn_like(input) * 0.5
            input_patches = torch.cat([input_patches, borders], dim=1)
        x_next, x_arr = sampling_fn(net, batch_size, input_patches, device=device)
        diff_patches.append(x_next.unsqueeze(1))

    diff_patches = torch.cat(diff_patches, dim=1)
    diff_img = kc.combine_tensor_patches(diff_patches, original_size=to_size, window_size=patch_size, stride=patch_size, unpadding=pad)
    return diff_img


def generate_sample_grid_bos_nonorm(
    net, test_dataset_iterator, lpips_fn, dataset_name='sony', curve='linear',
    self_norm=False, scale_norm=False, border_norm=False, device=torch.device('cuda')
):

    # sampling_fn = sampling_loop
    sampling_fn = sampling_loop_ilvr

    with torch.no_grad():

        data         = next(test_dataset_iterator)
        cond_img     = data['cond_img'].to(device)
        cond_img_fr  = data['cond_img_fr'].to(device)
        gt_img_fr    = data['gt_img_fr'].to(device)
        he_fr        = data['he_fr'].to(device)
        ratio        = data['ratio']
        batch_size, _, h, w = cond_img.shape

        dark_down = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
        clean_lr  = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
        hist  = resize(he_fr, from_size=(8 * h, 8 * w), to_size=(h, w))

        if scale_norm:
            imgs = torch.cat([dark_down, clean_lr, clean_lr], dim=1)
        else:
            imgs = torch.cat([dark_down, clean_lr], dim=1)

        if self_norm:
            imgs = torch.cat([imgs, hist], dim=1)

        if border_norm:
            mask    = utils.make_border_mask(batch_size, patch_size=h).to(device)
            borders0 = mask * clean_lr + (1. - mask) * torch.randn_like(clean_lr) * 0.5
            imgs = torch.cat([imgs, borders0], dim=1)

        diff_img0, x_arr = sampling_loop(net, batch_size, imgs, device=device)

        print('scale 1')
        start_time = time.time()
        diff_img1 = scale_loop(net, diff_img0, cond_img_fr, from_size=(h, w), to_size=(2 * h, 2 * w),
                               patch_size=(h,w),
                               self_norm=self_norm, scale_norm=scale_norm, border_norm=border_norm, sampling_fn=sampling_fn,
                               batch_size=batch_size, device=device, gt0=diff_img0, he=he_fr)
        print(f'time: {time.time() - start_time}')
        print('scale 2')
        start_time = time.time()
        diff_img2 = scale_loop(net, diff_img1, cond_img_fr, from_size=(2 * h, 2 * w), to_size=(4 * h, 4 * w),
                               patch_size=(h,w),
                               self_norm=self_norm, scale_norm=scale_norm, border_norm=border_norm, sampling_fn=sampling_fn,
                               batch_size=batch_size, device=device, gt0=diff_img0, he=he_fr)
        print(f'time: {time.time() - start_time}')
        print('scale 3')
        start_time = time.time()
        diff_img3 = scale_loop(net, diff_img2, cond_img_fr, from_size=(4 * h, 4 * w), to_size=(8 * h, 8 * w),
                               patch_size=(h,w),
                               self_norm=self_norm, scale_norm=scale_norm, border_norm=border_norm, sampling_fn=sampling_fn,
                               batch_size=batch_size, device=device, gt0=diff_img0, he=he_fr)
        print(f'time: {time.time() - start_time}')

        gt_img_fr   = (gt_img_fr + 1) * 0.5
        cond_img_fr = (cond_img_fr + 1) * 0.5
        diff_img0   =  (diff_img0 + 1) * 0.5
        diff_img1   = (diff_img1 + 1) * 0.5
        diff_img2   = (diff_img2 + 1) * 0.5
        diff_img3   =  (diff_img3 + 1) * 0.5

        ## compute metrics
        psnr, ssim, lpips = utils.compute_metrics(diff_img3, gt_img_fr, lpips_fn)

        assert x_arr.ndim == 5 and x_arr.shape[0] == 7
        x_arr = 0.5 * (x_arr + 1)

        if curve == 'linear': # for visualization
            diff_img3   = utils.linear_to_srgb(diff_img3)
            diff_img2   = utils.linear_to_srgb(diff_img2)
            diff_img1   = utils.linear_to_srgb(diff_img1)
            diff_img0   = utils.linear_to_srgb(diff_img0)
            gt_img_fr   = utils.linear_to_srgb(gt_img_fr)
            cond_img_fr = utils.linear_to_srgb(cond_img_fr)

        cond_img_fr = cond_img_fr * ratio.view(batch_size, 1, 1, 1).to(cond_img.device)
        cond_img_fr = torch.clamp(cond_img_fr, 0., 1.)

        gt     = resize(gt_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
        cond   = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
        final0 = torchvision.utils.make_grid(torch.cat([diff_img0, gt, cond], dim=0), nrow=2)

        gt     = resize(gt_img_fr, from_size=(8 * h, 8 * w), to_size=(2 * h, 2 * w))
        cond   = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(2 * h, 2 * w))
        final1 = torchvision.utils.make_grid(torch.cat([diff_img1, gt, cond], dim=0), nrow=2)

        gt     = resize(gt_img_fr, from_size=(8 * h, 8 * w), to_size=(4 * h, 4 * w))
        cond   = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(4 * h, 4 * w))
        final2 = torchvision.utils.make_grid(torch.cat([diff_img2, gt, cond], dim=0), nrow=2)

        final3      = torchvision.utils.make_grid(torch.cat([diff_img3, gt_img_fr, cond_img_fr], dim=0), nrow=2)
        final_x_arr = torchvision.utils.make_grid(make_final_x_arr(x_arr, batch_size), nrow=7)

    # final x arr and img are in CHW format
    return EasyDict({'diff3': final3,
                     'diff0': final0,
                     'diff1': final1,
                     'diff2': final2,
                     'x_arr': final_x_arr,
                     'psnr': psnr,
                     'ssim': ssim,
                     'lpips': lpips})


def generate_sample_grid_bos(
    net, test_dataset_iterator, lpips_fn, dataset_name='sony', curve='linear',
    self_norm=False, scale_norm=False, border_norm=False, device=torch.device('cuda'), scale=None,
):
    # sampling_fn = sampling_loop
    sampling_fn = sampling_loop_ilvr

    with torch.no_grad():

        data         = next(test_dataset_iterator)
        cond_img     = data['cond_img'].to(device)
        cond_img_fr  = data['cond_img_fr'].to(device)
        gt_img_fr    = data['gt_img_fr'].to(device)
        he_fr        = data['he_fr'].to(device)
        ratio        = data['ratio']
        batch_size, _, h, w = cond_img.shape

        dark_down = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
        clean_lr  = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
        hist  = resize(he_fr, from_size=(8 * h, 8 * w), to_size=(h, w))

        if scale_norm:
            imgs = torch.cat([dark_down, clean_lr, clean_lr], dim=1)
        else:
            imgs = torch.cat([dark_down, clean_lr], dim=1)

        if self_norm:
            imgs = torch.cat([imgs, hist], dim=1)

        if border_norm:
            mask    = utils.make_border_mask(batch_size, patch_size=h).to(device)
            borders0 = mask * clean_lr + (1. - mask) * torch.randn_like(clean_lr) * 0.5
            imgs = torch.cat([imgs, borders0], dim=1)

        if scale is None:
            diff_img0, x_arr = sampling_loop(net, batch_size, imgs, device=device)

            print('scale 1')
            start_time = time.time()
            diff_img1 = scale_loop(net, diff_img0, cond_img_fr, from_size=(h, w), to_size=(2 * h, 2 * w),
                                   patch_size=(h,w),
                                   self_norm=self_norm, scale_norm=scale_norm, border_norm=border_norm, sampling_fn=sampling_fn,
                                   batch_size=batch_size, device=device, gt0=diff_img0, he=he_fr)
            print(f'time: {time.time() - start_time}')
            print('scale 2')
            start_time = time.time()
            diff_img2 = scale_loop(net, diff_img1, cond_img_fr, from_size=(2 * h, 2 * w), to_size=(4 * h, 4 * w),
                                   patch_size=(h,w),
                                   self_norm=self_norm, scale_norm=scale_norm, border_norm=border_norm, sampling_fn=sampling_fn,
                                   batch_size=batch_size, device=device, gt0=diff_img0, he=he_fr)
            print(f'time: {time.time() - start_time}')
            print('scale 3')
            start_time = time.time()
            diff_img3 = scale_loop(net, diff_img2, cond_img_fr, from_size=(4 * h, 4 * w), to_size=(8 * h, 8 * w),
                                   patch_size=(h,w),
                                   self_norm=self_norm, scale_norm=scale_norm, border_norm=border_norm, sampling_fn=sampling_fn,
                                   batch_size=batch_size, device=device, gt0=diff_img0, he=he_fr)
            print(f'time: {time.time() - start_time}')

            gt_img_fr   = utils.unnorm_img(gt_img_fr, img_type='light', dataset_name=dataset_name, curve=curve)
            cond_img_fr = utils.unnorm_img(cond_img_fr, img_type='dark', dataset_name=dataset_name, curve=curve)
            diff_img0   = utils.unnorm_img(diff_img0, img_type='light', dataset_name=dataset_name, curve=curve)
            diff_img1   = utils.unnorm_img(diff_img1, img_type='light', dataset_name=dataset_name, curve=curve)
            diff_img2   = utils.unnorm_img(diff_img2, img_type='light', dataset_name=dataset_name, curve=curve)
            diff_img3   = utils.unnorm_img(diff_img3, img_type='light', dataset_name=dataset_name, curve=curve)

            ## compute metrics
            psnr, ssim, lpips = utils.compute_metrics(diff_img3, gt_img_fr, lpips_fn)

            assert x_arr.ndim == 5 and x_arr.shape[0] == 7
            x_arr = 0.5 * (x_arr + 1)

            if curve == 'linear': # for visualization
                diff_img3   = utils.linear_to_srgb(diff_img3)
                diff_img2   = utils.linear_to_srgb(diff_img2)
                diff_img1   = utils.linear_to_srgb(diff_img1)
                diff_img0   = utils.linear_to_srgb(diff_img0)
                gt_img_fr   = utils.linear_to_srgb(gt_img_fr)
                cond_img_fr = utils.linear_to_srgb(cond_img_fr)

            cond_img_fr = cond_img_fr * ratio.view(batch_size, 1, 1, 1).to(cond_img.device)
            cond_img_fr = torch.clamp(cond_img_fr, 0., 1.)

            gt     = resize(gt_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
            cond   = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
            final0 = torchvision.utils.make_grid(torch.cat([diff_img0, gt, cond], dim=0), nrow=2)

            gt     = resize(gt_img_fr, from_size=(8 * h, 8 * w), to_size=(2 * h, 2 * w))
            cond   = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(2 * h, 2 * w))
            final1 = torchvision.utils.make_grid(torch.cat([diff_img1, gt, cond], dim=0), nrow=2)

            gt     = resize(gt_img_fr, from_size=(8 * h, 8 * w), to_size=(4 * h, 4 * w))
            cond   = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(4 * h, 4 * w))
            final2 = torchvision.utils.make_grid(torch.cat([diff_img2, gt, cond], dim=0), nrow=2)

            final3      = torchvision.utils.make_grid(torch.cat([diff_img3, gt_img_fr, cond_img_fr], dim=0), nrow=2)
            final_x_arr = torchvision.utils.make_grid(make_final_x_arr(x_arr, batch_size), nrow=7)

            # final x arr and img are in CHW format
            return EasyDict({'diff3': final3,
                             'diff0': final0,
                             'diff1': final1,
                             'diff2': final2,
                             'x_arr': final_x_arr,
                             'psnr': psnr,
                             'ssim': ssim,
                             'lpips': lpips})
        elif scale == 3:
            gt0 = resize(gt_img_fr, from_size=(8 *h, 8 * w), to_size=(h,w))

            diff_img3 = scale_loop(net, gt0, cond_img_fr, from_size=(h, w), to_size=(8 * h, 8 * w),
                                   patch_size=(h, w),
                                   self_norm=self_norm, scale_norm=scale_norm, border_norm=border_norm,
                                   sampling_fn=sampling_fn,
                                   batch_size=batch_size, device=device, he=he_fr, gt0=gt0)

            gt_img_fr = utils.unnorm_img(gt_img_fr, img_type='light', dataset_name=dataset_name, curve=curve)
            cond_img_fr = utils.unnorm_img(cond_img_fr, img_type='dark', dataset_name=dataset_name, curve=curve)
            diff_img3 = utils.unnorm_img(diff_img3, img_type='light', dataset_name=dataset_name, curve=curve)

            ## compute metrics
            psnr, ssim, lpips = utils.compute_metrics(diff_img3, gt_img_fr, lpips_fn)

            if curve == 'linear':  # for visualization
                diff_img3 = utils.linear_to_srgb(diff_img3)
                gt_img_fr = utils.linear_to_srgb(gt_img_fr)
                cond_img_fr = utils.linear_to_srgb(cond_img_fr)

            cond_img_fr = cond_img_fr * ratio.view(batch_size, 1, 1, 1).to(cond_img.device)
            cond_img_fr = torch.clamp(cond_img_fr, 0., 1.)

            final3 = torchvision.utils.make_grid(torch.cat([diff_img3, gt_img_fr, cond_img_fr], dim=0), nrow=2)

            # final x arr and img are in CHW format
            return EasyDict({'diff3': final3,
                             'psnr': psnr,
                             'ssim': ssim,
                             'lpips': lpips})



def generate_sample_grid_bos_singlestage(
    net, test_dataset_iterator, scale, lpips_fn, dataset_name='sony', curve='linear',
    self_norm=False, scale_norm=False, border_norm=False, device=torch.device('cuda')
):
    # sampling_fn = sampling_loop
    sampling_fn = sampling_loop_ilvr

    with torch.no_grad():

        data         = next(test_dataset_iterator)
        cond_img     = data['cond_img'].to(device)
        cond_img_fr  = data['cond_img_fr'].to(device)
        gt_img_fr    = data['gt_img_fr'].to(device)
        he_fr        = data['he_fr'].to(device)
        ratio        = data['ratio']
        batch_size, _, h, w = cond_img.shape

        dark_down = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
        clean_lr  = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
        hist  = resize(he_fr, from_size=(8 * h, 8 * w), to_size=(h, w))

        if scale_norm:
            imgs = torch.cat([dark_down, clean_lr, clean_lr], dim=1)
        else:
            imgs = torch.cat([dark_down, clean_lr], dim=1)

        if self_norm:
            imgs = torch.cat([imgs, hist], dim=1)

        if border_norm:
            mask    = utils.make_border_mask(batch_size, patch_size=h).to(device)
            borders0 = mask * clean_lr + (1. - mask) * torch.randn_like(clean_lr) * 0.5
            imgs = torch.cat([imgs, borders0], dim=1)

        if scale == 0:
            diff_img0, x_arr = sampling_loop(net, batch_size, imgs, device=device)
        elif scale == 3:
            diff_img3 = scale_loop(net, diff_img0, cond_img_fr, from_size=(4 * h, 4 * w), to_size=(8 * h, 8 * w),
                                   patch_size=(h,w),
                                   self_norm=self_norm, scale_norm=scale_norm, border_norm=border_norm, sampling_fn=sampling_fn,
                                   batch_size=batch_size, device=device, gt0=diff_img0, he=he_fr)

        gt_img_fr   = utils.unnorm_img(gt_img_fr, img_type='light', dataset_name=dataset_name, curve=curve)
        cond_img_fr = utils.unnorm_img(cond_img_fr, img_type='dark', dataset_name=dataset_name, curve=curve)
        if scale == 0:
            diff_img   = utils.unnorm_img(diff_img0, img_type='light', dataset_name=dataset_name, curve=curve)
        elif scale == 3:
            diff_img   = utils.unnorm_img(diff_img3, img_type='light', dataset_name=dataset_name, curve=curve)

        ## compute metrics
        psnr, ssim, lpips = utils.compute_metrics(diff_img, gt_img_fr, lpips_fn)

        assert x_arr.ndim == 5 and x_arr.shape[0] == 7

        if curve == 'linear': # for visualization
            diff_img   = utils.linear_to_srgb(diff_img)
            gt_img_fr   = utils.linear_to_srgb(gt_img_fr)
            cond_img_fr = utils.linear_to_srgb(cond_img_fr)

        cond_img_fr = cond_img_fr * ratio.view(batch_size, 1, 1, 1).to(cond_img.device)
        cond_img_fr = torch.clamp(cond_img_fr, 0., 1.)

        gt     = resize(gt_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
        cond   = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
        final = torchvision.utils.make_grid(torch.cat([diff_img, gt, cond], dim=0), nrow=2)

    # final x arr and img are in CHW format
    return EasyDict({'diff': final,
                     'psnr': psnr,
                     'ssim': ssim,
                     'lpips': lpips})


def generate_sample_grid_bos2(
    net, test_dataset_iterator, lpips_fn, dataset_name='sony', curve='linear',
    self_norm=False, scale_norm=False, border_norm=False, device=torch.device('cuda')
):

    # sampling_fn = sampling_loop
    sampling_fn = sampling_loop_ilvr

    with torch.no_grad():

        data         = next(test_dataset_iterator)
        cond_img     = data['cond_img'].to(device)
        cond_img_fr  = data['cond_img_fr'].to(device)
        gt_img_fr    = data['gt_img_fr'].to(device)
        he_fr        = data['he_fr'].to(device)
        ratio        = data['ratio']
        batch_size, _, h, w = cond_img.shape

        dark_down = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
        clean_lr  = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
        hist  = resize(he_fr, from_size=(8 * h, 8 * w), to_size=(h, w))

        if scale_norm:
            imgs = torch.cat([dark_down, clean_lr, clean_lr], dim=1)
        else:
            imgs = torch.cat([dark_down, clean_lr], dim=1)

        if self_norm:
            imgs = torch.cat([imgs, hist], dim=1)

        if border_norm:
            mask    = utils.make_border_mask(batch_size, patch_size=h).to(device)
            borders0 = mask * clean_lr + (1. - mask) * torch.randn_like(clean_lr) * 0.5
            imgs = torch.cat([imgs, borders0], dim=1)

        diff_img0, x_arr = sampling_loop(net, batch_size, imgs, device=device)

        diff_img3 = scale_loop(net, diff_img0, cond_img_fr, from_size=(4 * h, 4 * w), to_size=(8 * h, 8 * w),
                               patch_size=(h,w),
                               self_norm=self_norm, scale_norm=scale_norm, border_norm=border_norm, sampling_fn=sampling_fn,
                               batch_size=batch_size, device=device, gt0=diff_img0, he=he_fr)

        gt_img_fr   = utils.unnorm_img(gt_img_fr, img_type='light', dataset_name=dataset_name, curve=curve)
        cond_img_fr = utils.unnorm_img(cond_img_fr, img_type='dark', dataset_name=dataset_name, curve=curve)
        diff_img0   = utils.unnorm_img(diff_img0, img_type='light', dataset_name=dataset_name, curve=curve)
        diff_img3   = utils.unnorm_img(diff_img3, img_type='light', dataset_name=dataset_name, curve=curve)

        ## compute metrics
        psnr, ssim, lpips = utils.compute_metrics(diff_img3, gt_img_fr, lpips_fn)

        assert x_arr.ndim == 5 and x_arr.shape[0] == 7
        x_arr = 0.5 * (x_arr + 1)

        if curve == 'linear': # for visualization
            diff_img3   = utils.linear_to_srgb(diff_img3)
            diff_img0   = utils.linear_to_srgb(diff_img0)
            gt_img_fr   = utils.linear_to_srgb(gt_img_fr)
            cond_img_fr = utils.linear_to_srgb(cond_img_fr)

        cond_img_fr = cond_img_fr * ratio.view(batch_size, 1, 1, 1).to(cond_img.device)
        cond_img_fr = torch.clamp(cond_img_fr, 0., 1.)

        gt     = resize(gt_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
        cond   = resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(h, w))
        final0 = torchvision.utils.make_grid(torch.cat([diff_img0, gt, cond], dim=0), nrow=2)

        final3      = torchvision.utils.make_grid(torch.cat([diff_img3, gt_img_fr, cond_img_fr], dim=0), nrow=2)
        final_x_arr = torchvision.utils.make_grid(make_final_x_arr(x_arr, batch_size), nrow=7)

    # final x arr and img are in CHW format
    return EasyDict({'diff3': final3,
                     'diff0': final0,
                     'x_arr': final_x_arr,
                     'psnr': psnr,
                     'ssim': ssim,
                     'lpips': lpips})


def generate_sample_grid(
    net, test_dataset_iterator, lpips_fn, dataset_name='sony', curve='linear', device=torch.device('cuda')):
    net.eval()
    data = next(test_dataset_iterator)

    cond_img = data['cond_img'].to(device)
    gt_img = data['gt_img'].to(device)
    ratio = data['ratio']

    batch_size = cond_img.shape[0]
    x_next, x_arr = sampling_loop(net, batch_size, cond_img, device=device)

    if cond_img.shape[1] > 3: # remove self normalized image
        cond_img = cond_img[:, :3, : ,:]

    gt_img = utils.unnorm_img(gt_img, img_type='light', dataset_name=dataset_name, curve=curve)
    cond_img = utils.unnorm_img(cond_img, img_type='dark', dataset_name=dataset_name, curve=curve)
    x_next = utils.unnorm_img(x_next, img_type='light', dataset_name=dataset_name, curve=curve)

    ## compute metrics
    img_psnr, img_ssim, img_lpips = utils.compute_metrics(x_next, gt_img, lpips_fn)
    assert x_arr.ndim == 5 and x_arr.shape[0] == 7
    x_arr = 0.5 * (x_arr + 1)

    if curve == 'linear': # for visualization
        x_next      = utils.linear_to_srgb(x_next)
        gt_img      = utils.linear_to_srgb(gt_img)
        cond_img    = utils.linear_to_srgb(cond_img)
    cond_img = cond_img * ratio.view(batch_size, 1, 1, 1).to(cond_img.device)
    cond_img = torch.clamp(cond_img, 0, 1.0)

    final_img = torchvision.utils.make_grid(torch.cat([x_next, gt_img, cond_img], dim=0), nrow=2)
    final_x_arr = torchvision.utils.make_grid(make_final_x_arr(x_arr, batch_size), nrow=7)
    net.train()
    # final x arr and img are in CHW format
    return EasyDict({'samples': final_img,
            'x_arr': final_x_arr,
            'psnr': img_psnr,
            'ssim': img_ssim,
            'lpips': img_lpips})



def sampling_loop_ilvr(net, batch_size, cond_img, device=torch.device('cuda:0'),
                  return_time=False, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
                S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):

    def il_resize(img, out_shape):
        return resizer.resize(img, out_shape=out_shape)

    b, c, h, w = cond_img.shape
    ref = cond_img[:, 3:6]

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
    start_time = time.time()
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

            if i % 3 == 0:
                x_arr = torch.cat([x_arr, x_next.clone()[None, ...]], dim=0)

        # # q sample with reference image we want low frequency information from
        if i < 5:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            ref_hat = ref + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(ref)
            x_next = x_next - il_resize(il_resize(x_next, out_shape=(h//2, w//2)), out_shape=(h,w)) + il_resize(il_resize(ref_hat, out_shape=(h//2, w//2)), out_shape=(h,w))


    if return_time:
        return x_next, x_arr, time.time() - start_time
    return x_next, x_arr



def sampling_loop(net, batch_size, cond_img, device=torch.device('cuda:0'),
                  return_time=False, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
                S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
    # num_steps = 100

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
    start_time = time.time()
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

            if i % 3 == 0:
                x_arr = torch.cat([x_arr, x_next.clone()[None, ...]], dim=0)

    if return_time:
        return x_next, x_arr, time.time() - start_time
    return x_next, x_arr


#----------------------------------------------------------------------------

def make_final_x_arr(x_arr, batch_size):
    final_x_arr = x_arr[:, 0]
    for i in range(1, batch_size):
        final_x_arr = torch.cat([final_x_arr, x_arr[:, i]], dim=0)
    return final_x_arr
