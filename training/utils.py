import os
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torchvision
from skimage.metrics import structural_similarity as ssim

from training.constants import *


def compute_metrics(x_next, gt_img, lpips_fn):
    if x_next.dtype is not torch.float32:
        x_next = x_next.to(torch.float32)
    psnr = get_psnr(x_next, gt_img)
    ssim = get_ssim(x_next, gt_img)
    lpips = lpips_fn(x_next * 2 - 1, gt_img * 2 - 1).sum().item() if lpips_fn is not None else 0.0
    return psnr, ssim, lpips


def find_ratio(dark_path, light_path):
    assert isinstance(dark_path, str) and isinstance(light_path, str)
    dark_fn = os.path.basename(dark_path)
    light_fn = os.path.basename(light_path)

    dark_exposure = float(dark_fn[9:-5])
    light_exposure = float(light_fn[9:-5])

    exp_ratio = min(light_exposure / dark_exposure, 300)
    return exp_ratio


def normalize(img, means, stds):
    assert isinstance(img, torch.Tensor)
    assert len(means) == 3 and len(stds) == 3

    if img.ndim == 3:
        c, h, w = img.shape
        img = img - torch.tensor(data=means, dtype=img.dtype, device=img.device).view(c, 1, 1)
        img = img * torch.tensor(data=[1/x for x in stds], dtype=img.dtype, device=img.device).view(c, 1, 1)
    elif img.ndim == 4:
        b, c, h, w = img.shape
        img = img - torch.tensor(data=means, dtype=img.dtype, device=img.device).view(1, c, 1, 1)
        img = img * torch.tensor(data=[1 / x for x in stds], dtype=img.dtype, device=img.device).view(1, c, 1, 1)
    else:
        raise NotImplementedError
    return img



def unnorm_img(img, img_type: str, dataset_name: str, curve: str):
    if dataset_name == 'sony' and curve == 'linear':
        if img_type == 'light':
            img      = unnormalize(img, means=SONY_LIGHT_LINEAR_MEAN, stds=SONY_LIGHT_LINEAR_STD)
        elif img_type == 'dark':
            img    = unnormalize(img, means=SONY_DARK_LINEAR_MEAN, stds=SONY_DARK_LINEAR_STD)

    elif dataset_name == 'sony' and curve == 'srgb':
        if img_type == 'light':
            img      = unnormalize(img, means=SONY_LIGHT_SRGB_MEAN, stds=SONY_LIGHT_SRGB_STD)
        elif img_type == 'dark':
            img    = unnormalize(img, means=SONY_DARK_SRGB_MEAN, stds=SONY_DARK_SRGB_STD)

    elif dataset_name in ['sony_tif', 'sony_tif_crop'] and curve == 'linear':
        if img_type == 'light':
            img      = unnormalize(img, means=SONY_TIF_LIGHT_LINEAR_MEAN, stds=SONY_TIF_LIGHT_LINEAR_STD)
        elif img_type == 'dark':
            img    = unnormalize(img, means=SONY_TIF_DARK_LINEAR_MEAN, stds=SONY_TIF_DARK_LINEAR_STD)

    elif dataset_name in ['sony_tif', 'sony_tif_crop'] and curve == 'srgb':
        if img_type == 'light':
            img      = unnormalize(img, means=SONY_TIF_LIGHT_SRGB_MEAN, stds=SONY_TIF_LIGHT_SRGB_STD)
        elif img_type == 'dark':
            img    = unnormalize(img, means=SONY_TIF_DARK_SRGB_MEAN, stds=SONY_TIF_DARK_SRGB_STD)

    elif dataset_name in ['lowlight', 'lowlighthalf'] and curve == 'linear':
        if img_type == 'light':
            img      = unnormalize(img, means=LL_LIGHT_LINEAR_MEAN, stds=LL_LIGHT_LINEAR_STD)
        elif img_type == 'dark':
            img    = unnormalize(img, means=LL_DARK_LINEAR_MEAN1, stds=LL_DARK_LINEAR_STD1)

    elif dataset_name in ['lowlight', 'lowlighthalf'] and curve == 'srgb':
        if img_type == 'light':
            img      = unnormalize(img, means=LL_LIGHT_SRGB_MEAN, stds=LL_LIGHT_SRGB_STD)
        elif img_type == 'dark':
            img    = unnormalize(img, means=LL_DARK_SRGB_MEAN1, stds=LL_DARK_SRGB_STD1)

    elif dataset_name in ['lol', 'lol2'] and curve == 'linear':
        if img_type == 'light':
            img      = unnormalize(img, means=LOL_LIGHT_LINEAR_MEAN, stds=LOL_LIGHT_LINEAR_STD, times_two=True)
        elif img_type == 'dark':
            img    = unnormalize(img, means=LOL_DARK_LINEAR_MEAN, stds=LOL_DARK_LINEAR_STD, times_two=True)

    elif dataset_name in ['lol', 'lol2'] and curve == 'srgb':
        if img_type == 'light':
            img      = unnormalize(img, means=LOL_LIGHT_SRGB_MEAN, stds=LOL_LIGHT_LINEAR_STD, times_two=True, cube=False)
        elif img_type == 'dark':
            img    = unnormalize(img, means=LOL_DARK_SRGB_MEAN, stds=LOL_DARK_SRGB_STD, times_two=True)

    elif dataset_name == 'cats':
        img = 0.5 * (img + 1)

    else:
        raise NotImplementedError

    return img


def unnormalize(img, means, stds, times_two=True, cube=True):
    assert isinstance(img, torch.Tensor)
    assert img.ndim == 4 and img.shape[1] == 3
    assert len(means) == 3 and len(stds) == 3
    if times_two:
        img = img * 2.0

    b, c, h, w = img.shape
    img = img * torch.tensor(data=stds, dtype=img.dtype, device=img.device).view(1, c, 1, 1)
    img = img + torch.tensor(data=means, dtype=img.dtype, device=img.device).view(1, c, 1, 1)
    if cube:
        img = img ** 4.0
    img = torch.clamp(img, 0., 1.0)
    return img


def equalize_histogram(img):
    assert img.ndim == 3 and img.shape[-1] == 3
    (r, g, b) = cv2.split(img)
    rh = cv2.equalizeHist(r)
    gh = cv2.equalizeHist(g)
    bh = cv2.equalizeHist(b)
    return cv2.merge((rh, gh, bh))


def linear_to_srgb(x, eps=1e-8):
    assert isinstance(x, torch.Tensor)
    a = 0.055
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.0031308, 12.92 * x, (1. + a) * x ** (1. / 2.4) - a)


def srgb_to_linear_np(x, eps=1e-8):
    # assert isinstance(x, torch.Tensor)
    assert x.min() >= 0.0 and x.max() <= 1.0
    x = np.clip(x, eps, 1.)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def srgb_to_linear(x, eps=1e-8):
    assert isinstance(x, torch.Tensor)
    assert x.min() >= 0.0 and x.max() <= 1.0
    x = x.clamp(eps, 1.)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def center_crop(img, size):
     assert (img.ndim == 4) and isinstance(size, int)
     b, c, h, w = img.shape
     start_x = (w // 2) - (size // 2)
     start_y = (h // 2) - (size // 2)
     return img[:, :, start_y:start_y + size, start_x:start_x + size]


def random_crop(img, size):
    assert(img.ndim == 4) and isinstance(size, int)
    b, c, h, w = img.shape
    start_x = random.randint(0, w-size-1)
    start_y = random.randint(0, h-size-1)
    return img[:, :, start_y:start_y + size, start_x:start_x + size]


def get_psnr(pred, gt):
    return 10 * torch.log10(1 / torch.mean((pred - gt) ** 2)).detach().cpu().numpy()


def get_ssim(pred, gt):
    ssims = []
    for i in range(pred.shape[0]):
        pred_i = pred[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        gt_i = gt[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        # set weights to more closely match MatLab implementation by Wang et al.
        ssims.append(ssim(pred_i, gt_i, channel_axis=2, data_range=1.0,
                          gaussian_weights=True, use_sample_covariance=False,
                          K1=0.01, K2=0.03, sigma=1.5))
    return sum(ssims) / len(ssims)


def get_psnr_stats(pred, gt):
    psnrs = []
    for i in range(pred.shape[0]):
        psnrs.append(10 * torch.log10(1 / torch.mean((pred[i] - gt[i]) ** 2)).detach().cpu().numpy())
    return np.mean(psnrs), np.std(psnrs)


def get_ssim_stats(pred, gt):
    ssims = []
    for i in range(pred.shape[0]):
        pred_i = pred[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        gt_i = gt[i, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        ssims.append(ssim(pred_i, gt_i, channel_axis=2, gaussian_weights=True, K1=0.01, K2=0.03, sigma=1.5))
    return np.mean(ssims), np.std(ssims)


def get_lpips_stats(lpips_fn, pred, gt):
    assert pred.min() >= 0 and pred.max() <= 1.0
    assert gt.min() >= 0 and gt.max() <= 1.0
    losses = []
    pred = pred * 2 - 1
    gt = gt * 2 - 1
    for i in range(pred.shape[0]):
        losses.append(lpips_fn(pred[i], gt[i]).item())
    return np.mean(losses), np.std(losses)


def trim_files(run_dir, key_word):
    files = sorted([f for f in os.listdir(run_dir) if os.path.isfile(os.path.join(run_dir, f)) and key_word in f])
    if len(files) > 3:
        os.remove(f'{run_dir}/{files[0]}')


def make_border_mask(batch_size, patch_size, pick=None):
    width = 3
    mask = torch.zeros((batch_size, 3, patch_size-(width*2), patch_size-(width*2)))
    mask = torchvision.transforms.Pad(width, fill=1)(mask)

    if pick is None:
        pick = random.randint(0, 8)
    if pick == 0:
        return mask
    if pick == 1:
        new_mask = torch.zeros((batch_size, 3, patch_size, patch_size))
        new_mask[:, :, width:, :] = mask[:, :, width:, :]
    if pick == 2:
        new_mask = torch.zeros((batch_size, 3, patch_size, patch_size))
        new_mask[:, :, :, width:] = mask[:, :, :, width:]
    if pick == 3:
        new_mask = torch.zeros((batch_size, 3, patch_size, patch_size))
        new_mask[:, :, :-width, :] = mask[:, :, :-width, :]
    if pick == 4:
        new_mask = torch.zeros((batch_size, 3, patch_size, patch_size))
        new_mask[:, :, :, :-width] = mask[:, :, :, :-width]
    if pick == 5:
        new_mask = torch.zeros((batch_size, 3, patch_size, patch_size))
        new_mask[:, :, width:, :-width] = mask[:, :, width:, :-width]
    if pick == 6:
        new_mask = torch.zeros((batch_size, 3, patch_size, patch_size))
        new_mask[:, :, width:, width:] = mask[:, :, width:, width:]
    if pick == 7:
        new_mask = torch.zeros((batch_size, 3, patch_size, patch_size))
        new_mask[:, :, :-width, width:] = mask[:, :, :-width, width:]
    if pick == 8:
        new_mask = torch.zeros((batch_size, 3, patch_size, patch_size))
        new_mask[:, :, :-width, :-width] = mask[:, :, :-width, :-width]
    return new_mask