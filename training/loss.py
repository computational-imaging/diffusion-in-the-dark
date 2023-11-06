# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import random

import kornia.contrib as kc
import torch
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt

import training.utils as utils
from torch_utils import persistence


#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, use_lpips=False):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.use_lpips = use_lpips

    def __call__(self, net, gt_img, cond_img, lpips_fn=None, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([gt_img.shape[0], 1, 1, 1], device=gt_img.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        assert augment_pipe is None
        augment_labels = None
        n = torch.randn_like(gt_img) * sigma

        D_yn = net(gt_img + n, cond_img, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - gt_img) ** 2)

        if self.use_lpips:
            loss += 5 * lpips_fn(D_yn, gt_img)
        return loss


@persistence.persistent_class
class EDMBosLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, scale=None, self_norm=False,
                 add_noise=False, scale_norm=False, border_norm=False, use_lpips=False):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.scale = scale
        self.self_norm = self_norm
        self.add_noise = add_noise
        self.scale_norm = scale_norm
        self.use_lpips = use_lpips
        self.border_norm = border_norm

    def _resize(self, img, from_size, to_size):
        assert img.shape[-1] == from_size[-1] and img.shape[-2] == from_size[-2]
        assert isinstance(to_size, tuple)

        return F.interpolate(img, size=to_size, mode='bilinear', align_corners=False, antialias=True)

    def select_random_patches(self, imgs, num_patches, size):
        assert imgs.dtype in [torch.float32, torch.float64]
        assert imgs.ndim == 4
        assert isinstance(size, int)

        b, c, h, w = imgs.shape
        assert h >= size * 2 and w >= size * 2

        start_h_idxs = random.choices(range(0, h-size), k=num_patches)
        end_h_idxs = [x+size for x in start_h_idxs]
        start_w_idxs = random.choices(range(0, w-size), k=num_patches)
        end_w_idxs = [x+size for x in start_w_idxs]

        patches = []
        for i in range(0, num_patches):
            patches.append(imgs[i:i+1, :, start_h_idxs[i]:end_h_idxs[i], start_w_idxs[i]:end_w_idxs[i]])

        patches = torch.cat(patches, dim=0)
        return patches

    def __call__(self, net, gt_img, cond_img, gt_img_fr, cond_img_fr, he, he_fr, labels=None, augment_pipe=None, lpips_fn=None):
        b, c, h, w = gt_img.shape
        sz_map = {1: 2 * h,
                  2: 4 * h,
                  3: 8 * h}
        
        if self.scale is None: # single version
            scale = random.randint(0, 3)
        else: # multi model version
            scale = self.scale

        if scale == 0:
            dark_down  = cond_img
            light_up   = cond_img
            gt_down    = gt_img
            hist       = he
        elif scale == 1:
            dark_down  = self._resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(2 * h, 2 * w))
            light_down = self._resize(gt_img_fr, from_size=(8 * h, 8 * w), to_size=(h,  w))
            light_up   = self._resize(light_down, from_size=(h,  w), to_size=(2 * h, 2 * w))
            gt_down    = self._resize(gt_img_fr, from_size=(8 * h, 8 * w), to_size=(2 * h, 2 * w))
        elif scale == 2:
            dark_down  = self._resize(cond_img_fr, from_size=(8 * h, 8 * w), to_size=(4 * h,  4 * w))
            light_down = self._resize(gt_img_fr, from_size=(8 * h, 8 * w), to_size=(2 * h, 2 * w))
            light_up   = self._resize(light_down, from_size=(2 * h, 2 * w), to_size=(4 * h,  4 * w))
            gt_down    = self._resize(gt_img_fr, from_size=(8 * h, 8 * w), to_size=(4 * h,  4 * w))

        elif scale == 3:
            dark_down  = cond_img_fr
            light_down = self._resize(gt_img_fr, from_size=(8 * h, 8 * w), to_size=(4 * h,  4 * w))
            light_up   = self._resize(light_down, from_size=(4 * h,  4 * w), to_size=(8 * h, 8 * w))
            gt_down    = gt_img_fr
        else:
            raise NotImplementedError


        if self.add_noise:
            ### NEW ADDITION: add some noise to the gt image
            light_up = light_up + torch.randn_like(light_up) * random.random()

        if self.scale_norm and scale > 0:
            tiny_gt = self._resize(gt_img_fr, from_size=(8 * h, 8 * w), to_size=(h,  w))
            tiny_gt = self._resize(tiny_gt, from_size=(h,  w), to_size=(sz_map[scale], sz_map[scale]))
        #
        if self.self_norm and scale > 0:
            hist = self._resize(he_fr, from_size=(8 * h, 8 * w), to_size=(sz_map[scale], sz_map[scale]))

        if scale == 0 and not self.scale_norm:
            cond_patches = torch.cat([dark_down, light_up], dim=1)
            gt_patches = gt_down
            if self.self_norm:
                cond_patches = torch.cat([cond_patches, hist], dim=1)
        elif scale == 0 and self.scale_norm:
            cond_patches = torch.cat([dark_down, light_up, light_up], dim=1)
            gt_patches = gt_down
            if self.self_norm:
                cond_patches = torch.cat([cond_patches, hist], dim=1)

        elif scale > 0 and self.scale_norm:
            imgs    = torch.cat([dark_down, light_up, tiny_gt, gt_down], dim=1) # concat so we get consistent patches
            if self.self_norm:
                imgs = torch.cat([imgs, hist], dim=1)

            # patches dim N C H W
            patches = self.select_random_patches(imgs, gt_img.shape[0], size=32)

            cond_patches = patches[:, 0:9] # dark down, light up
            gt_patches   = patches[:, 9:12]

            if self.self_norm:
                cond_patches = torch.cat([cond_patches, patches[:, 12:15]], dim=1)

        elif scale > 0 and not self.scale_norm:
            imgs    = torch.cat([dark_down, light_up, gt_down], dim=1) # concat so we get consistent patches
            if self.self_norm:
                imgs = torch.cat([imgs, hist], dim=1)

            # patches dim N C H W
            patches = self.select_random_patches(imgs, gt_img.shape[0], size=32)

            cond_patches = patches[:, 0:6] # dark down, light up
            gt_patches   = patches[:, 6:9]

            if self.self_norm:
                cond_patches = torch.cat([cond_patches, patches[:, 9:12]], dim=1)

        if self.border_norm and scale > 0:
            mask         = utils.make_border_mask(gt_img.shape[0], patch_size=h).to(gt_img.device)
            gt_border    = mask * gt_patches + (1. - mask) * torch.randn_like(gt_patches) * 0.5 # use gt patches to get color
            cond_patches = torch.cat([cond_patches, gt_border], dim=1)
        elif self.border_norm and scale == 0:
            mask         = utils.make_border_mask(cond_img.shape[0], patch_size=h).to(cond_img.device)
            cond_border  = mask * cond_patches[:, 0:3] + (1. - mask) * torch.randn_like(cond_patches[:, 0:3]) * 0.5
            cond_patches = torch.cat([cond_patches, cond_border], dim=1)


        rnd_normal = torch.randn([gt_patches.shape[0], 1, 1, 1], device=gt_patches.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        assert augment_pipe is None
        augment_labels = None

        n = torch.randn_like(gt_patches) * sigma

        D_yn = net(gt_patches + n, cond_patches, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - gt_patches) ** 2)
        if self.use_lpips:
            loss += 5 * lpips_fn(D_yn, gt_patches)
        return loss


