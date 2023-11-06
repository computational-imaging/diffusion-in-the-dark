import skimage.io
import skimage.io
import torch
from easydict import EasyDict

import training.utils as utils
from training.constants import *


def read_image(path):
    img = skimage.io.imread(path)
    return torch.from_numpy(img).permute(2, 0, 1)[None, :] / 255.0


def process_lol(dark_path, light_path, curve, phase, pool, self_norm, resolution=None, eps=1e-8):
    dark_img = read_image(dark_path)
    light_img = read_image(light_path)

    ratio = light_img.mean() / dark_img.mean() * 2.0

    imgs = torch.cat([light_img, dark_img], dim=0)

    if curve == 'linear':
        imgs = utils.srgb_to_linear(imgs)

    if resolution == 32:
        imgs = pool(imgs)

    if resolution is not None:
        if phase == 'train':
            imgs = utils.random_crop(imgs, size=resolution)
        elif phase == 'test':
            imgs = utils.center_crop(imgs, size=resolution)

    light_img = imgs[0]
    dark_img = imgs[1]

    if curve == 'linear':
        light_img = light_img ** (1 / 4)
        light_img = utils.normalize(light_img, means=LOL_LIGHT_LINEAR_MEAN, stds=LOL_LIGHT_LINEAR_STD) / 2.0

        dark_img = dark_img ** (1 / 4)
        dark_img = utils.normalize(dark_img, means=LOL_DARK_LINEAR_MEAN, stds=LOL_DARK_LINEAR_STD) / 2.0
    elif curve == 'srgb':
        light_img = utils.normalize(light_img, means=LOL_LIGHT_SRGB_MEAN, stds=LOL_LIGHT_SRGB_STD) / 2.0

        dark_img = dark_img ** (1 / 4)
        dark_img = utils.normalize(dark_img, means=LOL_DARK_SRGB_MEAN, stds=LOL_DARK_SRGB_STD)

    return EasyDict({'cond_img': dark_img, 'gt_img': light_img, 'ratio': ratio})


def process_lowlight(dark_path, light_path, curve, pool, phase, self_norm, resolution=None, eps=1e-8):
    dark_img = read_image(dark_path)
    light_img = read_image(light_path)

    ratio = light_img.mean() / dark_img.mean()

    imgs = torch.cat([light_img, dark_img], dim=0)
    if curve == 'linear':
        imgs = utils.srgb_to_linear(imgs)

    imgs = pool(imgs)
    imgs_full = imgs.clone()

    if resolution is not None:
        if phase == 'train':
            imgs = utils.random_crop(imgs, size=resolution)
            # imgs = self.augment(imgs)
        elif phase == 'test':
            imgs = utils.center_crop(imgs, size=resolution)

    if self_norm:
        imgs_full = utils.random_crop(imgs_full, size=resolution)
        imgs_full = imgs_full ** (1/4)
        light_img_full = imgs_full[0]

        if curve == 'linear':
            light_img_full = utils.normalize(light_img_full, means=LL_LIGHT_LINEAR_MEAN, stds=LL_LIGHT_LINEAR_STD)
        elif curve == 'srgb':
            light_img_full = utils.normalize(light_img_full, means=LL_LIGHT_SRGB_MEAN, stds=LL_LIGHT_SRGB_STD)

    imgs = imgs ** (1 / 4.)

    light_img = imgs[0]
    dark_img = imgs[1]


    if curve == 'linear':
        light_img = utils.normalize(light_img, means=LL_LIGHT_LINEAR_MEAN, stds=LL_LIGHT_LINEAR_STD)
        dark_img = utils.normalize(dark_img, means=LL_DARK_LINEAR_MEAN1, stds=LL_DARK_LINEAR_STD1)
    elif curve == 'srgb':
        light_img = utils.normalize(light_img, means=LL_LIGHT_SRGB_MEAN, stds=LL_LIGHT_SRGB_STD)
        dark_img = utils.normalize(dark_img, means=LL_DARK_SRGB_MEAN1, stds=LL_DARK_SRGB_STD1)

    if self_norm:
        dark_img = torch.cat([dark_img, light_img_full], dim=0)

    dark_img /= 2.0
    light_img /= 2.0

    return EasyDict({'cond_img': dark_img, 'gt_img': light_img, 'ratio': ratio})

def t2n(img):
    return img.permute(1,2,0).detach().cpu().numpy()