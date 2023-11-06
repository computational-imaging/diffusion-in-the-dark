# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os

import numpy as np
import rawpy
import skimage.io
import torch.nn.functional as F
from debayer import Debayer3x3, Layout
from easydict import EasyDict
from einops import rearrange
from torchvision import transforms as T

import dnnlib
from training.augmenter import augmenter
from training.dataset_utils import *


# ----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 name,  # Name of the dataset.
                 raw_shape,  # Shape of the raw image data (NCHW).
                 max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
                 use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
                 xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
                 random_seed=0,  # Random seed to use when applying max_size.
                 cache=False,  # Cache images in CPU memory?
                 ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict()  # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return 3
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


# ----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class SonyDataset(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 use_pyspng=True,  # Use pyspng if available?
                 phase='.',
                 subsample=False,
                 curve='linear',
                 dataset_name='.',
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve

        assert phase == 'train' or phase == 'test'
        assert curve == 'linear' or curve == 'srgb'

        self._debayer = Debayer3x3(layout=Layout.RGGB)

        self._type = 'dir'
        self._dark_path = f'{self._path}/short'
        self._light_path = f'{self._path}/long'

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

        self._dark_fnames = sorted(
            [os.path.join(path, name) for path, subdirs, files in os.walk(self._dark_path) for name in files])
        if len(self._dark_fnames) == 0:
            raise IOError('No image files found in the specified path')

        num_imgs = len(self._dark_fnames)
        if self._phase == 'train':
            self._dark_fnames = self._dark_fnames[:int(num_imgs * 0.8)]
        elif self._phase == 'test':
            self._dark_fnames = self._dark_fnames[int(num_imgs * 0.8):]
        raw_shape = [len(self._dark_fnames)] + list(self._load_raw_image(0)['cond_img'].shape)

        super().__init__(name='sony', raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _pack_raw(self, dark_img):
        H, W, C = dark_img.shape
        dark_img = np.concatenate(
            (dark_img[0:H:2, 0:W:2, :],
             (dark_img[0:H:2, 1:W:2, :] + dark_img[1:H:2, 0:W:2, :]) / 2.,
             dark_img[1:H:2, 1:W:2, :]), axis=2
        )
        return dark_img

    def _load_raw_image(self, raw_idx):
        dark_path = self._dark_fnames[raw_idx]

        light_path = f'{self._light_path}/{dark_path.split("/")[-1][0:5]}_00_10s.ARW'
        if not os.path.exists(light_path):
            light_path = f'{self._light_path}/{dark_path.split("/")[-1][0:5]}_00_30s.ARW'

        dark_img = rawpy.imread(dark_path)  # RGGB
        dark_img = dark_img.raw_image_visible.astype(np.float32)
        dark_img = np.maximum(dark_img - 512, 0) / (16383 - 512)  # subtract black level
        dark_img = np.expand_dims(dark_img, axis=2)

        light_img = rawpy.imread(light_path)
        # opts = rawpy.Params(output_color=rawpy.ColorSpace.raw, no_auto_bright=True,
        #                     user_wb=[1.0, 1.0, 1.0, 1.0], gamma=(1, 1), output_bps=16)
        # light_img = light_img.postprocess(opts)
        light_img = light_img.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16, gamma=(1, 1))
        light_img = np.float32(light_img / 65535.0)
        light_img = torch.from_numpy(light_img)[None, ...]
        light_img = rearrange(light_img, 'b h w c -> b c h w')

        if self._subsample:
            dark_img = self._pack_raw(dark_img)
            b, c, h, w = light_img.shape
            light_img = F.interpolate(light_img, size=(h // 2, w // 2), mode='bilinear', align_corners=False)

            dark_img = torch.from_numpy(dark_img)[None, ...]
            dark_img = rearrange(dark_img, 'b h w c -> b c h w')
            imgs = torch.cat([light_img, dark_img], dim=0)
            imgs = F.interpolate(imgs, size=(1424 // 4, 2128 // 4), mode='bilinear', align_corners=False)
        else:
            dark_img = torch.from_numpy(dark_img)[None, :]
            dark_img = rearrange(dark_img, 'b h w c -> b c h w')  # a Bx1xHxW, [0..1], torch.float32 RGGB-Bayer tensor
            dark_img = self._debayer(dark_img)  # b, 3, h, w

            imgs = torch.cat([light_img, dark_img], dim=0)
            imgs = F.interpolate(imgs, size=(2848 // 8, 4256 // 8), mode='bilinear', align_corners=False)

        if self._curve == 'srgb':
            imgs = utils.linear_to_srgb(imgs)

        if self._phase == 'train':
            imgs = utils.random_crop(imgs, size=self._resolution)
            imgs = self.augment(imgs)
        elif self._phase == 'test':
            imgs = utils.center_crop(imgs, size=self._resolution)

        imgs = imgs ** (1 / 4)

        light_img = imgs[0]
        dark_img = imgs[1]

        if self._self_norm:
            dark_img_norm = dark_img.clone()
            idv_mean = [x.item() for x in torch.mean(dark_img_norm, dim=[1, 2])]
            idv_stds = [x.item() + self.eps for x in
                        torch.std(dark_img_norm, dim=[1, 2])]  # add-eps to prevent division by zero
            dark_img_norm = utils.normalize(dark_img_norm, means=idv_mean, stds=idv_stds)

        if self._curve == 'linear':
            light_img = utils.normalize(light_img, means=SONY_LIGHT_LINEAR_MEAN, stds=SONY_LIGHT_LINEAR_STD)
            dark_img = utils.normalize(dark_img, means=SONY_DARK_LINEAR_MEAN, stds=SONY_DARK_LINEAR_STD)
        elif self._curve == 'srgb':
            light_img = utils.normalize(light_img, means=SONY_LIGHT_SRGB_MEAN, stds=SONY_LIGHT_SRGB_STD)
            dark_img = utils.normalize(dark_img, means=SONY_DARK_SRGB_MEAN, stds=SONY_DARK_SRGB_STD)

        if self._self_norm:
            dark_img = torch.cat([dark_img, dark_img_norm], dim=0)

        ratio = 1.0
        dark_img /= 2.0
        light_img /= 2.0

        # image should be mean=0, std=0.5
        # range is ~[-1, 1]
        return EasyDict({'cond_img': dark_img, 'gt_img': light_img, 'ratio': ratio})

    def __getitem__(self, idx):
        assert self._phase in ['train', 'test']
        assert self._curve in ['srgb', 'linear']
        return self._load_raw_image(idx)


class SonyTifBosDataset(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 use_pyspng=True,  # Use pyspng if available?
                 phase='.',
                 subsample=False,
                 self_norm=False,
                 curve='linear',
                 dataset_name='.',
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve
        self.self_norm = self_norm

        assert phase in ['train', 'test', 'true_test']
        assert curve == 'linear' or curve == 'srgb'

        self._type = 'dir'
        self._dark_path = f'{self._path}/short_tif'
        self._light_path = f'{self._path}/long_tif'

        self.pool = torch.nn.AvgPool2d((4, 4))

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

        if self._phase == 'train':
            file_path = f'{self._path}/Sony_train_list.txt'
        elif self._phase == 'test':
            file_path = f'{self._path}/Sony_val_list.txt'
        elif self._phase == 'true_test':
            file_path = f'{self._path}/Sony_test_list.txt'

        assert os.path.isfile(file_path)
        with open(file_path) as file:
            lines = []
            while (line := file.readline().rstrip()):
                lines.append(line)

        input_fnames = []
        gt_fnames = []
        for i in range(len(lines)):
            items = lines[i].split(' ')
            input_fnames.append(items[0].split('/')[-1][:-4])
            gt_fnames.append(items[1].split('/')[-1][:-4])
        self.input_fnames = input_fnames
        self.gt_fnames = gt_fnames

        raw_shape = [len(self.input_fnames)] + list(self._load_raw_image(0)['cond_img'].shape)

        super().__init__(name='sony', raw_shape=raw_shape, **super_kwargs)

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def norm_imgs(self, light_img, dark_img):
        light_img = light_img ** (1 / 4)
        dark_img = dark_img ** (1 / 4)

        if self._curve == 'linear':
            light_img = utils.normalize(light_img, means=SONY_TIF_LIGHT_LINEAR_MEAN, stds=SONY_TIF_LIGHT_LINEAR_STD)
            dark_img = utils.normalize(dark_img, means=SONY_TIF_DARK_LINEAR_MEAN, stds=SONY_TIF_DARK_LINEAR_STD)
        elif self._curve == 'srgb':
            light_img = utils.normalize(light_img, means=SONY_TIF_LIGHT_SRGB_MEAN, stds=SONY_TIF_LIGHT_SRGB_STD)
            dark_img = utils.normalize(dark_img, means=SONY_TIF_DARK_SRGB_MEAN, stds=SONY_TIF_DARK_SRGB_STD)
        dark_img /= 2.0
        light_img /= 2.0
        return light_img, dark_img

    def _equalize_histogram(self, img):
        (r, g, b) = cv2.split(img)
        rh = cv2.equalizeHist(r)
        gh = cv2.equalizeHist(g)
        bh = cv2.equalizeHist(b)
        return cv2.merge((rh, gh, bh))

    def _load_raw_image(self, raw_idx):
        # dark_path = f'{self._dark_path}/{self.input_fnames[raw_idx]}'
        # light_path = f'{self._light_path}/{self.gt_fnames[raw_idx]}'

        file = self.input_fnames[raw_idx]
        dark_path = f'{self._dark_path}/{file}.tif'
        id = file

        gt_file = file.split('_')
        options = gt_file[:1] + ['00', '10s'] + gt_file[3:]

        if os.path.isfile(f'{self._light_path}/{("_").join(options)}.tif'):
            light_path = f'{self._light_path}/{("_").join(options)}.tif'
        else:
            options = gt_file[:1] + ['00', '30s'] + gt_file[3:]
            light_path = f'{self._light_path}/{("_").join(options)}.tif'

        dark_img = skimage.io.imread(dark_path)
        light_img = skimage.io.imread(light_path)

        he = self._equalize_histogram((dark_img * 255.0).astype(np.uint8))

        dark_img = torch.from_numpy(dark_img.astype(np.float32))
        light_img = torch.from_numpy(light_img.astype(np.float32))

        dark_img = rearrange(dark_img, 'h w c -> c h w')
        light_img = rearrange(light_img, 'h w c -> c h w')

        imgs = torch.stack([light_img, dark_img], dim=0)

        imgs = self.pool(imgs)

        assert self._resolution * 8 <= min(imgs.shape[-1], imgs.shape[-2])

        if self._phase == 'train':
            imgs = utils.random_crop(imgs, size=int(self._resolution * 8))
            imgs = self.augment(imgs)
        elif self._phase in ['test', 'true_test']:
            imgs = utils.center_crop(imgs, size=int(self._resolution * 8))

        ratio = light_img.mean() / dark_img.mean() / 20

        if self._curve == 'srgb':
            imgs = utils.linear_to_srgb(imgs)

        imgs_full = imgs.clone()
        imgs = F.interpolate(imgs, size=(self._resolution, self._resolution), antialias=True, align_corners=False,
                             mode='bilinear')

        light_img_fr, dark_img_fr = self.norm_imgs(imgs_full[0], imgs_full[1])
        light_img, dark_img = self.norm_imgs(imgs[0], imgs[1])

        return EasyDict({'cond_img': dark_img, 'gt_img': light_img,
                         'cond_img_fr': dark_img_fr, 'gt_img_fr': light_img_fr,
                         'he': he, 'he_fr': light_img_fr,
                         'ratio': ratio, 'dataset': 'sony_tif', 'names': id})

    def __getitem__(self, idx):
        assert self._phase in ['train', 'test', 'true_test']
        assert self._curve in ['srgb', 'linear']
        return self._load_raw_image(idx)


class LowLightDataset(Dataset):
    # Learning to Restore Low-Light Images via Decomposition-and-Enhancement
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 phase='.',
                 subsample=False,
                 curve='linear',
                 dataset_name='.',
                 self_norm=False,
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve

        assert phase == 'train' or phase == 'test'
        assert curve == 'linear' or curve == 'srgb'

        self._type = 'dir'
        self._dark_path = f'{self._path}/input'
        self._light_path = f'{self._path}/gt'

        self._self_norm = self_norm
        self.eps = 1e-8

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=1.0),
            T.RandomVerticalFlip(p=1.0),
        ])
        if resolution == 64:
            self.pool = torch.nn.AvgPool2d((4, 4))
        else:
            self.pool = torch.nn.AvgPool2d((8, 8))

        if phase == 'train':
            self._dark_path += '/train/1'
            self._light_path += '/train'
        else:
            self._dark_path += '/test/1'
            self._light_path += '/test'

        self._dark_fnames = sorted(
            [os.path.join(path, name) for path, subdirs, files in os.walk(self._dark_path) for name in files])
        assert len(self._dark_fnames) > 0
        raw_shape = [len(self._dark_fnames)] + list(self._load_raw_image(0)['cond_img'].shape)

        super().__init__(name='lowlight', raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        dark_path = self._dark_fnames[raw_idx]

        light_path = f'{self._light_path}/{dark_path.split("/")[-1][0:5]}_00_10s.png'
        if not os.path.exists(light_path):
            light_path = f'{self._light_path}/{dark_path.split("/")[-1][0:5]}_00_30s.png'

        # image should be mean=0, std=0.5
        # range is ~[-1, 1]
        return process_lowlight(dark_path, light_path, self._curve, self.pool, self._phase, self._resolution,
                                self._self_norm)

    def __getitem__(self, idx):
        assert self._phase in ['train', 'test']
        assert self._curve in ['srgb', 'linear']

        # dark_img, light_img, ratio, self._phase = self._load_raw_image(idx)
        # return light_img, ratio
        return self._load_raw_image(idx)


class LOLBosDataset(Dataset):
    # Learning to Restore Low-Light Images via Decomposition-and-Enhancement
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 phase='.',
                 subsample=False,
                 curve='linear',
                 dataset_name='.',
                 self_norm=False,
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve
        self._self_norm = self_norm

        assert phase == 'train' or phase == 'test'
        assert curve == 'linear' or curve == 'srgb'

        if phase == 'train':
            self.gt_path = f'{path}/our485/high'
            self.raw_path = f'{path}/our485/low'
        elif phase == 'test':
            self.gt_path = f'{path}/eval15/high'
            self.raw_path = f'{path}/eval15/low'

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

        self._path = path
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve

        self.dark_paths = sorted(
            [os.path.join(path, name) for path, subdirs, files in os.walk(self.raw_path) for name in files])
        self.dark_paths = [x for x in self.dark_paths if '.png' in x]
        assert len(self.dark_paths) > 0
        raw_shape = [len(self.dark_paths)] + list(self._load_raw_image(0)['cond_img'].shape)

        super().__init__(name='lol', raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def norm_imgs(self, img, img_type):
        assert img_type in ['light', 'dark', 'he']
        assert img.ndim == 3 and img.shape[0] == 3

        if self._curve == 'linear':
            if img_type == 'light':
                img = utils.normalize(img ** (1/4), means=LOL_LIGHT_LINEAR_MEAN, stds=LOL_LIGHT_LINEAR_STD) / 2.0
            elif img_type == 'dark':
                img = utils.normalize(img ** (1/4), means=LOL_DARK_LINEAR_MEAN, stds=LOL_DARK_LINEAR_STD) / 2.0
            elif img_type == 'he':
                img = utils.normalize(img, means=LOL_DARK_HE_LINEAR_MEAN, stds=LOL_DARK_HE_LINEAR_STD)
        elif self._curve == 'srgb':
            if img_type == 'light':
                img = utils.normalize(img, means=LOL_LIGHT_SRGB_MEAN, stds=LOL_LIGHT_SRGB_STD) / 2.0
            elif img_type == 'dark':
                img = utils.normalize(img ** (1/4), means=LOL_DARK_SRGB_MEAN, stds=LOL_DARK_SRGB_STD) / 2.0
            elif img_type == 'he':
                img = utils.normalize(img, means=LOL_DARK_HE_SRGB_MEAN, stds=LOL_DARK_HE_SRGB_STD)
        return img

    def _load_raw_image(self, index):
        dark_path = self.dark_paths[index]
        light_path = f'{self.gt_path}/{dark_path.split("/")[-1]}'
        id = dark_path.split("/")[-1]
        dark_img = skimage.io.imread(dark_path)
        light_img = skimage.io.imread(light_path)

        if self._curve == 'linear':
            dark_img_he = (utils.srgb_to_linear_np(dark_img / 255.0) * 255.0).astype(np.uint8)
        else:
            dark_img_he = dark_img

        he = utils.equalize_histogram(dark_img_he)

        ratio = light_img.mean() / dark_img.mean() * 2.0

        if self._phase == 'train':
            light_img = augmenter(image=light_img)

        light_img = torch.from_numpy(light_img).permute(2, 0, 1)[None, ...] / 255.0
        dark_img  = torch.from_numpy(dark_img).permute(2, 0, 1)[None, ...] / 255.0
        he = torch.from_numpy(he).permute(2, 0, 1)[None, ...] / 255.0

        imgs = torch.cat([light_img, dark_img, he, light_img], dim=0)

        if self._curve == 'linear':
            imgs = utils.srgb_to_linear(imgs)

        assert self._resolution * 8 <= min(imgs.shape[-1], imgs.shape[-2])

        if self._phase == 'train':
            imgs = utils.random_crop(imgs, size=int(self._resolution * 8))
            imgs = self.augment(imgs)
        elif self._phase == 'test':
            imgs = utils.center_crop(imgs, size=int(self._resolution * 8))

        og_light = imgs[0]
        imgs_full = imgs.clone()

        imgs = F.interpolate(imgs, size=(self._resolution), antialias=True, align_corners=False, mode='bilinear')

        light_img_fr = self.norm_imgs(imgs_full[0], img_type='light')
        dark_img_fr = self.norm_imgs(imgs_full[1], img_type='dark')
        light_img = self.norm_imgs(imgs[0], img_type='light')
        dark_img = self.norm_imgs(imgs[1], img_type='dark')

        he_fr = self.norm_imgs(imgs_full[2], img_type='he')
        he = self.norm_imgs(imgs[2], img_type='he')

        return EasyDict({'cond_img': dark_img, 'gt_img': light_img,
                         'cond_img_fr': dark_img_fr, 'gt_img_fr': light_img_fr,
                         'he': he, 'he_fr': he_fr,
                         'ratio': ratio, 'dataset': 'lol', 'names': id, 'og_light': og_light})

    def __getitem__(self, idx):
        assert self._phase in ['train', 'test']
        assert self._curve in ['srgb', 'linear']

        return self._load_raw_image(idx)


class LOLBosNoNormDataset(Dataset):
    # Learning to Restore Low-Light Images via Decomposition-and-Enhancement
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 phase='.',
                 subsample=False,
                 curve='linear',
                 dataset_name='.',
                 self_norm=False,
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve
        self._self_norm = self_norm

        assert phase == 'train' or phase == 'test'
        assert curve == 'linear' or curve == 'srgb'

        if phase == 'train':
            self.gt_path = f'{path}/our485/high'
            self.raw_path = f'{path}/our485/low'
        elif phase == 'test':
            self.gt_path = f'{path}/eval15/high'
            self.raw_path = f'{path}/eval15/low'

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

        self._path = path
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve

        self.dark_paths = sorted(
            [os.path.join(path, name) for path, subdirs, files in os.walk(self.raw_path) for name in files])
        self.dark_paths = [x for x in self.dark_paths if '.png' in x]
        assert len(self.dark_paths) > 0
        raw_shape = [len(self.dark_paths)] + list(self._load_raw_image(0)['cond_img'].shape)

        super().__init__(name='lol', raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def norm_imgs(self, img, img_type):
        assert img_type in ['light', 'dark', 'he']
        assert img.ndim == 3 and img.shape[0] == 3

        return img * 2.0 - 1.0


    def _load_raw_image(self, index):
        dark_path = self.dark_paths[index]
        light_path = f'{self.gt_path}/{dark_path.split("/")[-1]}'
        id = dark_path.split("/")[-1]
        dark_img = skimage.io.imread(dark_path)
        light_img = skimage.io.imread(light_path)

        if self._curve == 'linear':
            dark_img_he = (utils.srgb_to_linear_np(dark_img / 255.0) * 255.0).astype(np.uint8)
        else:
            dark_img_he = dark_img

        he = utils.equalize_histogram(dark_img_he)

        ratio = light_img.mean() / dark_img.mean() * 2.0

        if self._phase == 'train':
            light_img = augmenter(image=light_img)

        light_img = torch.from_numpy(light_img).permute(2, 0, 1)[None, ...] / 255.0
        dark_img  = torch.from_numpy(dark_img).permute(2, 0, 1)[None, ...] / 255.0
        he = torch.from_numpy(he).permute(2, 0, 1)[None, ...] / 255.0

        imgs = torch.cat([light_img, dark_img, he, light_img], dim=0)

        if self._curve == 'linear':
            imgs = utils.srgb_to_linear(imgs)

        assert self._resolution * 8 <= min(imgs.shape[-1], imgs.shape[-2])

        if self._phase == 'train':
            imgs = utils.random_crop(imgs, size=int(self._resolution * 8))
            imgs = self.augment(imgs)
        elif self._phase == 'test':
            imgs = utils.center_crop(imgs, size=int(self._resolution * 8))

        og_light = imgs[0]
        imgs_full = imgs.clone()

        imgs = F.interpolate(imgs, size=(self._resolution), antialias=True, align_corners=False, mode='bilinear')

        light_img_fr = self.norm_imgs(imgs_full[0], img_type='light')
        dark_img_fr = self.norm_imgs(imgs_full[1], img_type='dark')
        light_img = self.norm_imgs(imgs[0], img_type='light')
        dark_img = self.norm_imgs(imgs[1], img_type='dark')

        he_fr = self.norm_imgs(imgs_full[2], img_type='he')
        he = self.norm_imgs(imgs[2], img_type='he')

        return EasyDict({'cond_img': dark_img, 'gt_img': light_img,
                         'cond_img_fr': dark_img_fr, 'gt_img_fr': light_img_fr,
                         'he': he, 'he_fr': he_fr,
                         'ratio': ratio, 'dataset': 'lol', 'names': id, 'og_light': og_light})

    def __getitem__(self, idx):
        assert self._phase in ['train', 'test']
        assert self._curve in ['srgb', 'linear']

        return self._load_raw_image(idx)


class LOL2BosDataset(Dataset):
    # Learning to Restore Low-Light Images via Decomposition-and-Enhancement
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 phase='.',
                 subsample=False,
                 curve='linear',
                 dataset_name='.',
                 self_norm=False,
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve
        self._self_norm = self_norm

        assert phase in ['train', 'test', 'test256']
        assert curve == 'linear' or curve == 'srgb'

        if phase == 'train':
            self.gt_path = f'{path}/Real_captured/Train/Normal'
            self.raw_path = f'{path}/Real_captured/Train/Low'
        elif phase == 'test':
            self.gt_path = f'{path}/Real_captured/Test/Normal'
            self.raw_path = f'{path}/Real_captured/Test/Low'
        elif phase == 'test256':
            self.gt_path = f'{path}/Normal'
            self.raw_path = f'{path}/Low'

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

        self._path = path
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve

        self.dark_paths = sorted([os.path.join(path, name) for path, subdirs, files in os.walk(self.raw_path) for name in files])
        self.dark_paths = [x for x in self.dark_paths if '.png' in x]


        assert len(self.dark_paths) > 0
        raw_shape = [len(self.dark_paths)] + list(self._load_raw_image(0)['cond_img'].shape)

        super().__init__(name='lol', raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def norm_imgs(self, img, img_type):
        assert img_type in ['light', 'dark', 'he']
        assert img.ndim == 3 and img.shape[0] == 3

        if self._curve == 'linear':
            if img_type == 'light':
                img = utils.normalize(img ** (1/4), means=LOL_LIGHT_LINEAR_MEAN, stds=LOL_LIGHT_LINEAR_STD) / 2.0
            elif img_type == 'dark':
                img = utils.normalize(img ** (1/4), means=LOL_DARK_LINEAR_MEAN, stds=LOL_DARK_LINEAR_STD) / 2.0
            elif img_type == 'he':
                img = utils.normalize(img, means=LOL_DARK_HE_LINEAR_MEAN, stds=LOL_DARK_HE_LINEAR_STD)
        elif self._curve == 'srgb':
            if img_type == 'light':
                img = utils.normalize(img, means=LOL_LIGHT_SRGB_MEAN, stds=LOL_LIGHT_SRGB_STD) / 2.0
            elif img_type == 'dark':
                img = utils.normalize(img ** (1/4), means=LOL_DARK_SRGB_MEAN, stds=LOL_DARK_SRGB_STD) / 2.0
            elif img_type == 'he':
                img = utils.normalize(img, means=LOL_DARK_HE_SRGB_MEAN, stds=LOL_DARK_HE_SRGB_STD)
        return img

    def _load_raw_image(self, index):
        dark_path = self.dark_paths[index]
        id = dark_path.split("/")[-1][3:]
        light_path = f'{self.gt_path}/normal{id}'
        dark_img = skimage.io.imread(dark_path)

        light_img = skimage.io.imread(light_path)

        if self._curve == 'linear':
            dark_img_he = (utils.srgb_to_linear_np(dark_img / 255.0) * 255.0).astype(np.uint8)
        else:
            dark_img_he = dark_img

        # he = utils.equalize_histogram(dark_img_he)

        ratio = light_img.mean() / dark_img.mean() * 2.0

        if self._phase == 'train':
            light_img = augmenter(image=light_img)

        light_img = torch.from_numpy(light_img).permute(2, 0, 1)[None, ...] / 255.0
        dark_img  = torch.from_numpy(dark_img).permute(2, 0, 1)[None, ...] / 255.0

        ### HE IS COLORMAP
        he = dark_img.exp() / (dark_img.exp().sum(dim=1, keepdims=True) + 1e-4)
        he = torch.from_numpy(he).permute(2, 0, 1)[None, ...] / 255.0

        imgs = torch.cat([light_img, dark_img, he], dim=0)

        if self._curve == 'linear':
            imgs = utils.srgb_to_linear(imgs)

        assert self._resolution * 8 <= min(imgs.shape[-1], imgs.shape[-2])

        if self._phase == 'train':
            imgs = utils.random_crop(imgs, size=int(self._resolution * 8))
            imgs = self.augment(imgs)
        elif self._phase == 'test':
            imgs = utils.center_crop(imgs, size=int(self._resolution * 8))

        og_light = imgs[0]
        imgs_full = imgs.clone()

        imgs = F.interpolate(imgs, size=(self._resolution), antialias=True, align_corners=False, mode='bilinear')

        light_img_fr = self.norm_imgs(imgs_full[0], img_type='light')
        dark_img_fr = self.norm_imgs(imgs_full[1], img_type='dark')
        light_img = self.norm_imgs(imgs[0], img_type='light')
        dark_img = self.norm_imgs(imgs[1], img_type='dark')

        he_fr = imgs_full[2] * 2 - 1
        he = imgs[2] * 2 - 1

        # he_fr = self.norm_imgs(imgs_full[2], img_type='he')
        # he = self.norm_imgs(imgs[2], img_type='he')

        return EasyDict({'cond_img': dark_img, 'gt_img': light_img,
                         'cond_img_fr': dark_img_fr, 'gt_img_fr': light_img_fr,
                         'he': he, 'he_fr': he_fr,
                         'ratio': ratio, 'dataset': 'lol', 'names': id, 'og_light': og_light})

    def __getitem__(self, idx):
        assert self._phase in ['train', 'test', 'test256']
        assert self._curve in ['srgb', 'linear']

        return self._load_raw_image(idx)


class SonyTifCropBosDataset(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 use_pyspng=True,  # Use pyspng if available?
                 phase='.',
                 subsample=False,
                 self_norm=False,
                 curve='linear',
                 dataset_name='.',
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve
        self.self_norm = self_norm

        assert phase in ['train', 'test', 'true_test']
        assert curve == 'linear' or curve == 'srgb'

        self._type = 'dir'
        self._dark_path = f'{self._path}/short_tif_crop_test'
        self._light_path = f'{self._path}/long_tif_crop_test'

        self.pool = torch.nn.AvgPool2d((2, 2))  # 2848, 4256

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

        if self._phase == 'train':
            file_path = f'{self._path}/Sony_train_list.txt'
        elif self._phase == 'test':
            file_path = f'{self._path}/Sony_val_list.txt'
        elif self._phase =='true_test':
            file_path = f'{self._path}/Sony_test_list.txt'

        assert os.path.isfile(file_path)
        with open(file_path) as file:
            lines = []
            while (line := file.readline().rstrip()):
                lines.append(line)

        input_fnames = []
        gt_fnames = []
        for i in range(len(lines)):
            items = lines[i].split(' ')
            input_fnames.append(items[0].split('/')[-1][:-4])
            gt_fnames.append(items[1].split('/')[-1][:-4])

        self.input_files = sorted([f for f in os.listdir(self._dark_path) if os.path.isfile(os.path.join(self._dark_path, f))])
        self.gt_files = sorted([f for f in os.listdir(self._light_path) if os.path.isfile(os.path.join(self._light_path, f))])
        self.input_fnames = [k for k in self.input_files if f'{("_").join(k.split("_")[:-1])}' in input_fnames]
        self.gt_fnames = [k for k in self.gt_files if f'{("_").join(k.split("_")[:-1])}' in gt_fnames]

        raw_shape = [len(self.input_fnames)] + list(self._load_raw_image(0)['cond_img'].shape)

        super().__init__(name='sony', raw_shape=raw_shape, **super_kwargs)

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def norm_imgs(self, img, img_type):
        img = img ** (1/4)
        if self._curve == 'linear':
            if img_type == 'light':
                img = utils.normalize(img, means=SONY_TIF_LIGHT_LINEAR_MEAN, stds=SONY_TIF_LIGHT_LINEAR_STD)
            elif img_type == 'dark':
                img = utils.normalize(img, means=SONY_TIF_DARK_LINEAR_MEAN, stds=SONY_TIF_DARK_LINEAR_STD)
        elif self._curve == 'srgb':
            if img_type == 'light':
                img = utils.normalize(img, means=SONY_TIF_LIGHT_SRGB_MEAN, stds=SONY_TIF_LIGHT_SRGB_STD)
            elif img_type == 'dark':
                img = utils.normalize(img, means=SONY_TIF_DARK_SRGB_MEAN, stds=SONY_TIF_DARK_SRGB_STD)

        return img / 2.0

    def _equalize_histogram(self, img):
        (r, g, b) = cv2.split(img)
        rh = cv2.equalizeHist(r)
        gh = cv2.equalizeHist(g)
        bh = cv2.equalizeHist(b)
        return cv2.merge((rh, gh, bh))

    def _load_raw_image(self, raw_idx):

        file = self.input_fnames[raw_idx]
        dark_path = f'{self._dark_path}/{file}'
        id = file

        gt_file = file.split('_')
        options = gt_file[:1] + ['00', '10s'] + gt_file[3:]

        if os.path.isfile(f'{self._light_path}/{("_").join(options)}'):
            light_path = f'{self._light_path}/{("_").join(options)}'
        else:
            options = gt_file[:1] + ['00', '30s'] + gt_file[3:]
            light_path = f'{self._light_path}/{("_").join(options)}'

        dark_img = skimage.io.imread(dark_path)
        light_img = skimage.io.imread(light_path)

        he = self._equalize_histogram((dark_img * 255.0).astype(np.uint8)) / 255.0

        dark_img  = torch.from_numpy(dark_img.astype(np.float32))
        light_img = torch.from_numpy(light_img.astype(np.float32))
        he = torch.from_numpy(he.astype(np.float32))

        dark_img = rearrange(dark_img, 'h w c -> c h w')
        light_img = rearrange(light_img, 'h w c -> c h w')
        he = rearrange(he, 'h w c -> c h w')

        imgs = torch.stack([light_img, dark_img, he], dim=0)

        imgs = self.pool(imgs)

        assert self._resolution * 8 <= min(imgs.shape[-1], imgs.shape[-2])

        if self._phase == 'train':
            # imgs = utils.random_crop(imgs, size=int(self._resolution * 8))
            imgs = self.augment(imgs)
        # elif self._phase == 'test':
            # imgs = utils.center_crop(imgs, size=int(self._resolution * 8))


        ratio = light_img.mean() / dark_img.mean() / 20

        if self._curve == 'srgb':
            imgs = utils.linear_to_srgb(imgs)

        imgs_full = imgs.clone()
        imgs = F.interpolate(imgs, size=(self._resolution, self._resolution), antialias=True, align_corners=False, mode='bilinear')

        light_img_fr = self.norm_imgs(imgs_full[0], 'light')
        dark_img_fr = self.norm_imgs(imgs_full[1], 'dark')
        light_img = self.norm_imgs(imgs[0], 'light')
        dark_img = self.norm_imgs(imgs[1], 'dark')
        he_fr = self.norm_imgs(imgs_full[2], 'light')
        he = self.norm_imgs(imgs[2], 'light')

        # light_img_fr, dark_img_fr = self.norm_imgs(imgs_full[0], imgs_full[1])
        # light_img, dark_img = self.norm_imgs(imgs[0], imgs[1])

        return EasyDict({'cond_img': dark_img, 'gt_img': light_img,
                         'cond_img_fr': dark_img_fr, 'gt_img_fr': light_img_fr,
                         'he': he, 'he_fr': he_fr,
                         'ratio': ratio, 'dataset': 'sony_tif', 'names': id})

    def __getitem__(self, idx):
        assert self._phase in ['train', 'test', 'true_test']
        assert self._curve in ['srgb', 'linear']
        return self._load_raw_image(idx)


class LOLDatasetLR(Dataset):
    # Learning to Restore Low-Light Images via Decomposition-and-Enhancement
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 phase='.',
                 subsample=False,
                 curve='linear',
                 dataset_name='.',
                 self_norm=False,
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve
        self._self_norm = self_norm

        assert phase == 'train' or phase == 'test'
        assert curve == 'linear' or curve == 'srgb'

        if phase == 'train':
            self.gt_path = f'{path}/our485/high'
            self.raw_path = f'{path}/our485/low'
        elif phase == 'test':
            self.gt_path = f'{path}/eval15/high'
            self.raw_path = f'{path}/eval15/low'

        self._path = path
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

        # self.pool = torch.nn.AvgPool2d((2, 2))

        self.dark_paths = sorted(
            [os.path.join(path, name) for path, subdirs, files in os.walk(self.raw_path) for name in files])
        self.dark_paths = [x for x in self.dark_paths if '.png' in x]
        assert len(self.dark_paths) > 0
        raw_shape = [len(self.dark_paths)] + list(self._load_raw_image(0)['cond_img'].shape)

        super().__init__(name='lol', raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _norm_imgs(self, imgs):

        light_img = imgs[0]
        dark_img = imgs[1]

        if self._curve == 'linear':
            light_img = light_img ** (1 / 4)
            light_img = utils.normalize(light_img, means=LOL_LIGHT_LINEAR_MEAN, stds=LOL_LIGHT_LINEAR_STD) / 2.0

            dark_img = dark_img ** (1 / 4)
            dark_img = utils.normalize(dark_img, means=LOL_DARK_LINEAR_MEAN, stds=LOL_DARK_LINEAR_STD) / 2.0
        elif self._curve == 'srgb':
            light_img = utils.normalize(light_img, means=LOL_LIGHT_SRGB_MEAN, stds=LOL_LIGHT_SRGB_STD) / 2.0

            dark_img = dark_img ** (1 / 4)
            dark_img = utils.normalize(dark_img, means=LOL_DARK_SRGB_MEAN, stds=LOL_DARK_SRGB_STD) / 2.0

        return dark_img, light_img

    def _load_raw_image(self, index):
        dark_path = self.dark_paths[index]
        light_path = f'{self.gt_path}/{dark_path.split("/")[-1]}'

        dark_img = skimage.io.imread(dark_path)
        light_img = skimage.io.imread(light_path)
        ratio = light_img.mean() / dark_img.mean() * 2.0

        if self._phase == 'train':
            light_img = augmenter(image=light_img)

        light_img = torch.from_numpy(light_img).permute(2, 0, 1) / 255.0
        dark_img = torch.from_numpy(dark_img).permute(2, 0, 1) / 255.0

        imgs = torch.stack([light_img, dark_img], dim=0)
        # full image is going to be a square crop
        down_size = 256
        if self._phase == 'train':
            imgs = utils.random_crop(imgs, size=down_size)
        elif self._phase == 'test':
            imgs = utils.center_crop(imgs, size=down_size)

        imgs = utils.srgb_to_linear(imgs) if self._curve == 'linear' else imgs
        imgs = self.augment(imgs) if self._phase == 'train' else imgs

        lr_imgs = F.interpolate(imgs, size=(32, 32), align_corners=False, mode='bilinear', antialias=True)

        dark_img, light_img = self._norm_imgs(lr_imgs)
        return EasyDict({'cond_img': dark_img, 'gt_img': light_img, 'ratio': ratio})

    def __getitem__(self, idx):
        assert self._phase in ['train', 'test']
        assert self._curve in ['srgb', 'linear']
        return self._load_raw_image(idx)


class LowLightDatasetLR(Dataset):
    # Learning to Restore Low-Light Images via Decomposition-and-Enhancement
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 phase='.',
                 subsample=False,
                 curve='linear',
                 dataset_name='.',
                 self_norm=False,
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve

        assert phase == 'train' or phase == 'test'
        assert curve == 'linear' or curve == 'srgb'

        self._type = 'dir'
        self._dark_path = f'{self._path}/input'
        self._light_path = f'{self._path}/gt'

        self._self_norm = self_norm
        self.eps = 1e-8

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

        self.pool = torch.nn.AvgPool2d((2, 2))

        if phase == 'train':
            self._dark_path += '/train/1'
            self._light_path += '/train'
        else:
            self._dark_path += '/test/1'
            self._light_path += '/test'

        self._dark_fnames = sorted(
            [os.path.join(path, name) for path, subdirs, files in os.walk(self._dark_path) for name in files])
        assert len(self._dark_fnames) > 0
        raw_shape = [len(self._dark_fnames)] + list(self._load_raw_image(0)['cond_img'].shape)

        super().__init__(name='lowlight', raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _norm_imgs(self, imgs):
        imgs = imgs ** (1 / 4.)

        light_img = imgs[0]
        dark_img = imgs[1]

        if self._curve == 'linear':
            light_img = utils.normalize(light_img, means=LL_LIGHT_LINEAR_MEAN, stds=LL_LIGHT_LINEAR_STD)
            dark_img = utils.normalize(dark_img, means=LL_DARK_LINEAR_MEAN1, stds=LL_DARK_LINEAR_STD1)
        elif self._curve == 'srgb':
            light_img = utils.normalize(light_img, means=LL_LIGHT_SRGB_MEAN, stds=LL_LIGHT_SRGB_STD)
            dark_img = utils.normalize(dark_img, means=LL_DARK_SRGB_MEAN1, stds=LL_DARK_SRGB_STD1)

        dark_img /= 2.0
        light_img /= 2.0
        return dark_img, light_img

    def _load_raw_image(self, raw_idx):
        dark_path = self._dark_fnames[raw_idx]

        light_path = f'{self._light_path}/{dark_path.split("/")[-1][0:5]}_00_10s.png'
        if not os.path.exists(light_path):
            light_path = f'{self._light_path}/{dark_path.split("/")[-1][0:5]}_00_30s.png'

        dark_img = skimage.io.imread(dark_path)
        light_img = skimage.io.imread(light_path)
        ratio = light_img.mean() / dark_img.mean()

        if self._phase == 'train':
            light_img = augmenter(image=light_img)

        imgs = torch.cat([light_img, dark_img], dim=0)
        imgs = utils.srgb_to_linear(imgs) if self._curve == 'linear' else imgs
        imgs = self.augment(imgs) if self._phase == 'train' else imgs

        imgs = self.pool(imgs)

        # full image is going to be a square crop
        down_size = 256
        if self._phase == 'train':
            imgs = utils.random_crop(imgs, size=down_size)
        elif self._phase == 'test':
            imgs = utils.center_crop(imgs, size=down_size)

        hr_imgs = imgs.clone()
        lr_imgs = F.interpolate(imgs, size=(32, 32), align_corners=False, mode='bilinear', antialias=True)

        dark_img, light_img = self._norm_imgs(lr_imgs)
        dark_hr, light_hr = self._norm_imgs(hr_imgs)

        return EasyDict({'cond_img': dark_img, 'gt_img': light_img, 'ratio': ratio})

    def __getitem__(self, idx):
        assert self._phase in ['train', 'test']
        assert self._curve in ['srgb', 'linear']

        return self._load_raw_image(idx)


class LowLightBosDataset(Dataset):
    # Learning to Restore Low-Light Images via Decomposition-and-Enhancement
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 phase='.',
                 subsample=False,
                 curve='linear',
                 dataset_name='.',
                 self_norm=False,
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve

        assert phase == 'train' or phase == 'test'
        assert curve == 'linear' or curve == 'srgb'

        self._type = 'dir'
        self._dark_path = f'{self._path}/input'
        self._light_path = f'{self._path}/gt'

        self._self_norm = self_norm
        self.eps = 1e-8

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])

        self.pool = torch.nn.AvgPool2d((2, 2))

        if phase == 'train':
            self._dark_path += '/train/1'
            self._light_path += '/train'
        else:
            self._dark_path += '/test/1'
            self._light_path += '/test'

        self._dark_fnames = sorted(
            [os.path.join(path, name) for path, subdirs, files in os.walk(self._dark_path) for name in files])
        assert len(self._dark_fnames) > 0
        raw_shape = [len(self._dark_fnames)] + list(self._load_raw_image(0)['cond_img'].shape)

        super().__init__(name='lowlight', raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _norm_imgs(self, imgs):
        imgs = imgs ** (1 / 4.)

        light_img = imgs[0]
        dark_img = imgs[1]

        if self._curve == 'linear':
            light_img = utils.normalize(light_img, means=LL_LIGHT_LINEAR_MEAN, stds=LL_LIGHT_LINEAR_STD)
            dark_img = utils.normalize(dark_img, means=LL_DARK_LINEAR_MEAN1, stds=LL_DARK_LINEAR_STD1)
        elif self._curve == 'srgb':
            light_img = utils.normalize(light_img, means=LL_LIGHT_SRGB_MEAN, stds=LL_LIGHT_SRGB_STD)
            dark_img = utils.normalize(dark_img, means=LL_DARK_SRGB_MEAN1, stds=LL_DARK_SRGB_STD1)

        dark_img /= 2.0
        light_img /= 2.0
        return dark_img, light_img

    def _load_raw_image(self, raw_idx):
        dark_path = self._dark_fnames[raw_idx]

        if '256' in self._dark_path:
            light_path = f'{self._light_path}/{dark_path.split("/")[-1]}'
        else:
            light_path = f'{self._light_path}/{dark_path.split("/")[-1][0:5]}_00_10s.png'
            if not os.path.exists(light_path):
                light_path = f'{self._light_path}/{dark_path.split("/")[-1][0:5]}_00_30s.png'
        id = dark_path.split("/")[-1]

        dark_img = skimage.io.imread(dark_path)
        light_img = skimage.io.imread(light_path)
        ratio = light_img.mean() / dark_img.mean()

        if self._phase == 'train':
            light_img = augmenter(image=light_img)

        light_img = torch.from_numpy(light_img).permute(2, 0, 1)[None, ...] / 255.0
        dark_img = torch.from_numpy(dark_img).permute(2, 0, 1)[None, ...] / 255.0
        imgs = torch.cat([light_img, dark_img], dim=0)
        imgs = utils.srgb_to_linear(imgs) if self._curve == 'linear' else imgs
        imgs = self.augment(imgs) if self._phase == 'train' else imgs

        if '256' not in self._dark_path:
            imgs = self.pool(imgs)

        # full image is going to be a square crop
        down_size = 256
        if self._phase == 'train':
            imgs = utils.random_crop(imgs, size=down_size)
        elif self._phase == 'test':
            imgs = utils.center_crop(imgs, size=down_size)

        hr_imgs = imgs.clone()
        lr_imgs = F.interpolate(imgs, size=(32, 32), align_corners=False, mode='bilinear', antialias=True)

        dark_img, light_img = self._norm_imgs(lr_imgs)
        dark_hr, light_hr = self._norm_imgs(hr_imgs)

        return EasyDict({'cond_img': dark_img, 'gt_img': light_img,
                         'cond_img_fr': dark_hr, 'gt_img_fr': light_hr, 'ratio': ratio, 'dataset': 'lowlight',
                         'he_fr': light_hr, 'he': light_img,
                         'names': id})

    def __getitem__(self, idx):
        assert self._phase in ['train', 'test']
        assert self._curve in ['srgb', 'linear']

        return self._load_raw_image(idx)



class LOLDataset(Dataset):
    # Learning to Restore Low-Light Images via Decomposition-and-Enhancement
    def __init__(self,
                 path,  # Path to directory or zip.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 phase='.',
                 subsample=False,
                 curve='linear',
                 dataset_name='.',
                 self_norm=False,
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve
        self._self_norm = self_norm

        assert phase == 'train' or phase == 'test'
        assert curve == 'linear' or curve == 'srgb'

        if phase == 'train':
            self.gt_path = f'{path}/our485/high'
            self.raw_path = f'{path}/our485/low'
        elif phase == 'test':
            self.gt_path = f'{path}/eval15/high'
            self.raw_path = f'{path}/eval15/low'

        self._path = path
        self._resolution = resolution
        self._phase = phase
        self._subsample = subsample
        self._curve = curve

        if self._resolution == 32:
            self.pool = torch.nn.AvgPool2d((2, 2))

        self.dark_paths = sorted(
            [os.path.join(path, name) for path, subdirs, files in os.walk(self.raw_path) for name in files])
        self.dark_paths = [x for x in self.dark_paths if '.png' in x]
        assert len(self.dark_paths) > 0
        raw_shape = [len(self.dark_paths)] + list(self._load_raw_image(0)['cond_img'].shape)

        super().__init__(name='lol', raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, index):
        dark_path = self.dark_paths[index]
        light_path = f'{self.gt_path}/{dark_path.split("/")[-1]}'

        if not hasattr(self, 'pool'):
            self.pool = torch.nn.AvgPool2d((2, 2))


        # image should be mean=0, std=0.5
        # range is ~[-1, 1]
        return process_lol(dark_path, light_path, self._curve, self._phase, self._resolution, self.pool, self._self_norm)

    def __getitem__(self, idx):
        assert self._phase in ['train', 'test']
        assert self._curve in ['srgb', 'linear']
        return self._load_raw_image(idx)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
