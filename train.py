# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import dnnlib
import shutil
import sys
from torch_utils import distributed as dist
from training import training_loop
from training.training_template import fill_template

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm|ddpmlp',   type=click.Choice(['ddpmpp', 'ncsnpp', 'adm', 'ddpmlp']), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       type=click.Choice(['vp', 've', 'edm']), default='edm', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=10000, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=8e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.0, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress (kimg/tick)', metavar='KIMG',    type=click.IntRange(min=1), default=15, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

# Model related
@click.option('--subsample',     help='Whether to subsample the raw data or not', metavar='BOOL',   type=bool, default=True, show_default=True)
@click.option('--curve',         help='Curve for data in and data out', metavar='srgb|linear',      type=click.Choice(['linear', 'srgb']), default='linear', show_default=True)
@click.option('--resolution',    help='resolution of testing images', metavar='INT',                type=int, default=32, show_default=True)
@click.option('--dataset',       help='name of dataset to use', metavar='sony|lol|cats|lowlight|lowlighthalf|sony_tif|lol2|sony_tif_crop|flo',   type=click.Choice(['sony', 'lol', 'cats', 'lowlight', 'lowlighthalf', 'sony_tif','flo', 'lol2', 'sony_tif_crop']), required=True, show_default=True)
@click.option('--self_norm',     help='Concat a HE', metavar='BOOL',                type=bool, default=False, show_default=True)
@click.option('--scale_norm',    help='Concat a previous output', metavar='BOOL',                   type=bool, default=False, show_default=True)
@click.option('--add_noise',     help='add noise to light version', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--use_lpips',     help='add lpips to loss', metavar='BOOL',                          type=bool, default=False, show_default=True)
@click.option('--border_norm', help='concats 2 pixels border', type=bool, default=False, show_default=True)


def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    print('ONLY ADDED COLOR AUGMENT TO LR AND BOS')
    c.arch = opts.arch


    if opts.resolution == 64:
        assert opts.dataset in ['lol', 'sony_tif']
    else:
        assert opts.resolution == 32

    if opts.dataset == 'lowlight':
        class_name = 'training.dataset.LowLightBosDataset'
    elif opts.dataset == 'flo':
        class_name = 'training.dataset.FloBosDataset'
    elif opts.dataset == 'lol':
        class_name = 'training.dataset.LOLBosDataset'
    elif opts.dataset == 'lol2':
        class_name = 'training.dataset.LOL2BosDataset'
    elif opts.dataset == 'sony_tif_crop':
        class_name = 'training.dataset.SonyTifCropBosDataset'
    elif opts.dataset == 'sony_tif':
        class_name = 'training.dataset.SonyTifBosDataset'
    else:
        raise NotImplementedError

    c.dataset_kwargs = dnnlib.EasyDict(class_name=class_name,
                                        path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache,
                                        phase='train', subsample=opts.subsample, curve=opts.curve,
                                       resolution=opts.resolution, dataset_name=opts.dataset, self_norm=opts.self_norm)
    c, opts, dataset_name = fill_template(c, opts, fullres=True)

    if opts.precond == 'edm':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.network_kwargs.scale_norm = opts.scale_norm
        c.network_kwargs.border_norm = opts.border_norm
        c.network_kwargs.self_norm = opts.self_norm
        c.loss_kwargs.class_name = 'training.loss.EDMBosLoss'
        c.loss_kwargs.self_norm = opts.self_norm
        c.loss_kwargs.add_noise = opts.add_noise
        c.loss_kwargs.scale_norm = opts.scale_norm
        c.loss_kwargs.use_lpips = opts.use_lpips
        c.loss_kwargs.border_norm = opts.border_norm
    else:
        raise NotImplementedError

    # Description string.
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'

    desc = f'{opts.dataset:s}-fullres-bos-{opts.arch:s}-{opts.precond:s}-{opts.curve:s}-' \
           f'-res{opts.resolution}x{opts.resolution}-noise{opts.add_noise}-llnorm{opts.self_norm}-' \
           f'scalenorm{opts.scale_norm}-borders{opts.border_norm}_lpips{opts.use_lpips}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]

        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]

        print(prev_run_ids)
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Curve:                   {opts.curve}')
    dist.print0(f'Subsample:               {opts.subsample}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

        shutil.make_archive(f'{c.run_dir}/training/', 'zip', './training')
        with open(f"{c.run_dir}/command.txt", 'w') as file:
            for row in sys.argv:
                file.write(row + '\n')

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
