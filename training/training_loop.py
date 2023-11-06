# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import random
import json
import pickle
import sys
import psutil
import lpips
import numpy as np
import torch
import dnnlib
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from training.utils import trim_files
from training.sample import generate_sample_grid_bos

torch.set_num_threads(8)

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    arch                = 'ddpmpp',
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()

    np_seed = (seed * dist.get_world_size() + dist.get_rank()) % (1 << 31)
    torch_seed = np.random.randint(1 << 31)
    np.random.seed(np_seed)
    random.seed(np_seed)

    dist.print0(f'NP seed: {np_seed}')
    dist.print0(f'torch seed: {torch_seed}')
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.cuda.manual_seed(torch_seed)

    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    train_dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    dist.print0('Loading test dataset...')
    dataset_kwargs.phase = 'test'
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    test_dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=2, **data_loader_kwargs))
    dataset_kwargs.phase = 'train'

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels,
                            label_dim=dataset_obj.label_dim)

    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)

    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=run_dir)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    if batch_gpu_total <= 4:
        step_factor = 10 if 'lowlight' in dataset_kwargs['dataset_name'] else 2
        step_factor = 100 if 'cats' in dataset_kwargs['dataset_name'] else step_factor
    else:
        if 'lowlight' in dataset_kwargs['dataset_name']:
            step_factor = 1200
        elif 'sony_tif' in dataset_kwargs['dataset_name']:
            step_factor = 1000
        elif 'lol' in dataset_kwargs['dataset_name']:
            step_factor = 900
        else:
            raise NotImplementedError
        if arch == 'adm':
            step_factor = step_factor // 2

    last_cur_nimg = cur_nimg
    with tqdm(total=total_kimg) as pbar:
        while True:

            # Accumulate gradients.
            optimizer.zero_grad(set_to_none=True)
            for round_idx in range(num_accumulation_rounds):
                with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                    data        = next(train_dataset_iterator)
                    cond_img    = data['cond_img'].to(device)
                    gt_img      = data['gt_img'].to(device)
                    cond_img_fr = data['cond_img_fr'].to(device)
                    gt_img_fr   = data['gt_img_fr'].to(device)
                    he          = data['he'].to(device)
                    he_fr       = data['he_fr'].to(device)
                    loss = loss_fn(net=ddp, gt_img=gt_img, cond_img=cond_img,
                                   gt_img_fr=gt_img_fr, cond_img_fr=cond_img_fr,
                                   he=he, he_fr=he_fr,
                                   labels=None, augment_pipe=None, lpips_fn=lpips_fn)
                    training_stats.report('Loss/loss', loss)

                    loss = loss.sum().mul(loss_scaling / batch_gpu_total)

                    if torch.isnan(loss):
                        sys.exit()
                    pbar.set_description(f'loss: {loss:.4f}')
                    if dist.get_rank() == 0 and ((cur_nimg-last_cur_nimg) > (batch_size * 1)):
                        writer.add_scalar(f'train/loss', loss, cur_nimg)
                    loss.backward()

            # Update weights.
            for g in optimizer.param_groups:
                g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            optimizer.step()
            pbar.update(batch_size)

            # Update EMA.
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
            for p_ema, p_net in zip(ema.parameters(), net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

            if ((cur_nimg-last_cur_nimg) > (batch_size * step_factor)) and dist.get_rank() == 0:

                out = generate_sample_grid_bos(net, test_dataset_iterator, lpips_fn=lpips_fn,
                                                      dataset_name=dataset_kwargs['dataset_name'],
                                                      curve=dataset_kwargs['curve'],
                                                      self_norm=dataset_kwargs['self_norm'],
                                                      scale_norm=loss_kwargs['scale_norm'],
                                               border_norm=loss_kwargs['border_norm'])
                writer.add_image(f'test/diff0',       out.diff0, cur_nimg)
                writer.add_image(f'test/diff1',       out.diff1, cur_nimg)
                writer.add_image(f'test/diff2',       out.diff2, cur_nimg)
                writer.add_image(f'test/diff3',       out.diff3, cur_nimg)
                writer.add_image(f'test/reconstruct', out.x_arr, cur_nimg)
                writer.add_scalar(f'test/psnr',       out.psnr, cur_nimg)
                writer.add_scalar(f'test/ssim',       out.ssim, cur_nimg)
                writer.add_scalar(f'test/lpips',      out.lpips, cur_nimg)
                last_cur_nimg = cur_nimg
                del out

                # Perform maintenance tasks once per tick.
            cur_nimg += batch_size
            done = (cur_nimg >= total_kimg * 1000)

            if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
                continue

            # Print status line, accumulating the same information in training_stats.
            tick_end_time = time.time()
            fields = []
            fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
            fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
            fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
            fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
            fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
            fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
            fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
            fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
            torch.cuda.reset_peak_memory_stats()
            dist.print0(' '.join(fields))

            # Check for abort.
            if (not done) and dist.should_stop():
                done = True
                dist.print0()
                dist.print0('Aborting...')

            # Save network snapshot.
            if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
                data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
                for key, value in data.items():
                    if isinstance(value, torch.nn.Module):
                        value = copy.deepcopy(value).eval().requires_grad_(False)
                        misc.check_ddp_consistency(value)
                        data[key] = value.cpu()
                    del value # conserve memory
                if dist.get_rank() == 0:
                    with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                        pickle.dump(data, f)
                    trim_files(run_dir, 'network-snapshot')
                del data # conserve memory

            # Save full dump of the training state.
            if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
                torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
                trim_files(run_dir, 'training-state')

            # Update logs.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
                stats_jsonl.flush()
            dist.update_progress(cur_nimg // 1000, total_kimg)

            # Update state.
            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time
            if done:
                break

    # Done.
    dist.print0()
    dist.print0('Exiting...')
