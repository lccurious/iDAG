# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import time
import torch
import os
from nvitop import select_devices


def local_launcher(commands, mem_usage='5GiB', num_parallel=8):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)


def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')


def multi_gpu_launcher(commands, mem_usage='5GiB', num_parallel=8):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        # available_gpus = select_devices(format='index', min_count=1, min_free_memory=mem_usage)
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                print(cmd)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()


def multi_available_gpu_launcher(commands, mem_usage='5GiB', num_parallel=8, launch_delay=5):
    procs_by_queue = [None] * num_parallel
    n_gpus = len(select_devices(format='index', min_count=1))
    try:
        allow_gpus = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        allow_gpus = [x for x in range(n_gpus)]
    print('CUDA_VISIBLE_DEVICES is', allow_gpus)
    while len(commands) > 0:
        for idx, proc in enumerate(procs_by_queue):
            available_gpus = select_devices(format='index', min_count=1, min_free_memory=mem_usage)
            gpu_idx = idx % n_gpus
            if (gpu_idx in available_gpus and gpu_idx in allow_gpus) and (proc is None or proc.poll() is not None):
                # if this gpus has enough memory; launch a command
                cmd = commands.pop(0)
                print(cmd)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True
                )
                procs_by_queue[idx] = new_proc
                time.sleep(60)
                break
        time.sleep(launch_delay)

    # wait for the last few tasks to finish before returning
    for p in procs_by_queue:
        if p is not None:
            p.wait()


REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    'multi_available_gpu': multi_available_gpu_launcher,
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
