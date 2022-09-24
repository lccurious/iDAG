import os
import builtins
import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torch.multiprocessing as mp
import torchvision
from sconf import Config
from prettytable import PrettyTable

from domainbed.datasets import get_dataset
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.logger import Logger
from domainbed.mp_trainer import main_worker


def main():
    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--data_dir", type=str, default="datadir/")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and random_hparams).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of steps. Default is dataset-dependent."
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps. Default is dataset-dependent.",
    )
    parser.add_argument("-j", "--workers", default=24, type=int, metavar='N',
                        help="number of data loading workers (default: 24)")
    parser.add_argument("-b", "--batch-size", default=32, type=int,
                        metavar='N',
                        help="mini-batch size (default: 32), this is the total "
                             "batch size of all GPUs on the current node when "
                             "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--multiprocessing_distributed", action="store_true",
                        help="Use multi-processing distributed training to launch "
                             "N processes per node, which has N GPUs. This is the "
                             "fastest way to use PyTorch for either single node or "
                             "multi node data parallel training.")
    parser.add_argument("--world_size", default=-1, type=int,
                        help="number of nodes for distributed training")
    parser.add_argument("--rank", default=-1, type=int,
                        help="node rank for distributed training")
    parser.add_argument("--dist_url", default="tcp://10.212.48.205:23456", type=str,
                        help="url used to set up distributed training")
    parser.add_argument("--dist_backend", default="nccl", type=str,
                        help="distributed backend")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use")
    parser.add_argument("--test_envs", type=int, nargs="+", default=None)  # sketch in PACS
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--model_save", default=None, type=int, help="Model save start step")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tb_freq", default=10)
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")
    parser.add_argument("--show", action="store_true", help="Show args and hparams w/o run")
    parser.add_argument(
        "--evalmode",
        default="fast",
        help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",
    )
    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")
    args, left_argv = parser.parse_known_args()

    # setup hparams
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)

    keys = ["config.yaml"] + args.configs
    keys = [open(key, encoding="utf8") for key in keys]
    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv)

    # setup debug
    if args.debug:
        args.checkpoint_freq = 5
        args.steps = 10
        args.name += "_debug"

    timestamp = misc.timestamp()
    args.unique_name = f"{timestamp}_{args.name}"

    # path setup
    args.work_dir = Path(".")
    args.data_dir = Path(args.data_dir)

    args.out_root = args.work_dir / Path("train_output") / args.dataset
    args.out_dir = args.out_root / args.unique_name
    args.out_dir.mkdir(exist_ok=True, parents=True)

    logger = Logger.get(args.out_dir / "log.txt")
    if args.debug:
        logger.setLevel("DEBUG")
    cmd = " ".join(sys.argv)
    logger.info(f"Command :: {cmd}")

    logger.nofmt("Environment:")
    logger.nofmt("\tPython: {}".format(sys.version.split(" ")[0]))
    logger.nofmt("\tPyTorch: {}".format(torch.__version__))
    logger.nofmt("\tTorchvision: {}".format(torchvision.__version__))
    logger.nofmt("\tCUDA: {}".format(torch.version.cuda))
    logger.nofmt("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.nofmt("\tNumPy: {}".format(np.__version__))
    logger.nofmt("\tPIL: {}".format(PIL.__version__))

    # Different to DomainBed, we support CUDA only.
    assert torch.cuda.is_available(), "CUDA is not available"

    logger.nofmt("Args:")
    for k, v in sorted(vars(args).items()):
        logger.nofmt("\t{}: {}".format(k, v))

    logger.nofmt("HParams:")
    for line in hparams.dumps().split("\n"):
        logger.nofmt("\t" + line)

    if args.show:
        exit()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    # Dummy datasets for logging information.
    # Real dataset will be re-assigned in train function.
    # test_envs only decide transforms; simply set to zero.
    dataset, _in_splits, _out_splits = get_dataset([0], args, hparams)

    # print dataset information
    logger.nofmt("Dataset:")
    logger.nofmt(f"\t[{args.dataset}] #envs={len(dataset)}, #classes={dataset.num_classes}")
    for i, env_property in enumerate(dataset.environments):
        logger.nofmt(f"\tenv{i}: {env_property} (#{len(dataset[i])})")
    logger.nofmt("")

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    logger.info(f"n_steps = {n_steps}")
    logger.info(f"checkpoint_freq = {checkpoint_freq}")

    org_n_steps = n_steps
    n_steps = (n_steps // checkpoint_freq) * checkpoint_freq + 1
    logger.info(f"n_steps is updated to {org_n_steps} => {n_steps} for checkpointing")

    if not args.test_envs:
        args.test_envs = [[te] for te in range(len(dataset))]
    logger.info(f"Target test envs = {args.test_envs}")

    ###########################################################################
    # Run
    ###########################################################################
    all_records = []
    results = collections.defaultdict(list)

    # Setup the distributed training
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # need to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size

    for test_env in args.test_envs:
        if args.multiprocessing_distributed:
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            res, records = train_on_node(ngpus_per_node=ngpus_per_node, 
                                         test_env=test_env,
                                         args=args,
                                         hparams=hparams,
                                         n_steps=n_steps,
                                         checkpoint_freq=checkpoint_freq,
                                         logger=logger)
        else:
            # Simply call main_worker function
            res, records = main_worker(args.gpu, ngpus_per_node, test_env, 
                                       args=args, 
                                       hparams=hparams, 
                                       n_steps=n_steps, 
                                       checkpoint_freq=checkpoint_freq, 
                                       logger=logger)

        all_records.append(records)
        for k, v in res.items():
            results[k].append(v)

    # log summary table
    logger.info("=== Summary ===")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info("Unique name: %s" % args.unique_name)
    logger.info("Out path: %s" % args.out_dir)
    logger.info("Algorithm: %s" % args.algorithm)
    logger.info("Dataset: %s" % args.dataset)

    table = PrettyTable(["Selection"] + dataset.environments + ["Avg."])
    for key, row in results.items():
        row.append(np.mean(row))
        row = [f"{acc:.3%}" for acc in row]
        table.add_row([key] + row)
    logger.nofmt(table)


def train_on_node(ngpus_per_node, test_env, args, hparams, n_steps, checkpoint_freq, logger):
    mannager = mp.Manager()
    return_queue = mannager.list()
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,
                                                       test_env,
                                                       args,
                                                       hparams,
                                                       n_steps,
                                                       checkpoint_freq,
                                                       logger,
                                                       return_queue),
                                                       join=True)
    for pack in return_queue:
        _res, _records = pack["ret"], pack["records"]
    res, records = _res, _records
    return res, records

if __name__ == "__main__":
    main()
