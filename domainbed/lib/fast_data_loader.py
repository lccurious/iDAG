# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler, manual_shuffle=False, batch_size=None, total_size=None):
        self.sampler = sampler
        self.manual_shuffle = manual_shuffle
        self._epoch = 0

    def __iter__(self):
        while True:
            if self.manual_shuffle:
                self.sampler.sampler.set_epoch(self._epoch)
                self._epoch += 1

            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers, args):
        super().__init__()

        if weights:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, replacement=True, num_samples=batch_size
            )
        elif args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=True
        )

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=_InfiniteSampler(batch_sampler, args.distributed, batch_size, len(dataset)),
            )
        )

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError


class FastDataLoader:
    """
    DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch.
    """

    def __init__(self, dataset, batch_size, num_workers, shuffle=False):
        super().__init__()

        if shuffle:
            # sampler = torch.utils.data.RandomSampler(dataset, replacement=False)
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        else:
            # sampler = torch.utils.data.SequentialSampler(dataset)
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=False,
        )

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=_InfiniteSampler(batch_sampler),
            )
        )

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length
