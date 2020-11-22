# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import math

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers,imbalance=False):
        super().__init__()

        if weights: # class balance weight (weight을 줘서 더 많이 sample하려는듯)
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        else:
            if imbalance:
                # 마지막 개수가 부족한 batch item의 경우 random over sampling으로 batch size 하나 만큼은 되도록 채운다.
                sampler = torch.utils.data.RandomSampler(dataset,
                                                         replacement=True,num_samples=math.ceil(len(dataset)/batch_size) * batch_size)
            else:
                sampler = torch.utils.data.RandomSampler(dataset,
                                                         replacement=True)  # replacement가 true가 되면 shuffle과는 달라진다. 뽑은거 또 뽑는거.

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)  # 이부분이 문제임. sampler의 length보다 batch가 큰경우 무한 루프에 빠진다.

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

class FastDataLoader: # just shuffle(not random over sampling in one batch.) & sampler의 length보다 batch가 큰경우 batch를 다뽑고 남은 나머지를 drop하지 않고 고대로 쓴다.
    # class 별 weight는 사용하지 않는다.
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length