# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import sys
from shutil import copyfile

import numpy as np
import torch
import tqdm
from collections import Counter

def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def split_smallmetaset(base_set, input_set, num_per_cls):
    meta_keys = []
    n_classes = len(base_set.classes)

    cls_keys = {c:[]for c in range(n_classes)}
    for x in tqdm.tqdm(input_set.keys):
        cls = base_set[x][1]
        if len(cls_keys[cls])>=num_per_cls:
            continue
        else:
            cls_keys[cls].append(x)
    for _,v in cls_keys.items():
        meta_keys += v

    remain_keys = [x for x in input_set.keys if x not in meta_keys]
    return _SplitDataset(base_set,meta_keys), _SplitDataset(base_set,remain_keys)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def random_pairs_of_minibatches_outputdom(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n],perm[i]), (xj[:min_n], yj[:min_n],perm[j])))

    return pairs


def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.cuda()
            if p.size(1) == 1: # for binary class setting
                # p.gt(0)은 prediction score가 0보다 큰 것을 true, 아니면 false로 나타냄.
                correct += (p.gt(0).eq(y).float() * batch_weights).sum().item()
            else: # for multi class setting
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total


def accuracy_percls(network, loader, weights, device,num_cls):
    correct_percls = np.zeros(num_cls).tolist()
    total_percls = np.zeros(num_cls).tolist()
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.cuda()
            if p.size(1) == 1:
                # p.gt(0)은 prediction score가 0보다 큰 것을 true, 아니면 false로 나타냄.
                correct += (p.gt(0).eq(y).float() * batch_weights).sum().item()
            else:
                correct_temp,total_temp = multi_acc(p,y,num_cls)
                correct_percls = [correct_percls[x] + correct_temp[x] for x in range(num_cls)]
                total_percls = [total_percls[x] + total_temp[x] for x in range(num_cls)]
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    acc_percls = [correct_percls[x] / total_percls[x] for x in range(num_cls)]
    total_acc = correct / total
    acc_percls.append(total_acc) # last element of acc_percls is total accuracy considering the weight.
    network.train()

    return acc_percls # [class1 accuacy,class2 accuacy,class3 accuacy,,, total accuracy]

def multi_acc(pred, label,num_cls):
    num_of_corrected_percls = []
    num_of_total_percls = []
    tags = torch.argmax(pred, dim=1)
    for c in range(num_cls):
        of_c = label == c
        num_total_per_label = of_c.sum() # batch 안에 class =c 인 개수.
        of_c &= tags == label # of_c가 그중에서 몇개 맞췄냐로 바뀐다.
        num_corrects_per_label = of_c.sum()
        num_of_corrected_percls.append(num_corrects_per_label.item())
        num_of_total_percls.append(num_total_per_label.item())
    return (num_of_corrected_percls,num_of_total_percls)



class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
