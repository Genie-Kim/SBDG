# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets, imbalance_dataset
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from domainbed.lib.imbalance import *
from itertools import product
from itertools import combinations


if __name__ == "__main__":
    # when starting
    # python -m domainbed.scripts.imbtrain_swing --data_dir=/home/genie/2hddb/dg_dataset --algorithm MLDG --dataset ImbalanceDomainNet --num_running_targets 2
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                             'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--num_combination', type=int, nargs='+', default=[1],
                        help='the list of num that how many the target domain selected.')

    args = parser.parse_args()
    # debugging 할때는 hparam registry에서 버전숫자 바꾸기, imbrate 극과 극으로만 바꾸기, imbalance_dataset에서 niter 바꾸기.
    # imbrates = [1, 16]  # debug
    # imbtrain(args, [1, 5], 1, 2, 'domain')  # debug
    ORIGINDATA = 'DomainNet'
    clsordom = 'domain'
    imbrates = [1,2,4,8,16,32,64]
    # check hparams_registry and delete the fixed target domain number.
    # domains = [0,1,2,3,4,5] # 여기서 fixed target domain은 뺀다.
    domains = [1,2,3,4,5]

    device = "cuda"
    idx2domain = list.copy(vars(imbalance_dataset)[args.dataset].ENVIRONMENTS)
    domain2idx = {k:v for v,k in enumerate(idx2domain)}
    columns = ['alg', 'source', 'fixtarget', 'runtarget', 'minor', 'imbrate', 'dom', 'trteval', 'cls', 'acc']
    acc_result_df = pd.DataFrame(columns=columns)
    for num_runtargets in args.num_combination:
        for running_target_tuple in tqdm(list(combinations(domains, num_runtargets))):
            running_targets = [x for x in running_target_tuple]
            balance_setting_check = False  # balance setting을 했는가.
            for minor_domain in domains:
                if not minor_domain in running_targets:
                    for imbrate in imbrates:
                        if imbrate == 1 & (not balance_setting_check):
                            balance_setting_check = True
                        elif imbrate == 1 & balance_setting_check:
                            continue

                        if args.hparams_seed == 0:  # default hyper parameter
                            hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
                        else:
                            hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                                      misc.seed_hash(args.hparams_seed, args.trial_seed))

                        if args.dataset in hparams_registry.IMBALANCE:
                            hparams['clsordom'] = clsordom  # make domain imbalance
                            hparams['imbrate'] = imbrate  # The degree of imbalance, expressed as a major/minor value.
                            hparams['minor_domain'] = minor_domain

                        if args.hparams:
                            hparams.update(json.loads(args.hparams))

                        params = (
                            hparams['imb_data_root'], hparams['dataset_version'], ORIGINDATA,
                            hparams['numcls'],
                            hparams['testrate'],
                            hparams['valrate'],
                            hparams['targets_fix'])

                        # get source domains and target domains and minor domain
                        # imbalance dataset의 원 environment를 참조.
                        fixtargets = [idx2domain[x] for x in hparams['targets_fix']]
                        runtargets = [idx2domain[x] for x in running_targets]
                        minor_name = idx2domain[hparams['minor_domain']]
                        target_domains = fixtargets + runtargets
                        source_domains = []
                        for dom_name in idx2domain:
                            if dom_name not in target_domains:
                                source_domains.append(dom_name)
                        target_domains.sort()
                        fixtargets.sort()
                        runtargets.sort()

                        imb_csvpath = get_imb_csvpath_bysetting(params, running_targets, 'imbtrain',
                                                                hparams['minor_domain'],
                                                                hparams['imbrate'], hparams['clsordom'])
                        imb_path = os.path.splitext(imb_csvpath)[0]

                        # load validation result json & get best validation model's result
                        with open(os.path.join(imb_path, 'results.jsonl'),'r') as json_file:
                            jsonl_str = json_file.read()
                            val_result = [json.loads(jline) for jline in jsonl_str.splitlines()]
                            val_acc_list = []
                            for item in val_result:
                                val_acc = 0
                                for dom_name in source_domains:
                                    val_acc += item['_'.join(['env',dom_name,'out','acc'])]
                                val_acc_list.append(val_acc/len(source_domains))
                            best_val_idx = val_acc_list.index(max(val_acc_list))
                            best_val_result = val_result[best_val_idx]


                        same_list = [hparams['dataset_version'], '_'.join(source_domains), '_'.join(fixtargets),'_'.join(runtargets),
                                                         idx2domain[hparams['minor_domain']], hparams['imbrate']]
                        weit_source_tr_acc = 0
                        mean_source_val_acc = 0
                        mean_target_val_acc = 0


                        for k,v in best_val_result.items():
                            keylist = k.split('_')

                            if keylist[0] == 'env':
                                if keylist[2] == 'out':
                                    tetrval = 'val'
                                    if keylist[-1] == 'acc': # validation accuracy
                                        cls = '@total'
                                        if keylist[1] in target_domains:
                                            mean_target_val_acc += v
                                        else:
                                            mean_source_val_acc += v
                                    else:
                                        cls = keylist[4]

                                elif keylist[2] == 'in':
                                    tetrval = 'train'
                                    if keylist[-1] == 'acc': # train accuracy
                                        cls = '@total'
                                        if keylist[1] == minor_name:
                                            # weighting for imbrate of minor domain
                                            weit_source_tr_acc += v/(hparams['imbrate']*(len(source_domains)-1)+1)
                                        else:
                                            weit_source_tr_acc += v*hparams['imbrate'] / (
                                                        hparams['imbrate'] * (len(source_domains) - 1) + 1)
                                    else:
                                        cls = keylist[4]

                                temp_df = pd.DataFrame([same_list+[keylist[1], tetrval, cls, v]],
                                                       columns=columns)

                                acc_result_df = pd.concat([acc_result_df, temp_df], ignore_index=True)

                        mean_source_val_acc = mean_source_val_acc/len(source_domains)
                        mean_target_val_acc = mean_target_val_acc/len(target_domains)

                        temp_df = pd.DataFrame([same_list + ['@source', 'train', '@mean', weit_source_tr_acc],
                                                same_list + ['@source', 'val', '@mean', mean_source_val_acc],
                                                same_list + ['@target', 'val', '@mean', mean_target_val_acc]],
                                               columns=columns)
                        acc_result_df = pd.concat([acc_result_df, temp_df], ignore_index=True)
    out_folder_name = get_imb_foldername_bysetting(hparams['dataset_version'], ORIGINDATA, hparams['numcls'], hparams['testrate'],
                                 hparams['valrate'], hparams['targets_fix'])
    acc_result_df.to_csv(os.path.join(hparams['imb_data_root'],out_folder_name,'train_val_result_summary.csv'),index=False)

