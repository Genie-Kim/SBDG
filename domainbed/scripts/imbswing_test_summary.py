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
    parser.add_argument('--num_combination', type=int, nargs='+', default=[1],help='the list of num that how many the target domain selected.')


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
    acc_result_df = pd.DataFrame(
        columns=columns)
    for num_runtargets in args.num_combination:
        for running_target_tuple in tqdm(list(combinations(domains,num_runtargets))):
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
                        # load best source mean validation model.
                        best_model_dict = torch.load(os.path.join(imb_path, 'best_val_model.pkl'))
                        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
                        algorithm = algorithm_class(best_model_dict['model_input_shape'],
                                                    best_model_dict['model_num_classes'],
                                                    best_model_dict['model_num_domains'], hparams)
                        algorithm.load_state_dict(best_model_dict['model_dict'])
                        algorithm.to(device)

                        test_dataset = vars(imbalance_dataset)[args.dataset](args.data_dir, 'test', running_targets, hparams)
                        # for making validation set
                        out_splits = []  # validation dataset instances of each domains spliting by args.holdout_fraction
                        for env_test_i, env_test in enumerate(test_dataset):
                            if hparams['class_balanced']:
                                out_weights = misc.make_weights_for_balanced_classes(env_test)
                            else:
                                in_weights, out_weights = None, None
                            out_splits.append((env_test, out_weights))

                        # eval_loaders는 domain 이 6개면 12개의 dataset loader가 들어간다. target domain 안빠짐.
                        eval_loaders = [FastDataLoader(
                            dataset=env,
                            batch_size=64,
                            num_workers=test_dataset.N_WORKERS)
                            for env, _ in out_splits]  # 한 도메인의 support set과 query set을 다 eval loader로 넣음
                        eval_weights = [None for _, weights in out_splits]
                        eval_loader_names = ['env_' + dom_name + '_out'
                                              for dom_name in idx2domain]

                        eval_acc_df = pd.DataFrame(columns=['dom', 'trteval', 'cls', 'acc'])
                        evals = zip(eval_loader_names, eval_loaders, eval_weights)
                        for name, loader, weights in evals:
                            # acc = misc.accuracy(algorithm, loader, weights, device)
                            acc = misc.accuracy_percls(algorithm, loader, weights, device, hparams['numcls'])

                            dom = name.split('_')[1]
                            total_acc = acc.pop(-1)

                            temp_df = pd.DataFrame([[dom, 'test', '@total', total_acc]],
                                                   columns=['dom', 'trteval', 'cls', 'acc'])
                            eval_acc_df = pd.concat([eval_acc_df, temp_df], ignore_index=True)

                            for cls, idx in loader._infinite_iterator._dataset.class_to_idx.items():
                                temp_df = pd.DataFrame([[dom, 'test', cls, acc[idx]]],
                                                       columns=['dom', 'trteval', 'cls', 'acc'])
                                eval_acc_df = pd.concat([eval_acc_df, temp_df], ignore_index=True)

                        test_pivot_df = pd.read_csv(test_dataset.test_csv_path).pivot_table(index='cls', columns='dom', aggfunc='count')
                        imgnum_testsets = {k[1]: v for k, v in test_pivot_df.sum(axis=0).to_dict().items()}

                        weit_target_te_acc = 0
                        weit_source_te_acc = 0

                        imgnum_source = 0
                        imgnum_target = 0
                        for dom in idx2domain:
                            if dom in source_domains:
                                imgnum_source += imgnum_testsets[dom]
                            elif dom in target_domains:
                                imgnum_target += imgnum_testsets[dom]

                        for dom in idx2domain:
                            if dom in source_domains:
                                ratio = imgnum_testsets[dom]/imgnum_source
                                weit_source_te_acc += ratio * float(eval_acc_df[(eval_acc_df['dom'] == dom) & (eval_acc_df['cls'] == '@total')]['acc'])
                            elif dom in target_domains:
                                ratio = imgnum_testsets[dom] / imgnum_target
                                weit_target_te_acc += ratio * float(eval_acc_df[(eval_acc_df['dom'] == dom) & (eval_acc_df['cls'] == '@total')]['acc'])

                        temp_df = pd.DataFrame([['@source', 'test', '@mean', weit_source_te_acc],['@target', 'test', '@mean', weit_target_te_acc]],
                                               columns=['dom', 'trteval', 'cls', 'acc'])
                        eval_acc_df = pd.concat([eval_acc_df, temp_df], ignore_index=True)

                        # merge eval_acc_df to acc_result_df
                        same_list = [hparams['dataset_version'], '_'.join(source_domains), '_'.join(fixtargets),'_'.join(runtargets),
                                                         idx2domain[hparams['minor_domain']], hparams['imbrate']]
                        temp = [same_list for i in range(len(eval_acc_df))]
                        temp_df = pd.DataFrame(temp, columns=columns[:6])
                        temp_df = pd.concat([temp_df, eval_acc_df], axis=1)
                        acc_result_df = pd.concat([acc_result_df, temp_df], ignore_index=True)

    out_folder_name = get_imb_foldername_bysetting(hparams['dataset_version'], ORIGINDATA, hparams['numcls'], hparams['testrate'],
                                 hparams['valrate'], hparams['targets_fix'])
    acc_result_df.to_csv(os.path.join(hparams['imb_data_root'],out_folder_name,'test_result_summary.csv'),index=False)



