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

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

import re
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pandas as pd
import copy


if __name__ == "__main__":
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
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--skip_model_save', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None
    # algorithm_dict = torch.load('/home/genie/PycharmProjects/DomainBed/performance/vlcs_mldg2/vlcs2/44eec3c2a184d18e7f36d4816576c454/model.pkl')

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        if v == 'True': # for hyper parameter search on sweep.py
            hparams[k] = True
        elif v=='False':
            hparams[k] = False
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    source_names = [x for i, x in enumerate(dataset.ENVIRONMENTS) if i not in args.test_envs]
    target_names = [x for i, x in enumerate(dataset.ENVIRONMENTS) if i in args.test_envs]
    class_names = list(dataset.datasets[0].classes)

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    in_splits = []
    out_splits = []
    meta_splits = []
    best_target_val_mean = 0
    best_source_val_mean = 0
    for env_i, env in enumerate(dataset):
        if env_i in args.test_envs:
            out, in_ = misc.split_dataset(env,
                int(len(env)),
                misc.seed_hash(args.trial_seed, env_i))
            in_ = copy.deepcopy(out)
        else:
            out, in_ = misc.split_dataset(env,
                int(len(env)*args.holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if 'num_smallmetaset' in hparams:
            # in_ 에서 class별 hparams['num_smallmetaset']개수만큼 뽑아서 balanced small meta set만든다.
            print('make small meta set:',env_i)
            smallmetaset_perdom, in_ = misc.split_smallmetaset(env,in_,hparams['num_smallmetaset'])
            meta_splits.append(smallmetaset_perdom)

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]
    if 'num_smallmetaset' in hparams:
        small_meta_loader = [InfiniteDataLoader(
            dataset=env,
            weights=None,
            batch_size=hparams['small_batch'],
            num_workers=2)
            for i, env in enumerate(meta_splits)
            if i not in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict['model_dict'])

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    last_results_keys = None


    if args.hparams:
        hyper_tuning = json.loads(args.hparams)
        tb_out = ''.join(['('+str(k)+'_'+str(v)for k,v in hyper_tuning.items()])
        writer = SummaryWriter(os.path.join(args.output_dir, tb_out))
    else:
        writer = SummaryWriter(os.path.join(args.output_dir, 'tb'))


    if 'inforecord' in hparams:
        loss_table_dom_cls = []
        info_chart = torch.zeros(len(source_names),dataset.num_classes,hparams['inforecord']).cuda().requires_grad_(False)
        total_num = torch.zeros(len(source_names),dataset.num_classes,hparams['inforecord']).cuda().requires_grad_(False)

    for step in tqdm(range(start_step, n_steps)):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        if 'num_smallmetaset' in hparams:
            step_vals = algorithm.update(minibatches_device,small_meta_loader)
        else:
            step_vals = algorithm.update(minibatches_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            if key == 'chart':
                info_chart+=val[0]
                total_num+=val[1]
                continue
            if key == 'sche':
                val.step()
                continue
            checkpoint_vals[key].append(val)
            if key == 'loss':
                writer.add_scalar('train loss', step_vals['loss'], step)


        if step % checkpoint_freq == 0:
            if 'inforecord' in hparams:
                info_dict = {k:{} for k in range(hparams['inforecord'])}
                total_num[total_num==0]=1
                info_chart=info_chart/total_num
                for d in range(algorithm.num_domains):
                    for c in range(algorithm.num_classes):
                        loss_table_dom_cls.append([source_names[d], class_names[c]] + info_chart[d, c, :].tolist() + [step])
                        for k in info_dict.keys():
                            info_dict[k][source_names[d]] = torch.mean(info_chart[d, :, k]).tolist()
                if args.algorithm in ['CMWN_MLDG','MWN_MLDG','MFM_MLDG']:
                    writer.add_scalars('inner loss info per domain', info_dict[0], step)
                    writer.add_scalars('outer loss info per domain', info_dict[1], step)
                    writer.add_scalars('meta loss info per domain', info_dict[2], step)
                    writer.add_scalars('weight info per domain', info_dict[3], step)
                    writer.add_scalars('accuracy info per domain', info_dict[4], step)
                elif args.algorithm in ['MLDG']:
                    writer.add_scalars('inner loss info per domain', info_dict[0], step)
                    writer.add_scalars('outer loss info per domain', info_dict[1], step)
                    writer.add_scalars('accuracy info per domain', info_dict[2], step)
                elif args.algorithm in ['CMWN_RSC']:
                    writer.add_scalars('loss info per domain', info_dict[0], step)
                    writer.add_scalars('meta loss info per domain', info_dict[1], step)
                    writer.add_scalars('weight info per domain', info_dict[2], step)
                    writer.add_scalars('accuracy info per domain', info_dict[3], step)
                elif args.algorithm in ['RSC']:
                    writer.add_scalars('loss info per domain', info_dict[0], step)
                    writer.add_scalars('accuracy info per domain', info_dict[1], step)
                info_chart.zero_()
                total_num.zero_()

            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                if key =='chart':
                    continue
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc

            source_train_accs = {}
            source_val_accs = {}
            target_train_accs = {}
            target_val_accs = {}

            p = re.compile('env(?P<env>\d+)_(?P<inorout>\w+)_acc')
            for key,val in results.items():
                temp = p.search(key)
                if temp is not None:
                    env_i = int(temp.group('env'))
                    inorout = temp.group('inorout')
                    if env_i in args.test_envs: # target in
                        if inorout == 'in':
                            target_train_accs[dataset.ENVIRONMENTS[env_i]]=val
                        else:
                            target_val_accs[dataset.ENVIRONMENTS[env_i]]=val

                    else: # sources
                        if inorout == 'in':
                            source_train_accs[dataset.ENVIRONMENTS[env_i]]=val
                        else:
                            source_val_accs[dataset.ENVIRONMENTS[env_i]]=val

            writer.add_scalars('source train accuracy of',source_train_accs, step)
            writer.add_scalars('source val accuracy of', source_val_accs, step)
            writer.add_scalars('target train accuracy of', target_train_accs, step)
            writer.add_scalars('target val accuracy of', target_val_accs, step)


            source_val_mean = np.array([v for k,v in source_val_accs.items()]).mean()
            target_val_mean = np.array([v for k,v in target_val_accs.items()]).mean()

            writer.add_scalar('source val mean', source_val_mean, step)
            writer.add_scalar('target val mean',target_val_mean,step)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            torch.cuda.empty_cache()

            if best_source_val_mean < source_val_mean:
                save_dict = {
                    "args": vars(args),
                    "model_input_shape": dataset.input_shape,
                    "model_num_classes": dataset.num_classes,
                    "model_num_domains": len(dataset) - len(args.test_envs),
                    "model_hparams": hparams,
                    "model_dict": algorithm.cpu().state_dict()
                }
                algorithm.cuda()
                torch.save(save_dict, os.path.join(args.output_dir, "best_val_model.pkl"))
                best_source_val_mean = source_val_mean

            if best_target_val_mean < target_val_mean:
                save_dict = {
                    "args": vars(args),
                    "model_input_shape": dataset.input_shape,
                    "model_num_classes": dataset.num_classes,
                    "model_num_domains": len(dataset) - len(args.test_envs),
                    "model_hparams": hparams,
                    "model_dict": algorithm.cpu().state_dict()
                }
                algorithm.cuda()
                torch.save(save_dict, os.path.join(args.output_dir, "best_target_model.pkl"))
                best_target_val_mean = target_val_mean

    # if not args.skip_model_save:
    #     save_dict = {
    #         "args": vars(args),
    #         "model_input_shape": dataset.input_shape,
    #         "model_num_classes": dataset.num_classes,
    #         "model_num_domains": len(dataset) - len(args.test_envs),
    #         "model_hparams": hparams,
    #         "model_dict": algorithm.cpu().state_dict()
    #     }
    #
    #     torch.save(save_dict, os.path.join(args.output_dir, "model.pkl"))

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done best target :'+str(best_target_val_mean)+', best source val :'+str(best_source_val_mean))


    if 'inforecord' in hparams:
        if args.algorithm in ['CMWN_MLDG', 'MWN_MLDG', 'MFM_MLDG']:
            df = pd.DataFrame(data=loss_table_dom_cls,
                              columns=['domain', 'class', 'innerloss', 'outerloss', 'metaloss', 'weight', 'accuracy',
                                       'step'])
            df.to_csv(os.path.join(args.output_dir, 'lossinfo_per_domcls.csv'))
        elif args.algorithm in ['MLDG']:
            df = pd.DataFrame(data=loss_table_dom_cls,
                              columns=['domain', 'class', 'innerloss', 'outerloss', 'accuracy', 'step'])
            df.to_csv(os.path.join(args.output_dir, 'lossinfo_per_domcls.csv'))
        elif args.algorithm in ['CMWN_RSC']:
            df = pd.DataFrame(data=loss_table_dom_cls,
                              columns=['domain', 'class', 'loss', 'metaloss', 'weight', 'accuracy','step'])
            df.to_csv(os.path.join(args.output_dir, 'lossinfo_per_domcls.csv'))
        elif args.algorithm in ['RSC']:
            df = pd.DataFrame(data=loss_table_dom_cls,
                              columns=['domain', 'class', 'loss', 'accuracy', 'step'])
            df.to_csv(os.path.join(args.output_dir, 'lossinfo_per_domcls.csv'))

    writer.close()