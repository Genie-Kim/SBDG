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


def imbtrain(args, running_targets, imbrate, minor_domain, clsordom):

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

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(imbalance_dataset):  # for imbalance dataset class.
        # 해당 데이터셋에 해당하는 클래스 instance를 반환. (root, target domain, hyper parameter)
        train_dataset = vars(imbalance_dataset)[args.dataset](args.data_dir, 'train', running_targets, hparams)
        val_dataset = vars(imbalance_dataset)[args.dataset](args.data_dir, 'val', running_targets, hparams)
    else:
        raise NotImplementedError

    params = (
        hparams['imb_data_root'], hparams['dataset_version'], train_dataset.ORIGINDATA, hparams['numcls'],
        hparams['testrate'],
        hparams['valrate'],
        hparams['targets_fix'])

    imb_csvpath = get_imb_csvpath_bysetting(params, running_targets, 'imbtrain', hparams['minor_domain'],
                                            hparams['imbrate'], hparams['clsordom'])
    output_dir = os.path.splitext(imb_csvpath)[0]
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(output_dir, 'err.txt'))

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

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    in_splits = []  # train dataset instances of each domains
    out_splits = []  # validation dataset instances of each domains spliting by args.holdout_fraction

    # for making validation set
    for env_val_i, env_val in enumerate(val_dataset):
        if hparams['class_balanced']:
            out_weights = misc.make_weights_for_balanced_classes(env_val)
        else:
            in_weights, out_weights = None, None
        out_splits.append((env_val, out_weights))

    for env_i, env in enumerate(train_dataset):
        if len(env) > 0:
            if hparams['class_balanced']:
                in_weights = misc.make_weights_for_balanced_classes(env)
            else:
                in_weights, out_weights = None, None
            in_splits.append((env, in_weights))
    # when imbalance ==True then use random over sampling for minor domain.
    # math.ceil(len(dataset)/batch_size) * batch_size 만큼 dataset에서 sampling한다.
    # domain generalization이기 때문에 random oversampling 사용.
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=train_dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)]


    # fast loader는 좀더 빨라졌다 뿐, infinite data loader와 비슷하다.
    # eval_loaders는 domain 이 6개면 12개의 dataset loader가 들어간다. target domain 안빠짐.
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=val_dataset.N_WORKERS)
        for env, _ in in_splits + out_splits]  # 한 도메인의 support set과 query set을 다 eval loader로 넣음
    # TODO : 만약 source domain의 크기가 크다면 eval loader의 경우 out split만 해야한다. validation set만 따져야 한다.

    eval_weights = [None for _, weights in (in_splits + out_splits)]

    # imbalance dataset의 원 environment를 참조.
    domain_list = list.copy(vars(imbalance_dataset)[args.dataset].ENVIRONMENTS)
    target_domains = []
    source_domains = train_dataset.ENVIRONMENTS
    for dom_name in domain_list:
        if dom_name not in source_domains:
            target_domains.append(dom_name)

    eval_loader_names = ['env_' + dom_name + '_in'
                         for dom_name in source_domains]
    eval_loader_names += ['env_' + dom_name + '_out'
                          for dom_name in domain_list] # TODO : 만약 source domain의 크기가 크다면 eval loader의 경우 out split만 해야한다. validation set만 따져야 한다.

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(train_dataset.input_shape, train_dataset.num_classes,
                                len(train_loaders), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)  # source domain의 loader를 묶는다.
    checkpoint_vals = collections.defaultdict(lambda: [])  # time과 같은 부수적인 결과물 dict로 저장.

    steps_per_epoch = min([len(env) / hparams['batch_size'] for env, _ in
                           in_splits])  # domain중에서 가장 작은 도메인 / batch size의 올림을 epoch의 itration 개수로 잡음.

    n_steps = args.steps or train_dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or train_dataset.CHECKPOINT_FREQ

    last_results_keys = None

    writer = SummaryWriter(os.path.join(output_dir, 'tb'))

    writer.add_text('domain summary', 'Target domains : ' + ', '.join(target_domains), 0)
    writer.add_text('domain summary', 'Source domains : ' + ', '.join(source_domains), 1)
    writer.add_text('domain summary', 'minor domain : ' + domain_list[hparams['minor_domain']], 2)

    best_mean_acc = 0  # source domain's validation's accuracy's best mean.
    for step in tqdm(range(start_step, n_steps)):  # iteration, epoch개념이 따로 없고 데이터셋 숫자로 계산하는 개념이 됨.
        step_start_time = time.time()
        # 각 source도메인 마다 (이미지 x batch_size,라벨 x batch_size) 튜플을 생성하여 source 도메인 개수만큼 만들고 그걸 리스트로 만든다.
        # 배치 개수는 hparams_registry.py 19째 줄에 기록됨.
        minibatches_device = [(x.to(device), y.to(device))
                              for x, y in next(train_minibatches_iterator)]
        step_vals = algorithm.update(
            minibatches_device)  # forward the algorithm 알고리즘 네트워크 업데이트 부분. step_vals에는 {'loss': 11.7731}같은 것들이 나옴.
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        writer.add_scalar('train loss', step_vals['loss'], step)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if step % checkpoint_freq == 0:  # validation step
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)  # result에는 한번 iteration하는데 걸리는 시간이라던가 loss값의 mean이 들어간다.

            eval_acc_df = pd.DataFrame(columns=['dom', 'inorout', 'cls', 'acc'])
            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                # acc = misc.accuracy(algorithm, loader, weights, device)
                acc = misc.accuracy_percls(algorithm, loader, weights, device, hparams['numcls'])

                inorout = name.split('_')[-1]
                dom = name.split('_')[1]
                total_acc = acc.pop(-1)
                results[name + '_acc'] = total_acc
                temp_df = pd.DataFrame([[dom, inorout, 'total@', total_acc]], columns=['dom', 'inorout', 'cls', 'acc'])
                eval_acc_df = pd.concat([eval_acc_df, temp_df], ignore_index=True)

                for cls, idx in loader._infinite_iterator._dataset.class_to_idx.items():
                    temp_df = pd.DataFrame([[dom, inorout, cls, acc[idx]]],
                                           columns=['dom', 'inorout', 'cls', 'acc'])
                    eval_acc_df = pd.concat([eval_acc_df, temp_df], ignore_index=True)
                    results[name + '_acc_' + cls] = acc[idx]

            # for surmmary writer
            for cls in eval_acc_df.groupby('cls').count().index:
                acc_percls_in = eval_acc_df[(eval_acc_df['cls'] == cls) & (eval_acc_df['inorout'] == 'in')].copy()
                acc_percls_out = eval_acc_df[(eval_acc_df['cls'] == cls) & (eval_acc_df['inorout'] == 'out')].copy()
                acc_percls_in = acc_percls_in.drop(['inorout', 'cls'], axis=1)
                acc_percls_out = acc_percls_out.drop(['inorout', 'cls'], axis=1)

                writer.add_scalars('train accuracy of /' + cls + ' per domain',
                                   acc_percls_in.set_index('dom').to_dict()['acc'], step)
                writer.add_scalars('val accuracy of /' + cls + ' per domain',
                                   acc_percls_out.set_index('dom').to_dict()['acc'], step)

            # for calculating best source validation model
            source_val_acc = 0
            target_val_acc = 0
            for dom in source_domains:
                source_val_acc += float(eval_acc_df[
                                            ((eval_acc_df['dom'] == dom) & (eval_acc_df['inorout'] == 'out')) & (
                                                        eval_acc_df['cls'] == 'total@')]['acc'])

            for dom in target_domains:
                target_val_acc += float(eval_acc_df[
                                            ((eval_acc_df['dom'] == dom) & (eval_acc_df['inorout'] == 'out')) & (
                                                    eval_acc_df['cls'] == 'total@')]['acc'])

            source_val_acc = source_val_acc / len(source_domains)
            target_val_acc = target_val_acc / len(target_domains)
            writer.add_scalars('val mean accuracy', {'source': source_val_acc, 'target': target_val_acc}, step)

            if best_mean_acc < source_val_acc:  # if best mean source validation score
                print('best model saving....')
                save_dict = {
                    "args": vars(args),
                    "model_input_shape": train_dataset.input_shape,
                    "model_num_classes": train_dataset.num_classes,
                    "model_num_domains": len(train_loaders),
                    "model_hparams": hparams,
                    "model_dict": algorithm.cpu().state_dict()
                }
                torch.save(save_dict, os.path.join(output_dir, "best_val_model.pkl"))
                best_mean_acc = source_val_acc
                algorithm.to(device)

            # printing result to stdout
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                # print('\n')
                # misc.print_row(results_keys, colwidth=20)
                last_results_keys = results_keys
            # misc.print_row([results[key] for key in results_keys],colwidth=20)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

    if not args.skip_model_save:
        save_dict = {
            "args": vars(args),
            "model_input_shape": train_dataset.input_shape,
            "model_num_classes": train_dataset.num_classes,
            "model_num_domains": len(train_loaders),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }

        torch.save(save_dict, os.path.join(output_dir, "model.pkl"))

    with open(os.path.join(output_dir, 'done'), 'w') as f:
        f.write('done')

    writer.close()



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
    parser.add_argument('--num_running_targets', type=int, default=None,
                        help='how many the target domain selected.')

    args = parser.parse_args()
    # debugging 할때는 hparam registry에서 버전숫자 바꾸기, imbrate 극과 극으로만 바꾸기, imbalance_dataset에서 niter 바꾸기.
    # imbrates = [1, 16]  # debug
    # imbtrain(args, [1, 5], 1, 2, 'domain')  # debug

    clsordom = 'domain'
    imbrates = [32,64]
    # check hparams_registry and delete the fixed target domain number.
    # domains = [0,1,2,3,4,5] # 여기서 fixed target domain은 뺀다.
    domains = [1,2,3,4,5]

    for running_target_tuple in list(combinations(domains, args.num_running_targets)):
        running_targets = [x for x in running_target_tuple]
        balance_setting_check = False  # balance setting을 했는가.
        for minor_domain in domains:
            if not minor_domain in running_targets:
                for imbrate in imbrates:
                    if imbrate == 1 & (not balance_setting_check):
                        balance_setting_check = True
                    elif imbrate == 1 & balance_setting_check:
                        continue
                    print('imbrate, minor_domain, running_targets', imbrate, minor_domain, running_targets)

#    yorn = input('\n swing by this settings?(y or n)-->')
    yorn = 'y'

    if yorn=='y':
        for running_target_tuple in list(combinations(domains,args.num_running_targets)):
            running_targets = [x for x in running_target_tuple]
            balance_setting_check = False # balance setting을 했는가.
            for minor_domain in domains:
                if not minor_domain in running_targets:
                    for imbrate in imbrates:
                        if imbrate ==1 & (not balance_setting_check):
                            balance_setting_check = True
                        elif imbrate ==1 & balance_setting_check:
                            continue
                        print ('\nimbrate : ',imbrate)
                        print('minor_domain : ',minor_domain)
                        print('running_targets : ', running_targets)
                        imbtrain(args, running_targets, imbrate, minor_domain, clsordom)

