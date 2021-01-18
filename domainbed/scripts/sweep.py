# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run sweeps
"""

import argparse
import copy
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import torch

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed import command_launchers

from itertools import product

import tqdm
import shlex

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = ['python', '-m', 'domainbed.scripts.train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')

def all_test_env_combinations(n):
    """For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs."""
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]

def make_args_list(n_trials, dataset_names, algorithms, n_hparams, steps,
    data_dir, hparams,checkpoint_freq):
    args_list = []
    for trial_seed in range(n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                # all_test_envs = all_test_env_combinations(
                #     datasets.num_environments(dataset))
                all_test_envs = [[x] for x in range(datasets.num_environments(dataset))] # for only one test envs
                for test_envs in all_test_envs:
                    for hparams_seed in range(n_hparams):
                        train_args = {}
                        train_args['dataset'] = dataset
                        train_args['algorithm'] = algorithm
                        train_args['test_envs'] = test_envs
                        train_args['hparams_seed'] = hparams_seed
                        train_args['data_dir'] = data_dir
                        train_args['trial_seed'] = trial_seed
                        train_args['seed'] = misc.seed_hash(dataset,
                            algorithm, test_envs, hparams_seed, trial_seed)
                        if steps is not None:
                            train_args['steps'] = steps
                        if checkpoint_freq is not None:
                            train_args['checkpoint_freq'] = checkpoint_freq
                        if hparams is not None:
                            train_args['hparams'] = hparams
                        args_list.append(train_args)
    return args_list

def make_args_list_forht(n_trials, dataset_names, algorithms, n_hparams, steps,
    data_dir, hparams,checkpoint_freq):

    #case1
    hlist1 = ['"1hid":100','"1hid":150','"1hid":50']
    hlist5 = ['"batch_size":128']
    hlist2 = ['"num_smallmetaset":6','"num_smallmetaset":12']
    hlist3 = ['"mod_lr":5e-5','"mod_lr":1e-5','"mod_lr":5e-6','"mod_lr":1e-6']
    hlist4 = ['"lr":1e-4']
    hparam_list = list(product(hlist1, hlist2,hlist3,hlist4,hlist5))

    # #case2
    # hlist1 = ['"mod_lr":1e-4','"mod_lr":1e-5','"mod_lr":5e-6']
    # hlist2 = ['"hidden_neurons":100','"hidden_neurons":200','"hidden_neurons":200']
    # hparam_list = list(product(hlist1, hlist2))

    # #case3
    # hlist1 = ['"1hid":100','"1hid":200','"1hid":300']
    # hlist2 = ['"clscond":"True"','"clscond":"False"']
    # hlist3 = ['"mod_lr":1e-2', '"mod_lr":1e-3', '"mod_lr":1e-4']
    # hparam_list = list(product(hlist1, hlist2,hlist3))


    print('parameter searching, total # --> ',len(hparam_list))

    args_list = []
    for trial_seed in range(n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                # all_test_envs = all_test_env_combinations(
                #     datasets.num_environments(dataset))
                all_test_envs = [[x] for x in range(datasets.num_environments(dataset))] # for only one test envs
                # all_test_envs = [3] # for hiper parameter search(test 1 dom)
                for test_envs in all_test_envs:
                    for hparams in hparam_list:
                        if type(hparams)==str:
                            hparams = '{'+hparams+'}'
                        else:
                            hparams = '{'+','.join(hparams)+'}'

                        hparams_seed = 0
                        train_args = {}
                        train_args['dataset'] = dataset
                        train_args['algorithm'] = algorithm
                        train_args['test_envs'] = test_envs
                        train_args['hparams_seed'] = hparams_seed
                        train_args['data_dir'] = data_dir
                        train_args['trial_seed'] = trial_seed
                        train_args['seed'] = misc.seed_hash(dataset,
                            algorithm, test_envs, hparams_seed, trial_seed)
                        if steps is not None:
                            train_args['steps'] = steps
                        if checkpoint_freq is not None:
                            train_args['checkpoint_freq'] = checkpoint_freq
                        train_args['hparams'] = hparams
                        args_list.append(train_args)
    return args_list

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--hypert', action='store_true')
    args = parser.parse_args()

    if not args.hypert:
        args_list = make_args_list(
            n_trials=args.n_trials,
            dataset_names=args.datasets,
            algorithms=args.algorithms,
            n_hparams=args.n_hparams,
            steps=args.steps,
            data_dir=args.data_dir,
            hparams=args.hparams,
            checkpoint_freq = args.checkpoint_freq
        )
    else:
        args_list = make_args_list_forht(
            n_trials=args.n_trials,
            dataset_names=args.datasets,
            algorithms=args.algorithms,
            n_hparams=args.n_hparams,
            steps=args.steps,
            data_dir=args.data_dir,
            hparams=args.hparams,
            checkpoint_freq=args.checkpoint_freq
        )

    jobs = [Job(train_args, args.output_dir) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)
