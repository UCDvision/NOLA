# Load accuracy values from logs of multiple runs and calculate mean and std values

import os
from os.path import join
import pdb

import numpy as np

from utils.utils import get_acc_from_file

# File structure: exp_dir/kshot/seed/run
base_dir = './exp'
arch = 'large'
tt = 'nola_mlp'
dset = 'CIFAR10'
rank = 1
ka = 1024
lr = '5e-3'

exp = 'e001_%s_ft_%s_%s_r%s_k%s_lr%s' % (arch, tt, dset, rank, ka, lr)

# print(exp)
exp_dir = join(base_dir, exp)
# kshot_dirs = sorted(os.listdir(exp_dir))
kshot_dirs = ['5shot', '10shot']
acc_kshots = []
mean_kshots = []
std_kshots = []
all_std_kshots = []
for kshot_dir in kshot_dirs:
    kshot = int(kshot_dir.split('shot')[0])
    # print('kshot: ', kshot)
    if not os.path.exists(join(exp_dir, kshot_dir)):
        print('Path does not exist: ', join(exp_dir, kshot_dir))
        continue
    seeds = sorted(os.listdir(join(exp_dir, kshot_dir)))
    acc_seeds = []
    std_seeds = []
    # seeds = ['v0', 'v1', 'v2', 'v3']
    for seed in seeds:
        # print('seed: ', seed)
        runs = sorted(os.listdir(join(exp_dir, kshot_dir, seed)))
        # runs = ['run0', 'run1', 'run2']
        acc_runs = []
        for run in runs:
            # print('run: ', run)
            try:
                logfiles = sorted(os.listdir(join(exp_dir, kshot_dir, seed, run)))
            except:
                print('Log file not found!', join(exp_dir, kshot_dir, seed, run))
                continue
            # Make sure the latest log file is the one needed to calculate metrics!!!
            logfiles = [item for item in logfiles if item.endswith('.log')]
            if logfiles is not None:
                logfile = logfiles[-1]
            else:
                print('Log file not found: ', join(exp_dir, kshot_dir, seed, run))
                continue
            try:
                acc = get_acc_from_file(join(exp_dir, kshot_dir, seed, run, logfile))
            except:
                print('Unable to get acc: ', join(exp_dir, kshot_dir, seed, run, logfile))
            # print(acc)
            if acc >= 0:
                acc_runs.append(acc)
            else:
                print(join(exp_dir, kshot_dir, seed, run))
                try:
                    acc = get_acc_from_file(join(exp_dir, kshot_dir, seed, run, logfiles[-2]))
                    if acc >= 0:
                        acc_runs.append(acc)
                        print('success')
                    else:
                        print('fail 1')
                except:
                    print('fail 2')
                    continue
                # pass
        std = np.std(acc_runs)
        std_seeds.append(std)
        acc_seeds.append(acc_runs)
    acc_seeds = [ii for item in acc_seeds for ii in item]
    mean_kshots.append(np.mean(acc_seeds))
    all_std_kshots.append(np.std(acc_seeds))
    acc_kshots.append(acc_seeds)
    std_kshots.append(std_seeds)
# print(acc_kshots)
# print(std_kshots)
print(exp)
print('Mean (Std): ', [f'{mean_kshots[i]:.1f} ({all_std_kshots[i]:.1f}) | ' for i in range(len(mean_kshots))])
