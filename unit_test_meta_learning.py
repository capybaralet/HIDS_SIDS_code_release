#!/usr/bin/env python

# ---------------------------------------------------------------
import argparse
import os
import shutil
import sys
import numpy 
np = numpy

import time

from pylab import *

import torch
T = torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# for 3D tube plots
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

################################
parser = argparse.ArgumentParser()
parser.add_argument('--n_agentss', type=str, default="[10,100,1000]") # LINE 3, 10, 100, 1000
#parser.add_argument('--learning_rate', type=float, default=.01) # LINE .1, .01, .001, .0001
parser.add_argument('--n_trials', type=int, default=10) 
parser.add_argument('--n_steps', type=int, default=1000)
parser.add_argument('--randomize_seeds', type=int, default=1) # TODO
parser.add_argument('--intervals', type=str, default="[0,1,10,100]") 
parser.add_argument('--momentum', type=float, default=0) 
parser.add_argument('--IL', type=str, default='REINFORCE') 
parser.add_argument('--OL', type=str, default='PBT') # it would make more sense if this was only optimizing hparams... but that's just not going to work... we'd need a more complicated environment...
parser.add_argument('--OL_lr', type=float, default=.1) 
parser.add_argument('--OL_target', type=str, default=None) 
parser.add_argument('--env_swaps', type=str, default="[none,deterministic]") 
parser.add_argument('--b1s', type=str, default="[0.5, 0, -0.5]")  # aka "beta"
#parser.add_argument('--b1s', type=str, default="-.5")  # aka "beta"
parser.add_argument('--permute', type=int, default=1) 
#permute = 1
#parser.add_argument('--normal_init', type=float, default=0) 
normal_init = 1
#
parser.add_argument('--save_dir', type=str, default=os.environ['SCRATCH']) # N.B.! you must specify the environment variable SCRATCH.  you can do this like: export $SCRATCH=<<complete file-path for the save_dir>>
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--make_plots', type=int, default=1) 
parser.add_argument('--plot_all', type=int, default=0) 
parser.add_argument('--fig_n_start', type=int, default=0) 

# ---------------------------------------------------------------
# PARSE ARGS and SET-UP SAVING (save_path/exp_settings.txt)
# NTS: we name things after the filename + provided args.  We could also save versions (ala Janos), and/or time-stamp things.

args = parser.parse_args()
print (args)
args_dict = args.__dict__

# TODO: why do I end up with single quotes around the directory name?
if args_dict['save_dir'] is None:
    try:
        save_dir = os.environ['SCRATCH']
    except:
        print ("\n\n\n\t\t\t\t WARNING: save_dir is None! Results will not be saved! \n\n\n")
else:
    # save_dir = filename + PROVIDED parser arguments
    flags = [flag.lstrip('--') for flag in sys.argv[1:] if not (flag.startswith('--save_dir') or flag.startswith('--train'))]
    exp_title = '_'.join(flags)
    save_dir = os.path.join(args_dict.pop('save_dir'), os.path.basename(__file__) + '___' + exp_title)
    print("\t\t save_dir=",  save_dir)

    # make directory for results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save a copy of THIS SCRIPT in save_dir
    shutil.copy(__file__, os.path.join(save_dir,'exp_script.py'))
    # save ALL parser arguments
    with open (os.path.join(save_dir,'exp_settings.txt'), 'w') as f:
        for key in sorted(args_dict):
            f.write(key+'\t'+str(args_dict[key])+'\n')

locals().update(args_dict)

# 
if OL_target is None:
    if OL == 'PBT':
        OL_target = "final"
    elif OL == "REINFORCE":
        OL_target = "sum"
    else:
        assert False

# this allows us to pass a list or a single value for these args
try:
    b1s = [float(b1s)]
except:
    b1s = [float(s) for s in b1s[1:-1].split(',')]
try:
    intervals = [int(intervals)]
except:
    intervals = [int(s) for s in intervals[1:-1].split(',')]
try:
    n_agentss = [int(n_agentss)]
except:
    n_agentss = [int(s) for s in n_agentss[1:-1].split(',')]
if env_swaps.startswith('['):
    env_swaps = [s for s in env_swaps[1:-1].split(',')]



##########################################################################3


#close('all')

exp_num = 0

t0 = time.time()


for fig_n, b1 in enumerate(b1s):
    f_name = "unit_test, beta = " + str(b1)
    close(f_name)
    f, axs = plt.subplots(2, 3, sharey=True, sharex=True, num=f_name)
    plot_refs = []
    labels = []
    for m, env_swap in enumerate(env_swaps):
        for n, n_agents in enumerate(n_agentss):
            for _, interval in enumerate(intervals):
                exp_num += 1
                print ("\nexp_num=", exp_num)
                print ("n_agents=", n_agents)
                print ("interval=", interval)

                if normal_init:
                    param = normal_init * randn(n_trials, n_agents)
                else:
                    param = rand(n_trials, n_agents)
                    param = np.log(param / (1. - param))

                # initial states
                state_is_cooperate = (rand(n_trials, n_agents) > .5).astype('int')
                # initial learning rates
                lr = 10**np.random.uniform(-2,0, (n_trials, n_agents))

                def reward_fn(state, action, b1=b1):
                    """ 0 = defect = silent """
                    return state + (b1 * action) - .5

                def sigmoid(x):
                    return 1. / (1 + np.exp(-x))

                if OL == 'analytic': # TODO! should the OL select on FINAL or AVERAGE performance? 
                    pass # TODO

                # logging:
                P_cooperates = -1. * np.ones((n_trials, n_steps, n_agents))

                n_parents = int(.2 * n_agents)
                print ("n_parents=", n_parents)
                print ("")

                OL_grad = 0
                OL_reward = 0

                for step in range(n_steps):

                    # action, reward, observation
                    P_cooperate = sigmoid(param) 
                    action_is_cooperate = (rand(n_trials, n_agents) < P_cooperate).astype('int')
                    P_cooperates[:,step,:] = P_cooperate
                    reward = reward_fn(state_is_cooperate, action_is_cooperate)

                    # TODO: double check this!
                    grad = action_is_cooperate * 1 / (1 + np.exp(param)) + (1 - action_is_cooperate) *  (1 / (1 + np.exp(param)) - 1) # d(P(a_t))/d(theta) (done by hand, and by Wolfram Alpha)
                    OL_reward += reward
                    OL_grad += grad

                    # inner loop updates
                    if IL == "REINFORCE":
                        # probability of the action taken
                        update = lr * reward * grad
                        param += update
                    elif IL == "analytic":
                        update = b1 * sigmoid(param) * (1 - sigmoid(param)) # dL / d(param) (done by hand)
                        param += lr * update
                    elif IL == "NONE":
                        pass
                    else:
                        assert False

                    state_is_cooperate = action_is_cooperate

                    if step % 125 == 0:
                        #print (P_cooperate)
                        print (P_cooperate.mean())

                    if interval > 0 and (step+1) % interval == 0:
                        if OL.startswith("PBT"):
                            parents_inds = np.random.choice(range(n_parents), size=(n_trials, n_parents), replace=True)
                            for trial in range(n_trials): 
                                # EXPLOIT:
                                if OL_target == 'final':
                                    if permute: # see https://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
                                        srts = np.lexsort((randn(n_agents), reward[trial]))
                                    else: # this didn't make a difference as far as we could tell.
                                        srts = np.argsort(reward[trial])
                                elif OL_target == 'sum':
                                    if permute:
                                        srts = np.lexsort((randn(n_agents), OL_reward[trial]))
                                    else:
                                        srts = np.argsort(OL_reward[trial])
                                else:
                                    assert False
                                winners = srts[-n_parents:] # These are the ones which scored well enough to be elligible to reproduce.
                                parents = winners[parents_inds[trial]] # These are the actual parents (for the new offspring)
                                param[trial, srts[:n_parents]] = param[trial, parents]
                                lr[trial, srts[:n_parents]] = lr[trial, parents]
                            # EXPLORE:
                            if not OL == "PBT_no_EXPLORE":
                                noise = (rand(n_trials, n_agents) > .5).astype('float')
                                lr = lr * 1.2 * noise + lr * .8 * (1 - noise)
                            if OL_target == 'sum':
                                OL_reward = 0
                        elif OL == "REINFORCE":
                            update = OL_lr * OL_reward * OL_grad / interval
                            param += update
                            OL_grad = 0
                            OL_reward = 0
                        else:
                            assert False

                    # Q: does it matter if I do this before vs. after PBT?   A: yes, in my implementation it does, because we only permute the parameters (not the rewards!)
                    if env_swap == 'deterministic':
                        param = np.hstack((param[:,1:], param[:,:1]))
                    elif env_swap == 'random':
                        param = param[:, np.random.permutation(range(n_agents))]
                    else:
                        pass

                print ("\t\t\t\t\t time", time.time() - t0)

                assert np.all(P_cooperates >= 0) # this makes sure that we've replaced all the initial values (of -1).

                if make_plots:
                    mean_P_cooperates = P_cooperates.mean(-1) # mean over agents
                    mean_ = mean_P_cooperates.mean(0)
                    std_err = mean_P_cooperates.std(0) / n_trials**.5 * 1.96
                    label = "PBT interval="+str(interval)
                    this_plot = axs[m,n].plot(mean_, label=label)
                    color = this_plot[0].get_color()
                    if plot_all:
                        for p in mean_P_cooperates: 
                            axs[m,n].plot(p, color=color, alpha=.15)
                    axs[m,n].fill_between(range(n_steps), mean_ - std_err, mean_ + std_err, alpha = .08, color=color)
                    #
                    if m == 0 and n == 0:
                        plot_refs.append(this_plot)
                        labels.append(label)
                    if m == 0:
                        axs[m,n].set_title("#agents="+str(n_agents))
                    if m == 1:
                        axs[m,n].set_xlabel("time-step")
                    if n == 0:
                        axs[m,n].set_ylabel("P(verbose)") # TODO: RHS
                    if n == 2 and m == 0:
                        axs[m,n].set_ylabel("no env swapping")# env_swap=" + env_swap, ) # TODO: RHS
                        axs[m,n].yaxis.set_label_position("right")
                    if n == 2 and m == 1:
                        axs[m,n].set_ylabel("env swapping")# env_swap=" + env_swap, ) # TODO: RHS
                        axs[m,n].yaxis.set_label_position("right")
                    axs[m,n].set_ylim(0,1)



    print (plot_refs)
    print (labels)

    f.set_size_inches(10,3.8)
    f.subplots_adjust(
    top=0.907,
    bottom=0.213,
    left=0.064,
    right=0.967,
    hspace=0.2,
    wspace=0.302,
    )

    ss = '__beta=' + str(b1)
    f.savefig(os.path.join(save_dir, "results_plot" + ss + ".png"))
    show()








