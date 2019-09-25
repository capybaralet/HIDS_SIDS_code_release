#!/usr/bin/env python

"""
TODO
"""


# ---------------------------------------------------------------
import argparse
import os
import shutil
import sys
import numpy 
np = numpy

import time

from pylab import *
#from capy_torch_utils import *

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


parser = argparse.ArgumentParser()
#
# ENVIRONMENT SETTINGS:
parser.add_argument('--nonstationary_P_x', type=int, default=1)
parser.add_argument('--nonstationary_P_y_given_x', type=int, default=1)
parser.add_argument('--n_users', type=int, default=10)
parser.add_argument('--n_articles', type=int, default=10)
parser.add_argument('--loyalty_update_rate', type=float, default=0.03) # alpha_1
parser.add_argument('--interest_update_rate', type=float, default=0.003) # alpha_2
parser.add_argument('--init_scale', type=float, default=0.03)
parser.add_argument('--normalize_W', type=int, default=1)
#
# LEARNING ALGORITHM SETTINGS:
parser.add_argument('--learning_rate', type=float, default=.01) 
parser.add_argument('--n_hids', type=int, default=100)
#
# EXPERIMENT SETTINGS: 
parser.add_argument('--environments_diverge', type=int, default=1) # environments of different learners diverge over time
parser.add_argument('--n_envs', type=int, default=20) # if using PBT, this should be a multiple of 5 
parser.add_argument('--n_trials', type=int, default=20) 
parser.add_argument('--n_steps', type=int, default=2000)
parser.add_argument('--randomize_seeds', type=int, default=0)
parser.add_argument('--starting_seed', type=int, default=0)
parser.add_argument('--shared_starting_env', type=int, default=1) # learners start in an environment with the same user distribution and user interests
parser.add_argument('--train', type=int, default=1) 
#
# OTHER:
parser.add_argument('--save_dir', type=str, default=os.environ['SCRATCH']) # N.B.! you must specify the environment variable SCRATCH.  you can do this like: export $SCRATCH=<<complete file-path for the save_dir>>
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--make_plots', type=int, default=1) 
parser.add_argument('--KL_hack', type=int, default=1) 
# NEW!
# NEW NEW
parser.add_argument('--PBT_interval', type=int, default=0) # TODO: check it
parser.add_argument('--environment_swapping', type=int, default=0)
# NEW NEW NEW
parser.add_argument('--explore', type=int, default=0)



# ---------------------------------------------------------------
# PARSE ARGS and SET-UP SAVING (save_path/exp_settings.txt)
# NTS: we name things after the filename + provided args.  We could also save versions (ala Janos), and/or time-stamp things.

args = parser.parse_args()
print (args)
args_dict = args.__dict__

# TODO: why do I get single quotes around the save_dir filepath??
if args_dict['save_dir'] is None: 
    try:
        save_dir = os.environ['SCRATCH']
    except:
        print ("\n\n\n\t\t\t\t WARNING: save_dir is None! Results will not be saved! \n\n\n")
else:
    # save_dir = filename + PROVIDED parser arguments
    flags = [flag.lstrip('--') for flag in sys.argv[1:] if not (flag.startswith('--save_dir') or flag.startswith('--train'))]
    save_dir = os.path.join(args_dict.pop('save_dir'), os.path.basename(__file__) + '___' + '_'.join(flags))
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


if 1:#PBT_interval > 0:
    assert n_envs % 5 == 0

if nonstationary_P_y_given_x:
    assert normalize_W

if not environments_diverge:
    assert shared_starting_env

if randomize_seeds:
    starting_seed = np.random.choice(2**32-1)
else:
    starting_seed = starting_seed

# ---------------------------------------------------------------

if make_plots:
    close('all')

    n_plot_points = 20 # TODO: this seems to be hardcoded somewhere...
    plot_interval = int(n_steps / n_plot_points)
    #figure(1)
    f, axs = plt.subplots(1, 3)#, sharey=True)
    f.set_size_inches(10,3.8)

    plot_refs = []
    plot_refs12 = []
    #linestyles = ['-', '--', '-.', ':']
    markers = ['s', 'P', 'D', 'X']
    colors = ['darkorchid', 'darkcyan', 'palevioletred', 'limegreen']
    perm = [1,0,3,2]
    if not environment_swapping or not environments_diverge: # no env swapping condition
        markers = markers[:2]
        colors = colors[:2]
        perm = perm[:2]

labels = []
plot_num = -1

t0 = time.time()




if environment_swapping:
    environment_swappings = [0,1]
else:
    environment_swappings = [0]
PBT_intervals = [0,10]





for environment_swapping in environment_swappings:
    for PBT_interval in PBT_intervals:
        plot_num += 1 # TODO: hacky...
        color = colors[plot_num]
        marker = markers[plot_num]

        if train:

            all_accuracies = []
            all_Ws = []
            all_gs = []


            # set label
            if PBT_interval == 0:
                label = "no PBT"
            else:
                label = "PBT"
            if environment_swapping:
                label += ", env. swapping"
            labels.append(label)

            # run multiple trials, for statistical significance.
            for n_, seed in enumerate(range(starting_seed, n_trials + starting_seed)):
                print ("\t\t\t\t\t TRIAL #", n_+1, "\t PBT=", PBT_interval)
                #print ("\t\t\t\t\t shared_starting_env=", shared_starting_env, "\t environment_swapping=", environment_swapping, "\t PBT=", PBT_interval )
                print ("\t\t\t\t\t time", time.time() - t0)


                np.random.seed(seed)
                rng = np.random.RandomState(seed)
                #
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)



                # ---------------------------------------------------------------
                # taken from https://raw.githubusercontent.com/andrewliao11/dni.pytorch/master/mlp.py
                class MLP(nn.Module):
                    def __init__(self, input_size, hidden_size, num_classes):
                        super(MLP, self).__init__()
                        self.fc1 = nn.Linear(input_size, hidden_size)
                        self.relu = nn.ReLU()
                        self.fc2 = nn.Linear(hidden_size, num_classes)
                        self.smax = nn.Softmax()

                    def forward(self, x):
                        out = self.fc1(x)
                        out = self.relu(out)
                        out = self.fc2(out)
                        out = self.smax(out)
                        return out


                # ---------------------------------------------------------------

                # SET-UP ENVIRONMENTS
                # The state consists of the loyalty and interests of different types of users, and the current user and which article they've clicked on:
                gs = T.zeros((n_steps, n_envs, n_users)) # loyalty (determines distribution of user types)
                xs = T.zeros((n_steps, n_envs, 1), dtype=T.long) # current user type
                Ws = T.zeros((n_steps, n_envs, n_articles, n_users)) # interests
                hs = T.zeros((n_steps, n_envs, n_articles)) # The interests of the current user type
                # which article the user clicks on
                ys = T.zeros((n_steps, n_envs, 1), dtype=T.long)


                # SET-UP LEARNERS
                # the model's predictive distribution for y
                y_hats = T.zeros((n_steps, n_envs, n_articles))
                # which article was placed first
                y_sampleds = T.zeros((n_steps, n_envs, 1), dtype=T.long)
                # was the prediction correct?
                accuracies = T.zeros((n_steps, n_envs, 1), dtype=T.long)
                # NLL
                losses = T.zeros((n_steps, n_envs))
                #
                if explore:
                    lrs = 10.**(np.random.uniform(-2,0,size=n_envs))
                else:
                    lrs = [learning_rate for n in range(n_envs)] 
                classifiers = [MLP(n_users, n_hids, n_articles) for n in range(n_envs)]
                optimizers = [torch.optim.SGD(classifiers[n].parameters(), lr=lrs[n]) for n in range(n_envs)]

                if shared_starting_env: # all agents start in an identical environment 
                    g = T.randn(1, n_users).expand(n_envs, n_users) # loyalty (determines distribution of user types)
                    W = T.randn(1, n_articles, n_users).expand(n_envs, n_articles, n_users) # interests
                else:
                    # agents start in different random environments
                    g = T.randn(n_envs, n_users)
                    W = T.randn(n_envs, n_articles, n_users)
                if init_scale is not None:
                    g = init_scale * g
                    W = init_scale * W
                if normalize_W:
                    if normalize_W:
                        if environments_diverge:
                            W = W / (((W**2).sum(1))**.5).unsqueeze(1)
                        else:
                            W /= ((((W**2).sum(1))**.5).unsqueeze(1)).mean(-1, keepdim=True)


                # N.B: the computations here are a bit weird, since the state updates are "out of sync" according to the usual conventions of RL
                x_onehot = T.zeros(n_envs, n_users)
                y_sampled_onehot = T.zeros(n_envs, n_articles)
                W_onehot = torch.einsum("eu,ea->eau", x_onehot, y_sampled_onehot)
                for t in range(n_steps):

                    # ENVIRONMENT DYNAMICS
                    # sample users based on loyalty
                    x = nn.Softmax(dim=-1)(g).multinomial(1)
                    xs[t] = x
                    x_onehot.zero_()
                    x_onehot.scatter_(1, x, 1)
                    # compute the users' current interests
                    h = (W * x_onehot.unsqueeze(1)).sum(-1)
                    hs[t] = h
                    # sample which article the users click on
                    y = nn.Softmax(dim=-1)(10*h).multinomial(1)
                    ys[t] = y

                    # PREDICTION / ACTION
                    for n in range(n_envs):
                        optimizers[n].zero_grad()
                        y_hat = classifiers[n](x_onehot[n])
                        y_hats[t,n] = y_hat
                        y_sampled = y_hat.multinomial(1)
                        y_sampleds[t,n] = y_sampled
                        accuracies[t,n] = (y_sampled == y[n])
                        # LEARNING
                        loss = nn.CrossEntropyLoss()(y_hat.unsqueeze(0), y[n:n+1,0])
                        losses[t,n] = loss
                        loss.sum().backward()
                        optimizers[n].step()
                    y_sampled = y_sampleds[t]
                    if PBT_interval > 0 and t % PBT_interval == 0: # apply PBT
                        srts = T.argsort(accuracies[t,:,0]).flip((0,)) # lower index ==> higher fitness
                        #print (srts)
                        # EXPLOIT
                        parents = np.random.choice(int(.2*n_envs), int(.2*n_envs), replace = True) 
                        #print (srts)
                        #print (parents)
                        for n in range(len(parents)):
                            parent = srts[parents[n]]
                            child = srts[-1 - n]
                            for p_new, p_old in zip(classifiers[parent].parameters(), classifiers[child].parameters()):
                                p_old.data = p_new.data
                            lrs[child] = lrs[parent]

                        # EXPLORE
                        if explore:
                            noise = (rand(n_envs) > .5).astype('float')
                            lrs = [lrs[n] * 1.2 * noise[n] + lrs[n] * .8 * (1 - noise[n]) for n in range(n_envs)]
                            optimizers = [torch.optim.SGD(classifiers[n].parameters(), lr=lrs[n]) for n in range(n_envs)]

                    if environment_swapping:
                        classifiers = [classifiers[n-1] for n in range(len(classifiers))]
                        optimizers = [optimizers[n-1] for n in range(len(classifiers))]
                        lrs = [lrs[n-1] for n in range(len(classifiers))]


                    # ENVIRONMENT DYNAMICS
                    # update P(x) (i.e. user loyalty) based on self-selection effect
                    if nonstationary_P_x:
                        # update P(x) (i.e. user loyalty) based on self-selection effect
                        y_sampled_onehot.zero_()
                        y_sampled_onehot.scatter_(1, y_sampled, 1)
                        if environments_diverge:
                            g = g + (loyalty_update_rate * x_onehot * (h * y_sampled_onehot).sum().unsqueeze(-1))
                        else:
                            update = (loyalty_update_rate * x_onehot * (h * y_sampled_onehot).sum().unsqueeze(-1)).mean(0, keepdim=True) / n_envs
                            #print ("checking updates")
                            #print (g, update)
                            g += update
                            #print (g)
                    if nonstationary_P_y_given_x:
                        # update P(y|x) (i.e. user interests) based on illusory truth effect
                        y_sampled_onehot.zero_()
                        y_sampled_onehot.scatter_(1, y_sampled, 1)
                        W_onehot = torch.einsum("eu,ea->eau", x_onehot, y_sampled_onehot)
                        if environments_diverge:
                            W = W + interest_update_rate * W_onehot # this assumes normalize_W!
                        else:
                            update = (interest_update_rate * W_onehot).mean(0, keepdim=True) / n_envs
                            #print (W, update)
                            W += update
                            #print (W)
                        if normalize_W:
                            if environments_diverge:
                                W = W / (((W**2).sum(1))**.5).unsqueeze(1)
                            else:
                                update = ((((W**2).sum(1))**.5).unsqueeze(1)).mean(0, keepdim=True)
                                #print (W, update)
                                W /= update
                                #print (W)
                        #assert False

                    gs[t] = g
                    Ws[t] = W

                    if verbose and t % 100 == 0:
                        print (t)

                all_accuracies.append(accuracies[:,:,0].sum(-1).numpy() / n_envs)
                all_Ws.append([torch.einsum('eau,eau->eu', Ws[0],Ws[t]).mean() for t in range(0, n_steps, 1)])
                # TODO: change names! (bad name!)
                all_gs.append(nn.Softmax(dim=-1)(gs).numpy())

            all_accuracies = np.array(all_accuracies)
            all_Ws = np.array(all_Ws)

            ##################print (all_accuracies[:,::plot_interval])

            # compute KL divergence for P(x) TODO: move / make efficient
            def kl(p,q):
                return np.where(p==0, 0, (p * (np.log(p) - np.log(q)))).sum()
                #return (p * (np.log(p) - np.log(q))).sum()

            all_KLs = np.empty((n_trials, n_steps, n_envs))
            for trial in range(n_trials):
                for env in range(n_envs):
                    for step in range(n_steps):
                        this_kl = kl(all_gs[trial][step, env], all_gs[trial][0, env])
                        if KL_hack and np.isnan(this_kl): # When the KL becomes NaN, it is due to numerical overflow, and the last non-NaN can be substituted.
                            this_kl = all_KLs[trial,step-1,env]
                        all_KLs[trial,step,env] = this_kl
            all_KLs = all_KLs.mean(-1)

            if environment_swapping:
                if PBT_interval == 0:
                    np.save(os.path.join(save_dir, 'all_accuracies_with_environment_swapping.npy'), all_accuracies)
                    np.save(os.path.join(save_dir, 'all_Ws_with_environment_swapping.npy'), all_Ws)
                    np.save(os.path.join(save_dir, 'all_gs_with_environment_swapping.npy'), np.array(all_gs))
                    np.save(os.path.join(save_dir, 'all_KLs_with_environment_swapping.npy'), np.array(all_KLs))
                else:
                    np.save(os.path.join(save_dir, 'all_accuracies_PBT_with_environment_swapping.npy'), all_accuracies)
                    np.save(os.path.join(save_dir, 'all_Ws_PBT_with_environment_swapping.npy'), all_Ws)
                    np.save(os.path.join(save_dir, 'all_gs_PBT_with_environment_swapping.npy'), np.array(all_gs))
                    np.save(os.path.join(save_dir, 'all_KLs_PBT_with_environment_swapping.npy'), np.array(all_KLs))
            else:
                if PBT_interval == 0:
                    np.save(os.path.join(save_dir, 'all_accuracies.npy'), all_accuracies)
                    np.save(os.path.join(save_dir, 'all_Ws.npy'), all_Ws)
                    np.save(os.path.join(save_dir, 'all_gs.npy'), np.array(all_gs))
                    np.save(os.path.join(save_dir, 'all_KLs.npy'), np.array(all_KLs))
                else:
                    np.save(os.path.join(save_dir, 'all_accuracies_PBT.npy'), all_accuracies)
                    np.save(os.path.join(save_dir, 'all_Ws_PBT.npy'), all_Ws)
                    np.save(os.path.join(save_dir, 'all_gs_PBT.npy'), np.array(all_gs))
                    np.save(os.path.join(save_dir, 'all_KLs_PBT.npy'), np.array(all_KLs))

        else: # just load and make plots!
            if environment_swapping:
                if PBT_interval == 0:
                    all_accuracies = np.load(os.path.join(save_dir, 'all_accuracies_with_environment_swapping.npy'))
                    all_Ws = np.load(os.path.join(save_dir, 'all_Ws_with_environment_swapping.npy'))
                    all_gs = np.load(os.path.join(save_dir, 'all_gs_with_environment_swapping.npy'))
                    all_KLs = np.load(os.path.join(save_dir, 'all_KLs_with_environment_swapping.npy'))
                else:
                    all_accuracies = np.load(os.path.join(save_dir, 'all_accuracies_PBT_with_environment_swapping.npy'))
                    all_Ws = np.load(os.path.join(save_dir, 'all_Ws_PBT_with_environment_swapping.npy'))
                    all_gs = np.load(os.path.join(save_dir, 'all_gs_PBT_with_environment_swapping.npy'))
                    all_KLs = np.load(os.path.join(save_dir, 'all_KLs_PBT_with_environment_swapping.npy'))
            else:
                if PBT_interval == 0:
                    all_accuracies = np.load(os.path.join(save_dir, 'all_accuracies.npy'))
                    all_Ws = np.load(os.path.join(save_dir, 'all_Ws.npy'))
                    all_gs = np.load(os.path.join(save_dir, 'all_gs.npy'))
                    all_KLs = np.load(os.path.join(save_dir, 'all_KLs.npy'))
                    if 0:# distributional shift vs. accuracy
                        figure(11)
                        subplot(121)
                        for i in range(20):
                            plot(all_accuracies[i][::20], all_KLs[i][::20], color=colors[0], alpha=.1)
                        plot(all_accuracies.mean(0)[::20], all_KLs.mean(0)[::20], color=colors[0], label='no PBT')
                        ylabel('P(X) KL')
                        xlabel('accuracy')
                        subplot(122)
                        for i in range(20):
                            plot(all_accuracies[i][::20], 1 - all_Ws[i][::20], color=colors[0], alpha=.1)
                        plot(all_accuracies.mean(0)[::20], 1 - all_Ws.mean(0)[::20], color=colors[0], label='no PBT')
                        ylabel('P(Y|X) (cosine)')
                        xlabel('accuracy')
                    if 1:
                        figure(12)
                        binned_accuracies = [np.where(np.isclose(all_accuracies,.05 * i)) for i in range(21)]
                        #
                        subplot(121)
                        corresponding_KLs = [all_KLs[i,j] for i,j in binned_accuracies]
                        KL_means = np.array([kl.mean() for kl in corresponding_KLs])
                        KL_stderrs = np.array([1.96 * kl.std() / len(kl)**.5 for kl in corresponding_KLs])
                        plot(np.arange(21) * .05, KL_means, label='no PBT', color=colors[0])
                        fill_between(np.arange(21) * .05, KL_means - KL_stderrs, KL_means + KL_stderrs, color=colors[0], alpha=.2, edgecolor=(0,0,0))
                        #
                        subplot(122)
                        corresponding_Ws = [all_Ws[i,j] for i,j in binned_accuracies]
                        W_means = 1 - np.array([w.mean() for w in corresponding_Ws])
                        W_stderrs = np.array([1.96 * w.std() / len(w)**.5 for w in corresponding_Ws])
                        plot(np.arange(21) * .05, W_means, label='no PBT', color=colors[0])
                        fill_between(np.arange(21) * .05, W_means - W_stderrs, W_means + W_stderrs, color=colors[0], alpha=.2, edgecolor=(0,0,0))
                else:
                    all_accuracies = np.load(os.path.join(save_dir, 'all_accuracies_PBT.npy'))
                    all_Ws = np.load(os.path.join(save_dir, 'all_Ws_PBT.npy'))
                    all_gs = np.load(os.path.join(save_dir, 'all_gs_PBT.npy'))
                    all_KLs = np.load(os.path.join(save_dir, 'all_KLs_PBT.npy'))
                    if 0:# distributional shift vs. accuracy
                        figure(11)
                        subplot(121)
                        for i in range(20):
                            plot(all_accuracies[i][::20], all_KLs[i][::20], color=colors[1], alpha=.1)
                        plot(all_accuracies.mean(0)[::20], all_KLs.mean(0)[::20], color=colors[1], label='no PBT')
                        ylabel('P(X) KL')
                        xlabel('accuracy')
                        subplot(122)
                        for i in range(20):
                            plot(all_accuracies[i][::20], 1 - all_Ws[i][::20], color=colors[1], alpha=.1)
                        plot(all_accuracies.mean(0)[::20], 1 - all_Ws.mean(0)[::20], color=colors[1], label='PBT')
                        ylabel('P(Y|X) (cosine)')
                        xlabel('accuracy')
                    if 1:
                        figure(12)
                        binned_accuracies = [np.where(np.isclose(all_accuracies, .05 * i)) for i in np.arange(21)]
                        #
                        subplot(121)
                        corresponding_KLs = [all_KLs[i,j] for i,j in binned_accuracies]
                        KL_means = np.array([kl.mean() for kl in corresponding_KLs])
                        KL_stderrs = np.array([1.96 * kl.std() / len(kl)**.5 for kl in corresponding_KLs])
                        plot(np.arange(21) * .05, KL_means, label='PBT', color=colors[1])
                        fill_between(np.arange(21) * .05, KL_means - KL_stderrs, KL_means + KL_stderrs, color=colors[1], alpha=.2, edgecolor=(0,0,0))
                        ylabel('P(X) (KL)')
                        xlabel('accuracy')
                        #
                        subplot(122)
                        corresponding_Ws = [all_Ws[i,j] for i,j in binned_accuracies]
                        W_means = 1 - np.array([w.mean() for w in corresponding_Ws])
                        W_stderrs = np.array([1.96 * w.std() / len(w)**.5 for w in corresponding_Ws])
                        plot(np.arange(21) * .05, W_means, label='PBT', color=colors[1])
                        fill_between(np.arange(21) * .05, W_means - W_stderrs, W_means + W_stderrs, color=colors[1], alpha=.2, edgecolor=(0,0,0))
                        ylabel('P(Y|X) (cosine)')
                        xlabel('accuracy')


        ################################
        ################################
        ################################
        # PLOTTING

        if make_plots:

            print ("\t\tbegin plotting\t", label)
            labelpad = 2.

            f12 = figure(12)
            binned_accuracies = [np.where(np.isclose(all_accuracies,.05 * i)) for i in range(21)]
            #
            subplot(121)
            corresponding_KLs = [all_KLs[i,j] for i,j in binned_accuracies]
            KL_means = np.array([kl.mean() for kl in corresponding_KLs])
            KL_stderrs = np.array([1.96 * kl.std() / len(kl)**.5 for kl in corresponding_KLs])
            plot_refs12.append(plot(np.arange(21) * .05, KL_means, label=label, color=color))#, marker=marker))
            fill_between(np.arange(21) * .05, KL_means - KL_stderrs, KL_means + KL_stderrs, color=color, alpha=.2, edgecolor=(0,0,0))
            title('Change in P(X) (user base)')
            ylabel("KL div. from original distribution", labelpad=labelpad)
            xlabel('accuracy')
            #
            subplot(122)
            corresponding_Ws = [all_Ws[i,j] for i,j in binned_accuracies]
            W_means = 1 - np.array([w.mean() for w in corresponding_Ws])
            W_stderrs = np.array([1.96 * w.std() / len(w)**.5 for w in corresponding_Ws])
            plot(np.arange(21) * .05, W_means, label=label, color=color)#, marker=marker)
            fill_between(np.arange(21) * .05, W_means - W_stderrs, W_means + W_stderrs, color=color, alpha=.2, edgecolor=(0,0,0))
            savefig(os.path.join(save_dir, 'fig4.png'), bbox_inches='tight')
            title('Change in P(Y|X) (user interests)')#: inverse cosine similarity between users original and final interests')
            xlabel('accuracy')
            ylabel("cosine distance from original interests", labelpad=labelpad)



            # PLOT 1
            #
            axs[0].set_title('Accuracy of click prediction')
            mean_ = all_accuracies[:,::plot_interval].mean(0)
            std_err = all_accuracies[:,::plot_interval].std(0) / n_trials**.5 * 1.96 #1.96 * all_accuracies[:,::plot_interval].std(0) / n_trials**.5)
            this_plot = axs[0].errorbar(range(n_plot_points), mean_, [0 for n in range(n_plot_points)], color=colors[plot_num], marker=markers[plot_num])
            plot_refs.append(this_plot)
            axs[0].fill_between(range(n_plot_points), mean_ - std_err, mean_ + std_err, color=colors[plot_num], alpha=.2, edgecolor=(0,0,0))
            axs[0].axvline(x=int(n_plot_points * .25), linestyle='--', color='k', alpha=.1)
            axs[0].set_xlabel('time-step')
            axs[0].set_ylabel('Accuracy', labelpad=labelpad)
            axs[0].set_xticks(range(0, 21, 4))#, range(0,n_steps+1,int(n_steps / 5.)))
            axs[0].set_xticklabels(range(0,n_steps+1,int(n_steps / 5.)))
            #
            axs[1].set_title('Change in P(Y|X) (user interests)')#: inverse cosine similarity between users original and final interests')
            mean_ = 1 - all_Ws[:,::plot_interval].mean(0)
            std_err = all_Ws[:,::plot_interval].std(0) / n_trials**.5 * 1.96 #1.96 * all_accuracies[:,::plot_interval].std(0) / n_trials**.5)
            axs[1].errorbar(range(n_plot_points), mean_, [0 for n in range(n_plot_points)], color=colors[plot_num], marker=markers[plot_num])
            axs[1].fill_between(range(n_plot_points), mean_ - std_err, mean_ + std_err, color=colors[plot_num], alpha=.2, edgecolor=(0,0,0))
            axs[1].axvline(x=int(n_plot_points * .25), linestyle='--', color='k', alpha=.1)
            #axs[1].errorbar(range(n_plot_points), 
            #              - all_Ws[:,::plot_interval].mean(0),
            #              #- 1.96 * all_Ws[:,::plot_interval].std(0) / n_trials**.5)
            #              all_Ws[:,::plot_interval].std(0) / n_trials**.5,
            #              marker=markers[plot_num])
            axs[1].set_xlabel('time-step')
            axs[1].set_ylabel("cosine distance from original interests", labelpad=labelpad)
            axs[1].set_xticks(range(0, 21, 4))#, range(0,n_steps+1,int(n_steps / 5.)))
            axs[1].set_xticklabels(range(0,n_steps+1,int(n_steps / 5.)))
            #
            #
            axs[2].set_title('Change in P(X) (user base)')#: KL-Divergence between original and final user distribution')
            mean_ = all_KLs[:,::plot_interval].mean(0)
            std_err = all_KLs[:,::plot_interval].std(0) / n_trials**.5 * 1.96 #1.96 * all_accuracies[:,::plot_interval].std(0) / n_trials**.5)
            axs[2].errorbar(range(n_plot_points), mean_, [0 for n in range(n_plot_points)], color=colors[plot_num], marker=markers[plot_num])
            axs[2].fill_between(range(n_plot_points), mean_ - std_err, mean_ + std_err, color=colors[plot_num], alpha=.2, edgecolor=(0,0,0))
            axs[2].axvline(x=int(n_plot_points * .25), linestyle='--', color='k', alpha=.1)
            axs[2].set_xlabel('time-step')
            axs[2].set_ylabel("KL div. from original distribution", labelpad=labelpad)
            axs[2].set_xticks(range(0, 21, 4))#, range(0,n_steps+1,int(n_steps / 5.)))
            axs[2].set_xticklabels(range(0,n_steps+1,int(n_steps / 5.)))
            #f.legend(plot_refs, )
            f.savefig(os.path.join(save_dir, 'results_plot.png'), bbox_inches='tight')
















            # TODO: reintroduce these
            if 0:
                # PLOT 2
                # visualizing the change in P(x) via a "3D-tube plot"
                n_trials_to_plot = min(n_trials, 5) # plot TWO random environments from each of the first 5 trials
                #fig = plt.figure(plot_offset + 3,figsize=plt.figaspect(4. / 5.))
                fig = plt.figure(2,figsize=plt.figaspect(4. / 5.))
                fig.set_size_inches(15,8)
                for trial_n in range(n_trials_to_plot):
                    
                    # random env 1
                    for n_, ind in enumerate(np.random.choice(n_envs, 2, replace=False)):
                        if PBT_interval == 0:
                            ax = fig.add_subplot(4, 5, 1 + trial_n + 10*n_, projection='3d')
                            X = np.arange(n_steps)
                            Y = np.arange(n_users)
                            X,Y = np.meshgrid(X,Y)
                            Z = all_gs[trial_n][:,ind,:].T
                            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                            title('no PBT')
                            fig.colorbar(surf, shrink=0.5, aspect=5)
                        else:
                            ax = fig.add_subplot(4, 5, 6 + trial_n + 10*n_, projection='3d')
                            X = np.arange(n_steps)
                            Y = np.arange(n_users)
                            X,Y = np.meshgrid(X,Y)
                            Z = all_gs[trial_n][:,ind,:].T
                            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                            title('PBT')
                            fig.colorbar(surf, shrink=0.5, aspect=5)

                savefig(os.path.join(save_dir, 'tube_plot.png'), bbox_inches='tight')
                
                #plt.show()

        if 0:
            print ("reproduces result?")
            print (np.all(all_accuracies == np.load('all_accuracies_test_values.npy')))
            print (np.all(all_Ws == np.load('all_Ws_test_values.npy')))
            #
            print("seed", starting_seed)
            #show()

f.legend([plot_refs[per] for per in perm], [labels[per] for per in perm], loc='lower center', ncol=4)#, fontsize='large')
f.subplots_adjust(
top=0.907,
bottom=0.213,
left=0.064,
right=0.967,
hspace=0.2,
wspace=0.302,
#top=0.88,
#bottom=0.215,
#left=0.1,
#right=0.95,
#hspace=0.2,
#wspace=0.25,
)
f.savefig(os.path.join(save_dir, 'results_plot.png'), bbox_inches='tight')

f12.legend([plot_refs[per] for per in perm], [labels[per] for per in perm], loc='lower center', ncol=4)#, fontsize='large')
f.subplots_adjust(
top=0.907,
bottom=0.213,
left=0.064,
right=0.967,
hspace=0.2,
wspace=0.302,
#top=0.88,
#bottom=0.215,
#left=0.1,
#right=0.95,
#hspace=0.2,
#wspace=0.25,
)
f12.savefig(os.path.join(save_dir, 'fig4_adjusted.png'), bbox_inches='tight')

#figure(12)
#suptitle("We want to see PBT being ABOVE noPBT; this means there is MORE distributional shift for a given level of accuracy")
#legend()

