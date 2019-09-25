import numpy as np
from pylab import *

"""
Value-based learning in the unit-test environment.
Note that the agent needs to start off believing that cooperate is better.
"""

ENVIRONMENT_SWAPPING = False

n_steps = 3000
n_learners = 5
eps = .1

# IIRC, initial state needs to be cooperate
states = [[1] for n in range(n_learners)]#.random.rand() > .5]
actions = [[] for n in range(n_learners)]#.random.rand() > .5]

# all observed rewards
# results depend on iniital values of R0, R1
# IIRC, initial value of R1 needs to be seeded to have a higher value than R0
R0s = [[-.5] for n in range(n_learners)]
R1s = [[0] for n in range(n_learners)]
# Q0 = Q(defect); Q1 = Q(cooperate)
Q0s = [[] for n in range(n_learners)]
Q1s = [[] for n in range(n_learners)]

#close('Q_values')
#figure('Q_values')
#close('P_cooperate')
#figure('P_cooperate')


for step in range(n_steps):
    for n in range(n_learners):
        Q0s[n].append(np.mean(R0s[n]))
        Q1s[n].append(np.mean(R1s[n]))
        if np.random.rand() < eps or Q0s[n][step] == Q1s[n][step]:
            action = int(np.random.rand() > .5)
        elif Q0s[n][step] > Q1s[n][step]:
            action = 0
        else:
            action = 1
        if action == 0 and states[n][step] == 0:
            R0s[n].append(-.5)
        if action == 0 and states[n][step] == 1:
            R0s[n].append(.5)
        if action == 1 and states[n][step] == 0:
            R1s[n].append(-1)
        if action == 1 and states[n][step] == 1:
            R1s[n].append(0)
        states[n].append(action)
        actions[n].append(action)
    #print (Q0,Q1)
    if ENVIRONMENT_SWAPPING:
        states = [states[n-1] for n in range(n_learners)]
        
P_cooperate = [np.cumsum(actions[n]) / np.arange(1, 1+n_steps) for n in range(n_learners)]

# When the values stay close together, the agent cooperates most of the time,
# Otherwise, sometimes Q0 and Q1 start to pull apart (with Q0 > Q1), and the agent never "recovers" and finds the "mostly cooperate" "solution"
# TODO
#f, axs = plt.subplots(3, 4, sharey=False, sharex=True)
f, axs = plt.subplots(2, n_learners, sharey=False, sharex=True)
for n in range(n_learners):
    #subplot(4,2,n+1)
    plotD = axs[0,n].plot(Q0s[n], 'r', label="Q(defect)")
    plotC = axs[0,n].plot(Q1s[n], 'g', label="Q(cooperate)")
    # TODO: env_swapping will mess this up, I fear...
    #subplot(4,2,n+2)
    axs[1,n].plot(P_cooperate[n], 'b', label="#cooperate/time-step")
    axs[1,n].set_xlabel("time-step")
axs[0,0].set_ylabel("Q-values")
axs[1,0].set_ylabel("P(cooperate)")
show()

# LEGEND!
#f.legend([plotC, plotD], ["Q(cooperate)", "Q(defect)"])#, loc='right', ncol=1)
#gs = axs[2, 2].get_gridspec()
#for ax in axs[2]:
#    ax.remove()
#legend_ax = f.add_subplot(gs[2,:])
legendary_material = (
        axs[0,n].get_legend_handles_labels()[0] + axs[1,n].get_legend_handles_labels()[0],
        axs[0,n].get_legend_handles_labels()[1] + axs[1,n].get_legend_handles_labels()[1])
#legend_ax.legend(*legendary_material, ncol=3, loc='center')
figlegend(*legendary_material, ncol=3, loc='lower center')
        #[plotC, plotD], ["Q(cooperate)", "Q(defect)"], loc='center')
#legend_ax.axis('off')
f.subplots_adjust(
    top=0.984,
    bottom=0.306,
    left=0.111,
    right=0.972,
    hspace=0.109,
    wspace=0.545
    )

