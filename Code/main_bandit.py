import matplotlib   # noqa
matplotlib.use('Agg')  # noqa

import matplotlib.pyplot as plt
import numpy as np

from bandits_environments import BernoulliBandit
from bandits_solvers import Solver, EpsilonGreedy, UCB1, BayesianUCB, ThompsonSampling


def visualize_results(solvers, solver_names, figname):
    """
    Visualize results due to a specific multi-armed bandit solver
    :param solvers: solver used for the environment
    :param solver_names: name of the solver
    :param figname: name of the figure to be plotted
    :return:
    """
    # DEBUG OPTIONS
    # assert len(solvers) == len(solver_names)
    # assert all(map(lambda solver: isinstance(solver, Solver), solvers))
    # assert all(map(lambda solver: len(solver.regrets) > 0, solvers))

    b = solvers[0].bandit   # extract solver

    fig = plt.figure(figsize=(14, 4))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Plot regrets over time
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)

    # Show probabilities estimated by each solver.
    sorted_indices = sorted(range(b.n), key=lambda idx: b.probas[idx])  # sort based on probas
    ax2.plot(range(b.n), [b.probas[x] for x in sorted_indices], 'k--', markersize=12)
    for s in solvers:
        ax2.plot(range(b.n), [s.estimated_probas[x] for x in sorted_indices], 'x', markeredgewidth=2)
    ax2.set_xlabel('Actions sorted by ' + r'$\theta$')
    ax2.set_ylabel('Estimated')
    ax2.grid('k', ls='--', alpha=0.3)

    # Show action counts
    for s in solvers:
        ax3.plot(range(b.n), np.array(s.counts) / float(len(solvers[0].regrets)), ls='steps', lw=2)
    ax3.set_xlabel('Actions')
    ax3.set_ylabel('Frac. # trials')
    ax3.grid('k', ls='--', alpha=0.3)

    plt.savefig(figname)


def run_experiment(K, N):
    """
    Run an experiment on solving a Bernoulli bandit
    :param K:  Number of bandits
    :param N: Sampling budget
    :return:
    """

    b = BernoulliBandit(K)
    print "The Bernoulli bandit instance has reward probabilities:\n", b.probas
    print "The best machine has index: {} and proba: {}".format(
        max(range(K), key=lambda i: b.probas[i]), max(b.probas))

    test_solvers = [
        EpsilonGreedy(b, 0),
        EpsilonGreedy(b, 1),
        EpsilonGreedy(b, 0.01),
        UCB1(b),
        BayesianUCB(b, 3, 1, 1),
        ThompsonSampling(b, 1, 1)
    ]
    solvers_names = [
        'Full-exploitation',
        'Full-exploration',
        r'$\epsilon$' + '-Greedy',
        'UCB1',
        'Bayesian UCB',
        'Thompson Sampling'
    ]

    for s in test_solvers:
        s.run(N)

    visualize_results(test_solvers, solvers_names, "results_K{}_N{}.png".format(K, N))


if __name__ == '__main__':
    run_experiment(10, 10000)
