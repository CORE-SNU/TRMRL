import torch
from rlkit.torch.pytorch_util import randn, rand, zeros_like


def acceptance_probability(p1, p2):
    """
    Computation of the acceptance probability for simulating Markov Chain Monte Carlo.

    :param p1: last sample
    :param p2: current sample
    :return: acceptance probability
    """
    return torch.exp(torch.minimum(zeros_like(p1), p2 - p1))


def walk(z):
    """
    Given a current sample (or samples), generate the next sample according to some symmetric transition kernel.
    Here the transition probability is simply given as a gaussian distribution centered at the current sample.
    :param x: current sample (or samples)
    :return: newly generated sample (or samples)
    """
    sigma = 1.          # standard deviation of gaussian distribution

    return z + sigma * randn(*z.shape)


def markov_chain_monte_carlo(num_tasks: int, dim: int, prob_func: callable, num_samples=1000, num_steps=1000):
    # initial distribution
    # i.i.d. samples Z1, ..., ZN ~ N(0, I)
    # TODO: sample initial iterates from the high-probability region of the posterior (how? )
    z = randn(num_tasks, num_samples, dim)

    for _ in range(num_steps):
        p1 = prob_func(z)       # shape = (# tasks, # samples)
        z_next = walk(z)
        p2 = prob_func(z_next)
        a = acceptance_probability(p1, p2)
        coin = rand(num_tasks, num_samples)
        accept = torch.unsqueeze(coin < a, -1)
        z = torch.where(accept, z_next, z)

    return z