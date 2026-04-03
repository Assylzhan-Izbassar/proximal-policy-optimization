import torch
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(sizes, activation=nn.Tanh, output_std=1.0, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        std = np.sqrt(2) if j < len(sizes) - 2 else output_std
        layers += [layer_init(nn.Linear(sizes[j], sizes[j + 1]), std), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_std):
        super().__init__()
        self.logits_net = mlp(
            sizes=[obs_dim] + list(hidden_sizes) + [act_dim],
            activation=activation,
            output_std=output_std,
        )

    def _distribution(self, obs):
        logits = self.logit_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        """Produce action dist-s for given obser-ns, and
        optionally compute the log-likelihood of given actions
        under those distributions."""
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation, output_std):
        super().__init__()
        self.v_net = mlp(
            sizes=[obs_dim] + list(hidden_sizes) + [1],
            activation=activation,
            output_std=output_std,
        )

    def forward(self, obs):
        return self.v_net(obs).squeeze(-1)  # critical to ensure v has right shape.


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        output_std_vals=(0.01, 1.0),
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
    ):
        super().__init__()

        # build policy function
        self.pi = Actor(
            obs_dim=observation_space,
            act_dim=action_space,
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_std=output_std_vals[0],
        )

        # build value function
        self.v = Critic(
            obs_dim=observation_space,
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_std=output_std_vals[1],
        )

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
