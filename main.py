import torch
import random
import numpy as np
import torch.nn as nn
from agent import Actor
from agent import Critic
from agent import ActorCritic
from config import args
import gymnasium as gym
from gymnasium.spaces import Discrete


def make_env(gym_id, seed):
    def thunk():
        env = gym.make(gym_id)

        # TRY NOT MODIFY: seeds
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk()


if __name__ == "__main__":
    # TRY NOT MODIFY: seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args.env, args.seed)

    assert isinstance(
        env.action_space, Discrete
    ), "The experiment takes only discrete action space."

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # actor = Actor(
    #     obs_dim=obs_dim,
    #     act_dim=act_dim,
    #     hidden_sizes=[args.hidden_size],
    #     activation=nn.Tanh,
    #     output_std=args.output_actor_std,
    # )
    # critic = Critic(
    #     obs_dim=obs_dim,
    #     hidden_sizes=[args.hidden_size],
    #     activation=nn.Tanh,
    #     output_std=1.0,
    # )

    actor_critic = ActorCritic(
        observation_space=obs_dim,
        action_space=act_dim,
        output_std_vals=(args.output_actor_std, args.output_critic_std),
        hidden_sizes=(args.hidden_size, args.hidden_size),
        activation=nn.Tanh,
    )

    print(actor_critic)
