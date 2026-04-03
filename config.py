import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="The ID of gym environment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=29,
        help="The random seed for the current experiment.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="The learning rate of an agent.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="The hidden size for the neural networks.",
    )
    parser.add_argument(
        "--output-actor-std",
        type=float,
        default=0.01,
        help="The standard deviation for output layer of actor network.",
    )
    parser.add_argument(
        "--output-critic-std",
        type=float,
        default=1.0,
        help="The standard deviation for output layer of critic network.",
    )

    args = parser.parse_args()
    return args


args = parse_args()
