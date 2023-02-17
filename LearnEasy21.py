import argparse
from Easy21Environment import *
from Easy21Agents import *

def run_mc(seed, num_episodes, n_zero):
    mc = Easy21MC(seed)
    easy = Easy21Environment(seed)
    mc.learn(easy, num_episodes, n_zero)
    fig = mc.plot_value_function()
    fig.savefig(f"mc_value_func_{num_episodes}_episodes_seed_{seed}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn the Easy21 game")
    parser.add_argument("--part", choices=['2','3','4',"all"], default="all",
        help="""Part of the assignment to run - 2 for MC, 3 for TD, 4 for
        Function Approximation or "all" for all parts""")
    parser.add_argument("--num_episodes", type=int, default=1000000,
        help="Number of episodes for learning")
    parser.add_argument("--n_zero", type=int, default=100,
        help="Constant for influencing e-greedy exploration policy evolution")
    parser.add_argument("--seed", type=int, default=None,
        help="Integer seed for reproducible agents and environments")
    args = parser.parse_args()

    if args.part == "all":
        run_mc(args.seed, args.num_episodes, args.n_zero)
    elif args.part == 2:
        run_mc(args.seed, args.num_episodes, args.n_zero)
    elif args.part == 3:
        print("Not implemented")
    elif args.part == 4:
        print("Not implemented")
