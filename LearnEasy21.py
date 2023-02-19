import argparse
import matplotlib.pyplot as plt
from Easy21Environment import *
from Easy21Agents import *

def run_mc(seed, num_episodes, n_zero):
    mc = Easy21MC(seed)
    easy = Easy21Environment(seed)
    mc.learn(easy, num_episodes, n_zero)
    fig = mc.plot_value_function()
    fig.savefig(f"mc_value_func_{num_episodes}_episodes_seed_{seed}.png")

def run_td(seed, num_episodes, n_zero):
    error_dict = {} # Keep errors by lambda value
    # Keep errors by lambda-episode value (i.e. maps lambda value to a
    # dictionary mapping episode to current MSE between TD results and MC
    # results)
    episode_mse_dict = {}
    index = 0

    # Run MC Control to approximate Q*(s,a)
    easy_compare = Easy21Environment(seed)
    mc_compare = Easy21MC(seed)
    mc_compare.learn(easy_compare, num_episodes, n_zero)

    # Evaluate over different values of lambda
    # Uses list comprehension for a float range
    for td_lambda in [0.1 * i for i in range(0,11)]:
        easy_compare = Easy21Environment(seed)
        td_compare = Easy21TD(seed)
        episode_mse_dict[index] = td_compare.learn(easy_compare, 1000,
            td_lambda, n_zero, mc_compare)
        error_dict[td_lambda] = compare_q_estimates(td_compare, mc_compare)
        index += 1

    # Prepare plot of MSE by lambda
    fig, ax = plt.subplots()
    ax.plot(list(error_dict.keys()), list(error_dict.values()))
    ax.set(xlabel="Lambda", ylabel="Mean Squared Error",
        title="MSE between TD and MC estimates, varying lambda")
    fig.savefig(f"td_mse_by_lambda_{num_episodes}_episodes_seed_{seed}.png")

    # Prepare plot of MSE by episode for lambda = 0 and lambda = 1
    fig, ax = plt.subplots()
    ax.plot(list(episode_mse_dict[0].keys()),
        list(episode_mse_dict[0].values()), color="blue", label="Lambda = 0")
    ax.plot(list(episode_mse_dict[index-1].keys()),
        list(episode_mse_dict[index-1].values()), color="red",
        label="Lambda = 1")
    ax.legend()
    ax.set(xlabel="Episode", ylabel="Mean Squared Error",
        title="MSE between TD and MC estimates, over episodes")
    fig.savefig(f"td_mse_by_episode_{num_episodes}_episodes_seed_{seed}.png")

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
        run_td(args.seed, args.num_episodes, args.n_zero)
    elif args.part == "2":
        run_mc(args.seed, args.num_episodes, args.n_zero)
    elif args.part == "3":
        run_td(args.seed, args.num_episodes, args.n_zero)
    elif args.part == "4":
        print("Not implemented")
