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

def run_td(seed, num_episodes, n_zero=100, approx=False, show=False):
    error_dict = {} # Keep errors by lambda value
    # Keep errors by lambda-episode value (i.e. maps lambda value to a
    # dictionary mapping episode to current MSE between TD results and MC
    # results)
    episode_mse_dict = {}
    index = 0
    td_agent = Easy21Linear if approx else Easy21TD
    title_str = "(Approx.)" if approx else "(Tabular)"
    file_str = "approx" if approx else "tabular"

    # Run MC Control to approximate Q*(s,a)
    easy_compare = Easy21Environment(seed)
    mc_compare = Easy21MC(seed)
    mc_compare.learn(easy_compare, num_episodes, n_zero)

    td_args = { "num_episodes" : 1000,
                "mc_comparison" : mc_compare}

    if not approx:
        td_args["n_zero"] = n_zero

    # Evaluate over different values of lambda
    # Uses list comprehension for a float range
    for td_lambda in [0.1 * i for i in range(0,11)]:
        easy_compare = Easy21Environment(seed)
        td_compare = td_agent(seed)
        td_args["environment"] = easy_compare
        td_args["td_lambda"] = td_lambda
        episode_mse_dict[index] = td_compare.learn(**td_args)
        error_dict[td_lambda] = compare_q_estimates(td_compare, mc_compare)
        index += 1

    # Prepare plot of MSE by lambda
    fig, ax = plt.subplots()
    ax.plot(list(error_dict.keys()), list(error_dict.values()))
    ax.set(xlabel="Lambda", ylabel="Mean Squared Error",
        title=f"MSE between TD {title_str} and MC estimates, varying lambda")
    if show:
        plt.show()
    else:
        fig.savefig(
            f"td_{file_str}" + \
                f"_mse_by_lambda_{num_episodes}_episodes_seed_{seed}.png")

    # Prepare plot of MSE by episode for lambda = 0 and lambda = 1
    fig, ax = plt.subplots()
    ax.plot(list(episode_mse_dict[0].keys()),
        list(episode_mse_dict[0].values()), color="blue", label="Lambda = 0")
    ax.plot(list(episode_mse_dict[index-1].keys()),
        list(episode_mse_dict[index-1].values()), color="red",
        label="Lambda = 1")
    ax.legend()
    ax.set(xlabel="Episode", ylabel="Mean Squared Error",
        title=f"MSE between TD {title_str} and MC estimates, over episodes")
    if show:
        plt.show()
    else:
        fig.savefig(
            f"td_{file_str}" + \
                f"_mse_by_episode_{num_episodes}_episodes_seed_{seed}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn the Easy21 game")
    parser.add_argument("--part", choices=['2','3','4',"all"], default="all",
        help="""Part of the assignment to run - 2 for MC, 3 for TD, 4 for
        Function Approximation or "all" for all parts""")
    parser.add_argument("--num_episodes", type=int, default=1000000,
        help="Number of episodes for learning")
    parser.add_argument("--n_zero", type=int, default=100,
        help="Constant for influencing e-greedy exploration policy evolution")
    parser.add_argument("--seed", type=int, default=1234,
        help="Integer seed for reproducible agents and environments")
    args = parser.parse_args()

    if args.part == "all":
        run_mc(args.seed, args.num_episodes, args.n_zero)
        run_td(args.seed, args.num_episodes, args.n_zero)
        run_td(args.seed, args.num_episodes, approx=True)
    elif args.part == "2":
        run_mc(args.seed, args.num_episodes, args.n_zero)
    elif args.part == "3":
        run_td(args.seed, args.num_episodes, args.n_zero)
    elif args.part == "4":
        run_td(args.seed, args.num_episodes, approx=True)
