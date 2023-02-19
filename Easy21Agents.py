import random
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

class Easy21TabularAgent:
    """
    Implements an abstract tabular agent with functions to be inherited

    This class sets the groundwork for tabular learning; in particular,
    retrieving all non-terminal states, representing the q function as
    a dictionary mapping (state,action) pairs to values, representing
    policies as dictionaries mapping states to "sub-dictionaries" that
    in turn map actions to probabilities (not the most succinct
    representation, but a simple, intuitive one). Furthermore, getting
    an action in line with the current policy and using this ability to
    then generate whole episodes (useful for MC learning) is
    implemented.

    Methods
    -------
    get_all_states()
        Returns a list of all non-terminal states
    get_tab_q_func(val=0)
        Instantiates a q function dictionary with the given value
    get_start_policy()
        Gets a policy dictionary with hit or stick equally likely
    get_action(state)
        Gets an action based on the passed state and current policy
    gen_episode(environment)
        Generates an episode using the passed environment and the
        current policy
    plot_value_function()
        Prepares a plot of the current state value function
    """
    def __init__(self, seed=None):
        """
        Parameters
        ----------
        seed : int
            A seed for the environment's RNG
        """
        self.actions = ("hit", "stick")
        self.rng = random.Random(seed)
        self.policy = self.get_start_policy()
        self.q_estimates = self.get_tab_q_func()

    def get_all_states(self):
        """Get a list of all non-terminal states - i.e. all possible
            (dealer's sum, player's sum) tuples"""
        return [(dealer_card, player_card)
            for dealer_card in range(1,11) for player_card in range(1,22)]

    def get_tab_q_func(self, val=0.0):
        """
        Get a tabular q function represented by a dictionary mapping
        (state, action) - (tuple,str) - pairs to the passed value

        Parameters
        ----------
        val : float
            An initial value for the q estimates
        """
        return {(state, action) : val 
            for state in self.get_all_states() for action in self.actions}

    def get_start_policy(self):
        """Get a start policy, represented by a dictionary mapping
        states to dictionaries mapping actions to probabilities, with
        stick or hit equally likely in all states"""
        return {state: {action : 0.5 for action in self.actions}
            for state in self.get_all_states()}

    def get_action(self, state):
        """
        Gets an action ("stick" or "hit") based on the current policy

        Parameters
        ----------
        state : tuple
            A (dealer's sum, player's sum) tuple
        """
        possibilities = self.policy[state]
        return self.rng.choices(
            list(possibilities.keys()), list(possibilities.values()))[0]

    def gen_episode(self, environment):
        """
        Returns an episode of Easy21 played out using the current
        policy, represented as a list of (state, action) tuples followed by
        rewards. 

        Parameters
        ----------
        environment : Easy21Environment
            An Easy21Environment instance
        """
        current_state = environment.get_start()
        episode = []

        # Does not append "terminal" to the episode as this is not useful;
        # keeps (state, action) tuples instead of unpacked values as these
        # are likely to be used for indexing into the q function
        while current_state != "terminal":
            action = self.get_action(current_state)
            episode.append((current_state,action))
            reward, current_state = environment.step(current_state, action)
            episode.append(reward)

        return episode

    def plot_value_function(self):
        """
        Prepares a plot of the current state value function (i.e. for
        each state, the max over the values for choosing hit or stick)

        Return
        ------
        A 3D Matplotlib figure
        """
        x,y,z = [], [], [] # Dealer cards, player cards, state values

        for state in self.get_all_states():
            x.append(state[0])
            y.append(state[1])
            z.append(max(self.q_estimates[(state,"hit")],
                self.q_estimates[(state, "stick")]))

        # Set up plot
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.set_xlabel("Dealer card")
        ax.set_ylabel("Player card")
        ax.set_zlabel("State value")

        ax.plot_surface(
            np.reshape(x,(10,21)), np.reshape(y,(10,21)),
            np.reshape(z,(10,21)), cmap="viridis")

        return fig

class Easy21MC(Easy21TabularAgent):
    """
    Implements the On-policy first-visit MC control (for e-soft
    policies) on page 101 of Sutton & Barto (2nd edition). Inherits
    from Easy21TabularAgent for tabular functionality.

    Methods
    -------
    learn(environment, num_episodes, n_zero=100)
        Executes Monte Carlo control on an Easy21Environment for the
        prescribed number of episodes, utilizing a constant that
        influences the e-greedy exploration policy evolution
    """

    def learn(self, environment, num_episodes, n_zero=100):
        """
        Implements Monte Carlo control, updating the agent's policy and
        q estimates

        Parameters
        ----------
        environment : Easy21Environment
            An Easy21Environment instance
        num_episodes : int
            The number of episodes to generate
        n_zero : int
            The constant used to influence e-greedy exploration
            policy evolution
        """

        # Easy21 assignment specific initialization
        num_state_visits = {state : 0 for state in self.get_all_states()}
        num_state_action_visits = self.get_tab_q_func() 

        # Loop
        for episode in range(num_episodes):
            curr_ep = self.gen_episode(environment)

            # There is no discounting and no intermediate rewards so the
            # return IS the last reward
            curr_ep_g = curr_ep[-1]
            curr_ep.pop()
            # Strip zero intermediate rewards
            curr_ep = [element
                for element in curr_ep if type(element) is not int]

            # Work backwards until whole episode has been processed
            while len(curr_ep) > 0:

                state_action_pair = curr_ep.pop()
                if state_action_pair not in curr_ep:
                    curr_state = state_action_pair[0]

                    # Update number of times state and state-action pair have
                    # been encountered
                    num_state_action_visits[state_action_pair] += 1
                    num_state_visits[curr_state] += 1

                    # Update estimate, using the "time-varying scalar
                    # step-size"
                    self.q_estimates[state_action_pair] = \
                        self.q_estimates[state_action_pair] + \
                        (1 / num_state_action_visits[state_action_pair]) * \
                        (curr_ep_g - self.q_estimates[state_action_pair])

                    # Update policy
                    hit_val = self.q_estimates[(curr_state,"hit")]
                    stick_val = self.q_estimates[(curr_state,"stick")]
                    a_star = "hit" if hit_val > stick_val else "stick"
                    epsilon = n_zero / (n_zero + num_state_visits[curr_state])
                    for act in self.actions:
                        if act == a_star:
                            self.policy[curr_state][act] = \
                                1 - epsilon + (epsilon / 2)
                        else:
                            self.policy[curr_state][act] = epsilon / 2


class Easy21TD(Easy21TabularAgent):
    """
    Implements the Sarsa(lambda) algorithm as described on slide 29 of David
    Silver's "Lecture 5: Model-Free Control" slides, with the adjustments
    needed to maintain an evolving e-greedy exploration and time-dependent
    update step-size.

    Methods
    -------
    learn(environment, num_episodes, td_lambda, n_zero=100)
        Executes TD (Sarsa) control on an Easy21Environment for the
        prescribed number of episodes, utilizing the passed lambda value in
        controlling the eligibility traces and a constant that influences the
        e-greedy exploration policy evolution
    """

    def learn(self, environment, num_episodes, td_lambda, n_zero=100):
        """
        Implements TD (Sarsa) control, updating the agent's policy and
        q estimates

        Parameters
        ----------
        environment : Easy21Environment
            An Easy21Environment instance
        num_episodes : int
            The number of episodes to generate
        td_lambda : float
            The constant used to update eligibility traces over an episode
        n_zero : int
            The constant used to influence e-greedy exploration
            policy evolution
        """

        # Easy21 assignment specific initialization
        num_state_visits = {state : 0 for state in self.get_all_states()}
        num_state_action_visits = self.get_tab_q_func()

        # Loop
        for episode in range(num_episodes):
            traces = self.get_tab_q_func() # Re-initialize traces
            curr_state = environment.get_start()
            curr_action = self.get_action(curr_state)
            while curr_state != "terminal":
                # Update number of times state and state-action pair have
                # been encountered
                num_state_action_visits[(curr_state, curr_action)] += 1
                num_state_visits[curr_state] += 1

                # Take action
                reward, next_state = environment.step(curr_state, curr_action)

                if next_state == "terminal":
                    next_action = None
                    # Q("terminal",a) is 0 for all a (i.e. stick or hit)
                    q_sp_ap = 0
                else:
                    next_action = self.get_action(next_state)
                    q_sp_ap = self.q_estimates[(next_state, next_action)]

                # Calculate update
                error = reward + q_sp_ap - \
                    self.q_estimates[(curr_state, curr_action)]

                traces[(curr_state, curr_action)] += 1

                # Update estimates and traces
                for sa_pair in list(self.q_estimates.keys()):
                    cred_asst = traces[sa_pair]
                    if cred_asst: # i.e. > 0, avoids division by zero
                        self.q_estimates[sa_pair] += \
                            (1 / num_state_action_visits[sa_pair]) * \
                                error * cred_asst

                        traces[sa_pair] *= td_lambda

                # Update policy
                hit_val = self.q_estimates[(curr_state,"hit")]
                stick_val = self.q_estimates[(curr_state,"stick")]
                a_star = "hit" if hit_val > stick_val else "stick"
                epsilon = n_zero / (n_zero + num_state_visits[curr_state])
                for act in self.actions:
                    if act == a_star:
                        self.policy[curr_state][act] = \
                            1 - epsilon + (epsilon / 2)
                    else:
                        self.policy[curr_state][act] = epsilon / 2

                curr_state, curr_action = next_state, next_action
