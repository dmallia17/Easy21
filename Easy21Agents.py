import math, random
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

        Return
        ------
            An action, either "hit" or "stick"
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
                    # Note that because of the estimate > best_value check,
                    # ties are always resolved in favor of the first action
                    # encountered with an equal estimate
                    # This is different from before, where the tie always
                    # resolved in favor of "stick"
                    epsilon = n_zero / (n_zero + num_state_visits[curr_state])
                    best_value, best_action = -math.inf, None
                    for act in self.actions:
                        estimate = self.q_estimates[(curr_state, act)]
                        if estimate > best_value:
                            best_value = estimate
                            best_action = act
                    for act in self.actions:
                        if act == best_action:
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
    learn(environment, num_episodes, td_lambda, n_zero=100, mc_comparison=None)
        Executes TD (Sarsa) control on an Easy21Environment for the
        prescribed number of episodes, utilizing the passed lambda value in
        controlling the eligibility traces and a constant that influences the
        e-greedy exploration policy evolution; if an MC agent is passed for
        comparison, it measures the MSE between the q-values of the MC
        agent and the TD agent at the end of each episode
    """

    def learn(self, environment, num_episodes, td_lambda, n_zero=100,
        mc_comparison=None):
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
        mc_comparison : Easy21MC
            Comparison MC agent to compare against for q-values

        Return
        ------
        A dictionary mapping episode (1-indexed) to MSE between self
        and passed MC comparison agent; empty if MC agent is omitted
        """

        # Easy21 assignment specific initialization
        num_state_visits = {state : 0 for state in self.get_all_states()}
        num_state_action_visits = self.get_tab_q_func()
        episode_mse = {}
        perform_comparison = mc_comparison is not None

        # Loop
        for episode in range(1,num_episodes+1):
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
                # Note that because of the estimate > best_value check,
                # ties are always resolved in favor of the first action
                # encountered with an equal estimate
                # This is different from before, where the tie always
                # resolved in favor of "stick"
                epsilon = n_zero / (n_zero + num_state_visits[curr_state])
                best_value, best_action = -math.inf, None
                for act in self.actions:
                    estimate = self.q_estimates[(curr_state, act)]
                    if estimate > best_value:
                        best_value = estimate
                        best_action = act
                for act in self.actions:
                    if act == best_action:
                        self.policy[curr_state][act] = \
                            1 - epsilon + (epsilon / 2)
                    else:
                        self.policy[curr_state][act] = epsilon / 2

                curr_state, curr_action = next_state, next_action

            if perform_comparison:
                episode_mse[episode] = compare_q_estimates(self, mc_comparison)

        return episode_mse


class Easy21Linear:
    """
    Implements the Sarsa(lambda) algorithm for linear function approximation
    (and binary features) as spelled out on page 305 of Sutton & Barto (2nd
    edition), and in David Silver's Lecture 6 slides. The Sutton & Barto
    example essentially avoids vectorized code, but the below heeds Silver's
    succinct description.

    Methods
    -------
    learn(environment, num_episodes, td_lambda, ex_p=0.05,
        step=0.01, mc_comparison=None)
        Executes TD (Sarsa) control with linear function approximation on an
        Easy21Environment for the prescribed number of episodes, utilizing the
        passed lambda value in controlling the eligibility traces, a constant
        (ex_p) to control e-greedy exploration and the passed step size for
        weight updates; if an MC agent is passed for comparison, it measures
        the MSE between the q-values of the MC agent and the TD agent at the
        end of each episode
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
        self.weights = np.zeros((36,1)) # Linear weights
        # Define the coarse coding scheme
        self.dealer_sums = (tuple(range(1,5)), tuple(range(4,8)),
                            tuple(range(7,11)))
        self.player_sums = (tuple(range(1,7)), tuple(range(4,10)),
                            tuple(range(7,13)), tuple(range(10,16)),
                            tuple(range(13,19)), tuple(range(16,22)))

    def get_state_act_vec(self, state_act):
        """
        Converts a state-action pair into a feature (row) vector heeding the
        coarse coding scheme defined above.

        Parameters
        ----------
        state_act : tuple
            A ((dealer's sum, player's sum), action) tuple

        Return
        ------
        A numpy feature (row) vector for the state-action pair
        """

        dealer_sum, player_sum = state_act[0]
        act = state_act[1]

        out = []
        for ds in self.dealer_sums:
            for ps in self.player_sums:
                for a in self.actions:
                    out.append(int((dealer_sum in ds) and \
                                   (player_sum in ps) and (act in a)))

        return np.array(out).reshape((1,36))

    def predict(self, state_act):
        """
        Executes linear regression - predicting a q-value for a given
        state-action pair.

        Parameters
        ----------
        state_act : tuple
            A ((dealer's sum, player's sum), action) tuple

        Return
        ------
        The predicted q-value (numpy array of size (1,1))
        """

        return np.matmul(self.get_state_act_vec(state_act), self.weights)

    def get_action(self, state, ex_p=0.05):
        """
        Retrieve an action for the passed state, heeding the epsilon value
        (ex_p) for controlling greedy behavior. This agent does not explicitly
        maintain a policy as with the tabular agents (as we are not using a
        tabular representation); instead it directly applies the e-greedy rule
        of behaving randomly with probability ex_p, else greedily.

        Parameters
        ----------
        state : tuple
            A (dealer's sum, player's sum) tuple
        ex_p : float
            The epsilon value for e-greedy behavior

        Return
        ------
        An action, either "hit" or "stick"
        """

        if self.rng.uniform(0,1) < ex_p:
            return self.rng.choice(self.actions)
        else:
            best_value, best_action = -math.inf, None
            for act in self.actions:
                estimate = self.predict((state, act))
                if estimate > best_value:
                    best_value = estimate
                    best_action = act
            return best_action

    def learn(self, environment, num_episodes, td_lambda, ex_p=0.05,
        step=0.01, mc_comparison=None):
        """
        Implements the Sarsa(lambda) learning algorithm for linear function
        approximation

        Parameters
        ----------
        environment : Easy21Environment
            An Easy21Environment instance
        num_episodes : int
            The number of episodes to generate
        td_lambda : float
            The constant used to update eligibility traces over an episode
        ex_p : float
            The constant used to control e-greedy exploration
            policy evolution
        step : float
            Step size for updates
        mc_comparison : Easy21MC
            Comparison MC agent to compare against for q-values

        Return
        ------
        A dictionary mapping episode (1-indexed) to MSE between self
        and passed MC comparison agent; empty if MC agent is omitted
        """

        # Easy21 assignment specific initialization
        episode_mse = {}
        perform_comparison = mc_comparison is not None

        # Loop
        for episode in range(1,num_episodes+1):
            # Here traces is a vector
            traces = np.zeros((1,36))
            curr_state = environment.get_start()
            curr_action = self.get_action(curr_state, ex_p)
            while curr_state != "terminal":
                # Take action
                reward, next_state = environment.step(curr_state, curr_action)

                if next_state == "terminal":
                    next_action = None
                    # Q("terminal",a) is 0 for all a (i.e. stick or hit)
                    q_sp_ap = 0
                else:
                    next_action = self.get_action(next_state)
                    q_sp_ap = self.predict((next_state, next_action))

                # Calculate error
                error = reward + q_sp_ap - \
                    self.predict((curr_state, curr_action))

                # Update traces
                traces += self.get_state_act_vec((curr_state, curr_action))

                # Update weights
                self.weights += (step * error * traces.T)

                # Decay traces
                traces *= td_lambda

                curr_state, curr_action = next_state, next_action

            if perform_comparison:
                episode_mse[episode] = compare_q_estimates(self, mc_comparison)

        return episode_mse

    def plot_value_function(self):
        """
        Prepares a plot of the current state value function (i.e. for
        each state, the max over the values for choosing hit or stick). For
        this agent, no explicit representation of the state value function is
        maintained, but it may be recovered by iterating over all possible
        states (which the agent does not "know" about elsewhere) and taking the
        max over the predicted q-values.

        Return
        ------
        A 3D Matplotlib figure
        """
        x,y,z = [], [], [] # Dealer cards, player cards, state values
        states = [(dealer_card, player_card)
            for dealer_card in range(1,11) for player_card in range(1,22)]
        for state in states:
            x.append(state[0])
            y.append(state[1])
            z.append(max(self.predict((state,"hit")),
                self.predict((state, "stick"))))

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


def compare_q_estimates(agent1, agent2):
    """
    Calculate the mean squared error (MSE) between the q estimates
    of two agents

    Parameters
    ----------
    agent1, agent2 : Easy21TabularAgent
        Instances of Easy21TabularAgent or descendants with learned
        estimates

    Return
    ------
    float, the MSE between agent1 and agent2's q estimates
    """

    curr_sum = 0.0
    n = 0
    for sa_pair in agent1.q_estimates.keys():
        curr_sum += \
            ((agent1.q_estimates[sa_pair] - agent2.q_estimates[sa_pair]) ** 2)
        n += 1
    return (curr_sum / n)
