import random

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

# Implements the On-policy first-visit MC control (for e-soft policies) on
# page 101 of Sutton & Barto 
class Easy21MC(Easy21TabularAgent):
    def run(self, environment, num_episodes, n_zero=100):
        # Easy 21 specific initialization
        num_state_visits = {state : 0 for state in self.get_all_states()}
        # Must initialize to 1 to avoid division by zero? Not true if
        # incrementing before updating
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

                    # Update estimate
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


