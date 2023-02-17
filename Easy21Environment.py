import random


class Easy21Environment:
    """
    Implements the Easy21 environment for reinforcement learning

    Defines the Easy21 environment along the lines delineated in the
    assignment: primarily this makes a "step" function available which
    accepts a state and an action, and returns a reward and subsequent
    state. To facilitate learning, this class also provides a
    "get_start" function for beginning an episode.

    States are represented as simple (dealer's sum, player's sum)
    tuples OR the string "terminal", indicating end of an episode;
    expected actions are strings, "hit" or "stick".
    """

    def __init__(self, seed=None, verbose=False):
        """
        Parameters
        ----------
        seed : int
            A seed for the environment's RNG
        verbose : bool
            Flag specifying if step should print what happens
        """

        self.rng = random.Random(seed)
        self.card_vals = tuple(range(1,11))
        # Directly represent colors red and black as -1 and 1, respectively, as
        # these can be used as the multiplier on a card's value when updating
        # the player or dealer's sums after drawing a card
        self.card_colors = (-1,1)
        self.card_colors_weights = (1/3, 2/3)
        # Simple hack for cleaner logging
        self.log = print if verbose else lambda x : None

    def _get_card_value(self):
        """'Private' function to get a random card value (1-10)"""
        return self.rng.choice(self.card_vals)

    def _get_multiplier(self):
        """'Private' function to get the card 'color'"""
        # Choices returns a list, must extract the choice with [0]
        return self.rng.choices(self.card_colors,
            weights=self.card_colors_weights)[0]

    def get_start(self):
        """Get an initial state tuple - (dealer's sum, player's sum)."""
        return (self._get_card_value(),self._get_card_value())

    def step(self, state, action):
        """
        Implements the game dynamics as described in the assignment.

        Parameters
        ----------
        state : tuple
            A (dealer's sum, player's sum) tuple; should not be the
            'terminal' string.
        action: str
            Either 'hit' or 'stick'.

        Returns
        -------
        reward : int
            The reward for the action in the state (all intermediate
            rewards are 0).
        state : tuple or str
            Either a (dealer's sum, player's sum) tuple or 'terminal'
        """

        self.log(f"Current state: {state}\nCurrent action: {action}")

        if action == "hit":
            # Draw new card and update sum
            new_player_sum = state[1] + \
                self._get_multiplier() * self._get_card_value()

            self.log(f"New player sum: {new_player_sum}")

            if new_player_sum > 21 or new_player_sum < 1:
                self.log("Player went bust")
                return -1, "terminal"
            else:
                self.log("Game continues")
                return 0, (state[0], new_player_sum)
        else: # Stick - playout dealer
            dealer_sum, player_sum = state
            self.log("Dealer playout")

            # Follow dealer policy of always hit below 17
            # Need to also check the dealer does not go bust 
            while dealer_sum < 17 and dealer_sum >= 1:
                dealer_sum += self._get_multiplier() * self._get_card_value()
                self.log(f"Dealer hit, new sum: {dealer_sum}")

            # Dealer went bust, player win
            if dealer_sum > 21 or dealer_sum < 1:
                reward = 1
                self.log("Dealer went bust")
            elif dealer_sum < player_sum: # Player had higher sum
                reward = 1
                self.log("Win")
            elif dealer_sum == player_sum: # Equal - draw
                reward = 0
                self.log("Draw")
            else: # Dealer had a higher sum
                reward = -1
                self.log("Loss")

            return reward, "terminal"
