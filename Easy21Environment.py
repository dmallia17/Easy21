import random

class Easy21Environment:
    def __init__(self, seed=1234, verbose=False):
        self.rng = random.Random(seed)
        self.card_vals = tuple(range(1,11))
        self.card_colors = ("red","black")
        self.card_colors_weights = (1/3, 2/3)
        self.log = print if verbose else lambda x : None
        
    def _get_card_color(self):    
        return self.rng.choices(self.card_colors,
            weights=self.card_colors_weights)[0]
    
    def _get_card_value(self):
        return self.rng.choice(self.card_vals)
    
    def _get_multiplier(self):
        return 1 if self._get_card_color() == "black" else -1
    
    def get_start(self):
        return (self._get_card_value(),self._get_card_value())
        
    def step(self, state, action):
        self.log(f"Current state: {state}\nCurrent action: {action}")
            
        if action == "hit":
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
            
            # Always hit below 17
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