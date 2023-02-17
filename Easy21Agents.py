import random

class Easy21TabularAgent:
    def __init__(self, seed=1234):
        self.actions = ("hit", "stick")
        self.rng = random.Random(seed)
        self.policy = self.get_start_policy()
        self.q_estimates = self.get_tab_q_func()

    def get_all_states(self):
        return [(dealer_card, player_card)
            for dealer_card in range(1,11) for player_card in range(1,22)]

    def get_tab_q_func(self, val=0):
        return {(state, action) : val 
            for state in self.get_all_states() for action in self.actions}

    def get_start_policy(self):
        return {state: {action : 0.5 for action in self.actions}
            for state in self.get_all_states()}

    def get_action(self, state):
        possibilities = self.policy[state]
        return self.rng.choices(
            list(possibilities.keys()), list(possibilities.values()))[0]

    def gen_episode(self, environment):
        current_state = environment.get_start()
        episode = []

        while current_state != "terminal":
            action = self.get_action(current_state)
            episode.append((current_state,action))
            reward, current_state = environment.step(current_state, action)
            episode.append(reward)

        return episode

