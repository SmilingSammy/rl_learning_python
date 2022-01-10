import numpy as np
from environment import GraphicDisplay, Env

class ValueIteration:
    def __init__(self, env):
        # environment
        self.env = env
        # value function
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        # discount rate
        self.discount_factor = 0.9

    # Calculate next value function
    def value_iteration(self):
        # next value function initialization
        next_value_table = [[0.00] * self.env.width for _ in range(self.env.height)]

        for state in self.env.get_all_states():
            # 종료지점
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue

            # Bellman Optimality Equation 계산
            value_list = []
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append(reward + self.discount_factor * next_value)

            # next value --> maximum value
            next_value_table[state[0]][state[1]] = max(value_list)

        self.value_table = next_value_table

    # 정책에 따른 행동 반환
    def get_action(self, state):
        if state == [2, 2]:
            return []

        value_list = []
        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = reward + self.discount_factor * next_value
            value_list.append(value)

        max_idx_list = np.argwhere(value_list == np.amax(value_list))
        action_list = max_idx_list.flatten().tolist()
        return action_list

    # 가치함수 값
    def get_value(self, state):
        return self.value_table[state[0]][state[1]]


if __name__ == "__main__":
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()
