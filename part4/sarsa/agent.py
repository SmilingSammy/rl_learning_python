import numpy as np
import random
from collections import defaultdict
from environment import Env

class SARSAgent:
    def __init__(self, actions):
        self.actions = actions
        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = defaultdict(lambda : [0.0, 0.0, 0.0, 0.0])

    # s, a, r, s', a' 샘플로부터 Q함수 update
    def learn(self, state, action, reward, next_state, next_action):
        state, next_state = str(state), str(next_state)
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        td = reward + self.discount_factor * next_state_q - current_q
        new_q = current_q + self.step_size * td
        self.q_table[state][action] = new_q

    # 엡실론 탐용 정책에 따른 행동 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # Q함수에 따른 행동 반환
            state = str(state)
            q_list = self.q_table[state]
            action = arg_max(q_list)
        return action

# Q함수의 값에 따른 최적의 행동 반환
def arg_max(q_list):
    max_idx_list = np.argwhere(q_list == np.amax(q_list))
    max_idx_list = max_idx_list.flatten().tolist()
    return random.choice(max_idx_list)


if __name__ == "__main__":
    env = Env()
    agent = SARSAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        # 게임 환경과 상태 초기화
        state = env.reset()
        # 현재 상태에 대한 행동 선택
        action = agent.get_action(state)

        while True:
            env.render()
            # 행동에 따른 다음 상태의 보상 및 에피소드의 종료 여부 확인
            next_state, reward, done = env.step(action)
            # 다음 상태에서의 다음 행동 선택
            next_action = agent.get_action(next_state)
            # Q함수 update
            agent.learn(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action

            env.print_value_all(agent.q_table)

            if done:
                break