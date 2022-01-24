import sys
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

class A2C(tf.keras.Model):
    def __init__(self, action_size):
        super(A2C, self).__init__()
        self.actor_fc = Dense(24, activation='relu')
        self.actor_out = Dense(action_size, activation='softmax', kernel_initializer=RandomUniform(-1e-3, 1e-3))
        self.critic_fc1 = Dense(24, activation='tanh')
        self.critic_fc2 = Dense(24, activation='tanh')
        self.critic_out = Dense(1, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        actor_x = self.actor_fc(x)
        policy = self.actor_out(actor_x)

        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)
        return policy, value

class A2CAgent:
    def __init__(self, action_size):
        self.render = True

        self.action_size = action_size

        # Hyper Parameter
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = A2C(self.action_size)
        self.optimizer = Adam(lr=self.learning_rate, clipnorm=5.0)

    # policy
    def get_action(self, state):
        policy, _ = self.model(state)
        policy = np.array(policy[0])
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # update
    def train_model(self, state, action, reward, next_state, done):
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            policy, value = self.model(state)
            _, next_value = self.model(next_state)
            target = reward + (1 - done) * self.discount_factor * next_value[0]

            one_hot_action = tf.one_hot([action], self.action_size)
            action_prob = tf.reduce_sum(one_hot_action * policy, axis=1)
            cross_entropy = -tf.math.log(action_prob + 1e-5)
            advantage = tf.stop_gradient(target - value[0])
            actor_loss = tf.reduce_mean(cross_entropy * advantage)

            critic_loss = 0.5 * tf.square(tf.stop_gradient(target) - value[0])
            critic_loss = tf.reduce_mean(critic_loss)

            loss = 0.2 * actor_loss + critic_loss

        # minimize loss
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return np.array(loss)

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2CAgent(action_size)

    scores, episodes = [], []
    score_avg = 0

    EPISODES = 1000
    for e in range(EPISODES):
        done = False
        score = 0
        loss_list = []
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            reward = 0.1 if not done or score == 500 else -1

            loss = agent.train_model(state, action, reward, next_state, done)
            loss_list.append(loss)
            state = next_state

            if done:
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score avg: {:3.2f} | loss: {:.3f}".format(e, score_avg, np.mean(loss_list)))

                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./save_graph/graph.png")

                if score_avg > 400:
                    agent.model.save_weights('save_model/model', save_format='tf')
                    sys.exit()