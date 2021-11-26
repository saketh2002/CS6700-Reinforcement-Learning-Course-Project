import numpy

from config import *
import time
import random

"""

DO not modify the structure of "class Agent".
Implement the functions of this class.
Look at the file run.py to understand how evaluations are done. 

There are two phases of evaluation:
- Training Phase
The methods "registered_reset_train" and "compute_action_train" are invoked here. 
Complete these functions to train your agent and save the state.

- Test Phase
The methods "registered_reset_test" and "compute_action_test" are invoked here. 
The final scoring is based on your agent's performance in this phase. 
Use the state saved in train phase here. 

"""


## order in configs: lr=0.9, discount rate=0.8 , eps=1, decay rate=0.005

class Agent:
    def __init__(self, env):

        self.env_name = env

        if self.env_name == 'acrobot':
            self.q_acro = numpy.random.uniform(high=-3, low=0, size=(15, 15, 15, 15, 15, 15, 3))
            self.config = config[self.env_name]
            self.episode_counts = 0
            self.curr_obs = None
            self.avail_acro = [0, 1, 2]
            self.lr = 0.6
            self.disc_fact = 0.98
            self.eps = 0.75
            self.eps_decay_rate = 0.0078
            self.obs_space_low = numpy.array([-1.0, -1.0, -1.0, -1.0, -12.566, -28.274])
            self.bucket_size = numpy.array([0.13333333, 0.13333333, 0.13333333, 0.13333333, 1.67551613, 3.76991119])

        if self.env_name == 'taxi':
            self.q_taxi = numpy.zeros((500, 6))
            self.config = config[self.env_name]
            self.episode_counts = 0
            self.curr_obs = None
            self.avail_taxi = [0, 1, 2, 3, 4, 5]
            self.lr = 1
            self.disc_r = 0.8
            self.eps = 0.99
            self.decay_r = 0.005

    def register_reset_train(self, obs):

        if self.env_name == 'acrobot':
            action = random.choice(self.avail_acro)
            self.curr_obs = tuple(
                numpy.clip(((obs - self.obs_space_low) / self.bucket_size).astype(numpy.int), None, 14))
            return action
        if self.env_name == 'taxi':
            self.curr_obs = obs
            action = random.choice(self.avail_taxi)
            return action

    def compute_action_train(self, obs, reward, done, info):

        if self.env_name == 'acrobot':

            discrete_obs = tuple(
                numpy.clip(((obs - self.obs_space_low) / self.bucket_size).astype(numpy.int), None, 14))

            if numpy.random.uniform(0, 1) < self.eps:
                action = random.choice(self.avail_acro)
            else:
                # get action based on current q_table
                action = numpy.argmax(self.q_acro[discrete_obs])

            self.q_acro[self.curr_obs][action] = self.q_acro[self.curr_obs][action] + self.lr * (
                    reward + self.disc_fact * numpy.max(self.q_acro[discrete_obs]) - self.q_acro[self.curr_obs][action])

            self.curr_obs = discrete_obs

            if done:
                self.episode_counts += 1
                #     self.eps_end_decay = (3 * self.episode_counts) // 4
                #    self.eps_decay_rate = self.eps / (self.eps_end_decay - self.eps_start_decay)
                if 1 <= self.episode_counts < 2000:
                    self.eps = numpy.exp(-self.eps_decay_rate * self.episode_counts)
            return action

        if self.env_name == 'taxi':

            if random.uniform(0, 1) < self.eps:
                # explore
                action = random.choice(self.avail_taxi)
            else:
                # exploit
                action = numpy.argmax(self.q_taxi[obs, :])

                # take action and observe reward
            #    new_obs, reward, done, info = env.step(action)

            # Q-learning algorithm self.q_taxi[obs,action] = self.q_taxi[obs,action] + learning_rate * (reward +
            # discount_rate * np.max(self.q_taxi[new_obs,:])-self.q_taxi[obs,action])
            self.q_taxi[self.curr_obs, action] = self.q_taxi[self.curr_obs, action] + self.lr * (
                    reward + (self.disc_r * numpy.max(self.q_taxi[obs, :])) - self.q_taxi[self.curr_obs, action])

            # Update to our new state
            #    obs = new_obs
            self.curr_obs = obs
            # Decrease epsilon
            # epsilon = np.exp(-decay_rate*episode)
            if done:
                self.episode_counts += 1
                self.eps = numpy.exp(-self.decay_r * self.episode_counts)
                self.lr = numpy.exp(-0.005*self.episode_counts)
            return action

    def register_reset_test(self, obs):

        if self.env_name == 'acrobot':
            action = random.choice(self.avail_acro)
            return action
        if self.env_name == 'taxi':
            action = numpy.argmax(self.q_taxi[obs, :])
            return action

    def compute_action_test(self, obs, reward, done, info):

        if self.env_name == 'acrobot':
            discrete_obs = tuple(
                numpy.clip(((obs - self.obs_space_low) / self.bucket_size).astype(numpy.int), None, 14))
            action = numpy.argmax(self.q_acro[discrete_obs])
            return action
        if self.env_name == 'taxi':
            action = numpy.argmax(self.q_taxi[obs, :])
            return action

