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
            self.q_acro = numpy.random.uniform(high=-3, low=0, size=(16, 16, 16, 16, 16, 16, 3))
            self.config = config[self.env_name]
            self.episode_counts = 0
            self.curr_obs = None
            self.avail_acro = [0, 1, 2]
            self.lr = 0.2
            self.disc_fact = 1
            self.eps = 1
            self.eps_decay_rate = 0.0001
            self.obs_space_low = numpy.array([-1.0, -1.0, -1.0, -1.0, -12.566, -28.274])
            self.bucket_size = numpy.array([0.125, 0.125, 0.125, 0.125, 1.57079637, 3.53429174])
            self.curr_obs = None
            self.act = None

        if self.env_name == 'taxi':
            self.q_taxi = numpy.zeros((500, 6))
            self.config = config[self.env_name]
            self.episode_counts = 0
            self.curr_obs = None
            self.avail_taxi = [0, 1, 2, 3, 4, 5]
            self.lr = 0.8
            self.disc_r = 0.8
            self.eps = 1
            self.decay_r = 0.0001
            self.act = None

        if self.env_name == 'kbca':
            self.q_kbca = numpy.zeros((50, 2))
            self.config = config[self.env_name]
            self.episode_counts = 0
            self.curr_obs = None
            self.avail_kbca = [0, 1]
            self.lr = 0.9
            self.disc_r = 0.8
            self.eps = 1
            self.decay_r = 0.0002
            self.act = None

        if self.env_name == 'kbcb':
            self.q_kbcb = numpy.zeros((50, 2))
            self.config = config[self.env_name]
            self.episode_counts = 0
            self.curr_obs = None
            self.avail_kbcb = [0, 1]
            self.lr = 0.9
            self.disc_r = 0.8
            self.eps = 1.0
            self.decay_r = 0.00005
            self.act = None

        if self.env_name == 'kbcc':
            self.q_kbcc = numpy.zeros((50, 3))
            self.config = config[self.env_name]
            self.episode_counts = 0
            self.curr_obs = None
            self.avail_kbcc = [0, 1, 2]
            self.lr = 0.9
            self.disc_r = 0.8
            self.eps = 1.0
            self.decay_r = 0.00005
            self.act = None

    def register_reset_train(self, obs):

        if self.env_name == 'acrobot':
            action = random.choice([0, 2])
            self.curr_obs = tuple(
                numpy.clip(((obs - self.obs_space_low) / self.bucket_size).astype(numpy.int), None, 15))
            self.act = action

        if self.env_name == 'taxi':
            self.curr_obs = obs
            action = random.choice(self.avail_taxi)
            self.act = action

        if self.env_name == 'kbca':
            obs_convt = [-2 if x == "" else x for x in obs]
            self.curr_obs = sum(obs_convt)
            action = 1
            self.act = action

        if self.env_name == 'kbcb':
            obs_convt = [-2 if x == "" else x for x in obs]
            self.curr_obs = sum(obs_convt)
            action = 1
            self.act = action

        if self.env_name == 'kbcc':
            obs_convt = [-2 if x == "" else x for x in obs]
            self.curr_obs = sum(obs_convt)
            action = 1
            self.act = action

        return action

    def compute_action_train(self, obs, reward, done, info):

        if self.env_name == 'acrobot':

            discrete_obs = tuple(
                numpy.clip(((obs - self.obs_space_low) / self.bucket_size).astype(numpy.int), None, 15))

            self.q_acro[self.curr_obs][self.act] = self.q_acro[self.curr_obs][self.act] + self.lr * (
                    reward + self.disc_fact * numpy.max(self.q_acro[discrete_obs]) - self.q_acro[self.curr_obs][self.act])

            if random.uniform(0, 1) < self.eps and obs[1] <= 0 and obs[4] < 0:
                action = 2

            elif random.uniform(0, 1) < self.eps and obs[1] <= 0 and obs[4] >= 0:
                action = 1

            elif random.uniform(0, 1) < self.eps and obs[1] > 0 and obs[4] >= 0:
                action = 0

            elif random.uniform(0, 1) < self.eps and obs[1] > 0 and obs[4] < 0:
                action = 2

            else:
                # get action based on current q_table
                action = numpy.argmax(self.q_acro[discrete_obs])

            self.curr_obs = discrete_obs
            self.act = action

            if done:
                self.episode_counts += 1
                #     self.eps_end_decay = (3 * self.episode_counts) // 4
                #     self.eps_decay_rate = self.eps / (self.eps_end_decay - self.eps_start_decay)
                if 1 <= self.episode_counts < 2000:
                    self.eps -= self.eps_decay_rate

            return action

        if self.env_name == 'taxi':

            self.q_taxi[self.curr_obs, self.act] = self.q_taxi[self.curr_obs, self.act] + self.lr * (
                    reward + self.disc_r * numpy.max(self.q_taxi[obs]) - self.q_taxi[self.curr_obs, self.act])

            if random.uniform(0, 1) < self.eps:
                # explore
                action = random.choice(self.avail_taxi)
            else:
                # exploit
                action = int(numpy.argmax(self.q_taxi[obs]))

            # Update to our new state
            #    obs = new_obs
            self.curr_obs = obs
            self.act = action
            # Decrease epsilon
            # epsilon = np.exp(-decay_rate*episode)
            if done:
                self.episode_counts += 1
                self.eps = numpy.exp(-self.decay_r * self.episode_counts)

            return action

        if self.env_name == 'kbca':

            obs_convt = [-2 if x == "" else x for x in obs]
            new_obs = sum(obs_convt)

            self.q_kbca[self.curr_obs, self.act] = self.q_kbca[self.curr_obs, self.act] + self.lr * (
                    reward + self.disc_r * numpy.max(self.q_kbca[new_obs, :]) - self.q_kbca[self.curr_obs, self.act])

            if random.uniform(0, 1) < self.eps and -29 <= new_obs <= -8:

                action = random.choices(self.avail_kbca, weights=[0.0, 1.0], k=1)[0]

            elif random.uniform(0, 1) < self.eps and -8 < new_obs <= 4:

                action = random.choices(self.avail_kbca, weights=[0.2, 0.8], k=1)[0]

            elif random.uniform(0, 1) < self.eps and 4 < new_obs <= 16:

                action = random.choices(self.avail_kbca, weights=[0.8, 0.2], k=1)[0]

            else:
                # exploit
                action = numpy.argmax(self.q_kbca[new_obs, :])

                # take action and observe reward
            #    new_obs, reward, done, info = env.step(action)

            # Q-learning algorithm
            #    self.q_taxi[obs,action] = self.q_taxi[obs,action] + learning_rate * (reward + discount_rate * np.max(self.q_taxi[new_obs,:])-self.q_taxi[obs,action])

            # Update to our new state
            #    obs = new_obs
            self.curr_obs = new_obs
            self.act = action
            # Decrease epsilon
            # epsilon = np.exp(-decay_rate*episode)
            if done:
                self.episode_counts += 1
                self.eps = numpy.exp(-self.decay_r * self.episode_counts)

            return action

        if self.env_name == 'kbcb':

            obs_convt = [-2 if x == "" else x for x in obs]
            new_obs = sum(obs_convt)

            self.q_kbcb[self.curr_obs, self.act] = self.q_kbcb[self.curr_obs, self.act] + self.lr * (
                    reward + self.disc_r * numpy.max(self.q_kbcb[new_obs, :]) - self.q_kbcb[self.curr_obs, self.act])

            if random.uniform(0, 1) < self.eps and -29 <= new_obs <= -8:

                action = random.choices(self.avail_kbcb, weights=[0.0, 1.0], k=1)[0]

            elif random.uniform(0, 1) < self.eps and -8 < new_obs <= 4:

                action = random.choices(self.avail_kbcb, weights=[0.3, 0.7], k=1)[0]

            elif random.uniform(0, 1) < self.eps and 4 < new_obs <= 16:

                action = random.choices(self.avail_kbcb, weights=[0.8, 0.2], k=1)[0]

            else:
                # exploit
                action = numpy.argmax(self.q_kbcb[new_obs, :])

                # take action and observe reward
            #    new_obs, reward, done, info = env.step(action)

            # Q-learning algorithm
            #    self.q_taxi[obs,action] = self.q_taxi[obs,action] + learning_rate * (reward + discount_rate * np.max(self.q_taxi[new_obs,:])-self.q_taxi[obs,action])

            # Update to our new state
            #    obs = new_obs
            self.curr_obs = new_obs
            self.act = action
            # Decrease epsilon
            # epsilon = np.exp(-decay_rate*episode)
            if done:
                self.episode_counts += 1
                self.eps = numpy.exp(-self.decay_r * self.episode_counts)

            return action

        if self.env_name == 'kbcc':

            obs_convt = [-2 if x == "" else x for x in obs]
            new_obs = sum(obs_convt)

            self.q_kbcc[self.curr_obs, self.act] = self.q_kbcc[self.curr_obs, self.act] + self.lr * (
                    reward + self.disc_r * numpy.max(self.q_kbcc[new_obs, :]) - self.q_kbcc[self.curr_obs, self.act])

            if random.uniform(0, 1) < self.eps and -29 <= new_obs <= -8:

                action = 1

            elif random.uniform(0, 1) < self.eps and -8 < new_obs <= 4:

                action = random.choices([0, 1], weights=[0.2, 0.8], k=1)[0]

            elif random.uniform(0, 1) < self.eps and 4 < new_obs <= 16:

                action = random.choices([0, 2], weights=[0.8, 0.2], k=1)[0]

            else:
                # exploit
                action = numpy.argmax(self.q_kbcc[new_obs, :])

                # take action and observe reward
            #    new_obs, reward, done, info = env.step(action)

            # Q-learning algorithm
            #    self.q_taxi[obs,action] = self.q_taxi[obs,action] + learning_rate * (reward + discount_rate * np.max(self.q_taxi[new_obs,:])-self.q_taxi[obs,action])

            # Update to our new state
            #    obs = new_obs
            self.curr_obs = new_obs
            self.act = action
            # Decrease epsilon
            # epsilon = np.exp(-decay_rate*episode)
            if done:
                self.episode_counts += 1
                self.eps = numpy.exp(-self.decay_r * self.episode_counts)

            return action

    def register_reset_test(self, obs):

        if self.env_name == 'acrobot':
            action = random.choice([0, 2])

        if self.env_name == 'taxi':
            action = int(numpy.argmax(self.q_taxi[obs]))

        if self.env_name == 'kbca':
            action = 1

        if self.env_name == 'kbcb':
            action = 1

        if self.env_name == 'kbcc':
            action = 1

        return action

    def compute_action_test(self, obs, reward, done, info):

        if self.env_name == 'acrobot':
            #   discrete_obs = tuple(numpy.clip(((obs - self.obs_space_low) / self.bucket_size).astype(numpy.int), None, 15))
            #   action = numpy.argmax(self.q_acro[discrete_obs])
            if obs[1] <= 0 and obs[4] < 0:
                action = 2

            elif obs[1] < 0 and obs[4] >= 0:
                action = 0

            elif obs[1] > 0 and obs[4] >= 0:
                action = 0

            elif obs[1] > 0 and obs[4] < 0:
                action = 2

        if self.env_name == 'taxi':
            action = int(numpy.argmax(self.q_taxi[obs]))

        if self.env_name == 'kbca':
            obs_convt = [-2 if x == "" else x for x in obs]
            new_obs = sum(obs_convt)
            action = numpy.argmax(self.q_kbca[new_obs, :])

        if self.env_name == 'kbcb':
            obs_convt = [-2 if x == "" else x for x in obs]
            new_obs = sum(obs_convt)
            action = numpy.argmax(self.q_kbcb[new_obs, :])

        if self.env_name == 'kbcc':
            obs_convt = [-2 if x == "" else x for x in obs]
            new_obs = sum(obs_convt)
            action = numpy.argmax(self.q_kbcc[new_obs, :])

        return action
