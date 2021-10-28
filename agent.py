from config import *

"""

DO not modify the structure of "class Agent".
Implement the functions of this class.
Look at the file run.py to understand how evaluations are done. 

There are two phases of evaluation:
- Training Phase
The method "learn" is invoked here. Use this phase to train your agents.

- Test Phase
The methods "registered_reset" and "compute_action" are invoked here. 
The final scoring is based on your agent's performance in this phase. 

"""


class Agent:

    def __init__(self,env):
        self.env_name = env
        self.config = config.config[env]
        pass


    def learn(self, state, action, reward, next_state, done):
        '''
        PARAMETERS  : 
            - state - discretized 'state'
            - action - 'action' performed in 'state'
            - reward - 'reward' received due to action taken
            - next_state - discretized 'next_state'
            - done - status flag to represent if an episode is done or not
        RETURNS     : 
            - NIL
        '''

        raise NotImplementedError

    def register_reset(self, obs):
        '''
        This function is called at the beginning of an episode. 
        PARAMETERS  : 
            - obs - raw 'observation' from environment
        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        '''

        raise NotImplementedError
        return action

    def compute_action(self, obs, reward, done, info):
        '''
        This function is called at all subsequent steps of an episode until done=True
        PARAMETERS  : 
            - observation - raw 'observation' from environment
            - reward - 'reward' obtained from previous action
            - done - True/False indicating the end of an episode
            - info -  'info' obtained from environment

        RETURNS     : 
            - action - discretized 'action' from raw 'observation'
        '''
        
        raise NotImplementedError
        return action
