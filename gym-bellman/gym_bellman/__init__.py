from gym.envs.registration import register

register(
    id='bellman-v0',
    entry_point='gym_bellman.envs:BellmansEnv',
)
