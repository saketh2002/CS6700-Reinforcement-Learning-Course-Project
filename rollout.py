import aicrowd_gym
import os
import time


while not os.path.exists("/tmp/let-go"):
    time.sleep(1)


# Create env and complete a few episodes
env = aicrowd_gym.make("Acrobot-v1")
env.reset()
done = False
scores = 0
while not done:
    obs, reward, done, _ = env.step(env.action_space.sample())
    scores += reward

print("Final scores:", scores)
