import matplotlib.pyplot as plt
import numpy as np
import pickle
from ddpg_agent import ddpgAgent
from unityagents import UnityEnvironment
import time

with open("tmp_checkpoints/agent_096500_avscore_0.000/score.p",'rb') as pf:
    scores_list, scores_mean100 = pickle.load(pf)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.clear()
ax.plot(np.arange(1, len(scores_list) + 1), scores_list)
ax.plot(np.arange(1, len(scores_mean100) + 1 - 50), scores_mean100[50:])

plt.ylabel('Score')
plt.xlabel('Episode #')
agent = ddpgAgent("checkpoints/lucky_shot_0/agent_047000_avscore_0.034")


env = UnityEnvironment(file_name="../Tennis_Linux/Tennis.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations
actions=[[],[]]

for i in range(50):
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations

    done =False
    while True:
        actions[0] = agent.get_action_loc(states[0])
        actions[1] = agent.get_action_loc(states[1])
        states = env_info.vector_observations
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]
        if any(env_info.local_done):
            break
        time.sleep(0.2)


