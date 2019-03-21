import matplotlib.pyplot as plt
import numpy as np
import pickle
from ddpg_agent import ddpgAgent
from unityagents import UnityEnvironment
import time

with open("checkpoints/lucky shot 1/score.p",'rb') as pf:
    scores_list, scores_mean100 = pickle.load(pf)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.clear()
ax.plot(np.arange(1, len(scores_list) + 1), scores_list)
ax.plot(np.arange(1, len(scores_mean100) + 1 - 50), scores_mean100[50:])

plt.ylabel('Score')
plt.xlabel('Episode #')
agent2 = ddpgAgent("checkpoints/lucky shot 1/agent_013580_s100_0.121_s25_0.146/1")
agent = ddpgAgent("checkpoints/lucky shot 1/agent_048951_s100_0.134_s25_0.142/2")


env = UnityEnvironment(file_name="../Tennis_Linux/Tennis.x86_64", seed=0)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]
states = env_info.vector_observations
actions=[[],[]]

log_score=[]

for i in range(100):
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    scores= np.zeros(2)

    done =False
    while True:
        actions[0] = agent.get_action_loc(states[0])
        actions[1] = agent2.get_action_loc(states[1])
        states = env_info.vector_observations
        scores += env_info.rewards
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]
        if any(env_info.local_done):
            break
    log_score.append(np.max(scores))
    print(np.mean(log_score))

print(np.mean(log_score))
wait = input("PRESS ENTER TO CONTINUE.")



