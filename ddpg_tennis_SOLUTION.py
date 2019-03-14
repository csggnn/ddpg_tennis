from unityagents import UnityEnvironment
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import ddpgAgent

env = UnityEnvironment(file_name="../Tennis_Linux/Tennis.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# for i in range(1, 60):                                      # play game for 5 episodes
#     env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
#     states = env_info.vector_observations                  # get the current state (for each agent)
#     scores = np.zeros(num_agents)                          # initialize the score (for each agent)
#     while True:
#         actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
#         actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
#         env_info = env.step(actions)[brain_name]           # send all actions to tne environment
#         next_states = env_info.vector_observations         # get next state (for each agent)
#         rewards = env_info.rewards                         # get reward (for each agent)
#         dones = env_info.local_done                        # see if episode finished
#         scores += env_info.rewards                         # update the score (for each agent)
#         states = next_states                               # roll over states to next time step
#         time.sleep(0.05)
#         if np.any(dones):                                  # exit loop if episode finished
#             break
#     print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))


agent = ddpgAgent(state_shape=states.shape[1], act_shape=action_size, seed=5)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)


noise=10
noise_decay=0.999
noise_stop=0.01
scores_deque = deque(maxlen=100)
scores_list = []
act_run=1
last_steps=[]
for i in range(num_agents):
    last_steps.append(deque(maxlen=10))

for episode in range(1, 6000):   # play game for 5 episodes
    actions = np.zeros([num_agents, action_size], np.float)
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    act_i=0
    noise=max(noise*noise_decay, noise_stop)
    ns=[np.random.rand(action_size)*2.0-1 for i in range(2)]
    rewards=np.array(env_info.rewards)
    dones =  env_info.local_done
    if np.any(dones):  # exit loop if episode finished
        break

    boost=0
    while True:
        if np.any(dones):  # exit loop if episode finished
            break
        act_i=act_i+1
        for i in range(2):
            [actions[i], ns[i]] = agent.get_action(states[i], noise, ns[i])
            boost+=0.1
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        for act_r in range(act_run):
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards += np.array(env_info.rewards) #    +boost   # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finishes
            scores += env_info.rewards                         # update the score (for each agent)
            if np.any(dones):                                  # exit loop if episode finished
                break
        for i in range(2):
            last_steps[i].append({"state": states[i], "action": actions[i], "reward": 0,"next_state": next_states[i], "done":dones[i]})
            if rewards[i]!=0:
                saved_steps=len(last_steps[i])
                for exp in last_steps[i]:
                    exp["reward"]=exp["reward"]+rewards[i]/saved_steps
            if len(last_steps[i])==last_steps[i].maxlen:
                exp=last_steps[i].popleft()
                agent.store_exp(exp)
        states = next_states                               # roll over states to next time step

        if act_i%2==0:
            agent.train()

    scores_deque.append(np.max(scores))
    scores_list.append(np.max(scores))
    if episode % 200 == 0:
        ax.clear()
        ax.plot(np.arange(1, len(scores_list) + 1), scores_list)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.draw()
        plt.pause(.001)
    print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))
env.close()

