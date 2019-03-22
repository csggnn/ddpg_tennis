from unityagents import UnityEnvironment
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
import random
from ddpg_agent import ddpgAgent

import pickle
import os


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
#states = env_info.vector_observations[:,-8:]
states = env_info.vector_observations
state_size = states[1].shape
print('There are {} agents. Each observes a state with length: {}'.format(states[0].shape, state_size))
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


agent = ddpgAgent(state_size=states.shape[1], act_size=action_size, seed=0)
agent2 = ddpgAgent(state_size=states.shape[1], act_size=action_size, seed=1)
np.random.seed(2)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)


noise_decay_p=0.99
noise_start_p=1
noise_stop_p=0.05

noise_std=0.5
noise_std_low = 0.02
noise_corr=0.7
noise_p=noise_start_p
scores_deque = deque(maxlen=100)
scores_list = []
scores_mean100 = []
act_run=1
last_steps=[]
lazy_player=0.0
coach_last_score=0.0
noise=[[],[]]
noise_episode=[[],[]]

max_score_mean100=0.001
max_score_mean25=0.001

for i in range(num_agents):
    last_steps.append(deque(maxlen=3))

for episode in range(1, 150000):   # play game for 5 episodes
    actions = np.zeros([num_agents, action_size], np.float)
    env_info = env.reset(train_mode=True)[brain_name]     # reset the environment
    states = env_info.vector_observations
    #states = env_info.vector_observations [:,-8:]                 # get the current state (for each agent)

    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    act_i=0
    noise_p = max(noise_p*noise_decay_p, noise_stop_p)
    #rewards=np.array(env_info.rewards)
    dones =  env_info.local_done
    if np.any(dones):  # exit loop if episode finished
        break


    for i in range(2):
        noise_episode[i] = np.random.rand()<noise_p
        noise[i]=np.random.randn(action_size)


    while True:
        if np.any(dones):  # exit loop if episode finished
            break
        act_i=act_i+1

        for i in range(2):
            noise[i] = noise[i]*noise_corr +  np.random.randn(action_size)*noise_std*(1-noise_corr)
        #for i in range(2):
        #    [actions[i], ns[i]] = agent.get_action(states[i], noise, ns[i])
        if episode>0:
            actions[0]=agent.get_action_loc(states[0])
            actions[0] +=noise[0]*(noise_episode[0]*noise_std + noise_std_low)
            actions[1]=agent2.get_action_loc(states[1])
            actions[1]+=noise[1]*(noise_episode[1]*noise_std + noise_std_low)
        else:
            actions[0]=np.random.randn(action_size)*noise[0]
            actions[1]=np.random.randn(action_size)*noise[1]

        #actions[1] = 0
        #actions[0] = 0
        #actions[:,1]=(1+actions[:,1])/2
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations # get next state (for each agent)
        #next_states = env_info.vector_observations[:, -8:]          # get next state (for each agent)

        rewards = (env_info.rewards-np.sum(np.square(actions), 1)*lazy_player)  # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finishes
        scores += env_info.rewards                         # update the score (for each agent)
        for i in range(2):
            last_steps[i].append(
                {"state": states[i], "action": actions[i], "reward": 0, "next_state": next_states[i], "done": dones[i]})
        if np.any(dones):                                  # exit loop if episode finished
            # first save the experiences
            for i in range(2):
                while len(last_steps[i])>0:
                    exp = last_steps[i].popleft()
                    agent.store_exp(exp)
                    agent2.store_exp(exp)
            # then go out
            break
        for i in range(2):
            if rewards[i]!=0:
                saved_steps=len(last_steps[i])
                for exp in last_steps[i]:
                    exp["reward"]=exp["reward"]+rewards[i]/saved_steps
            if len(last_steps[i])==last_steps[i].maxlen:
                exp=last_steps[i].popleft()
                agent.store_exp(exp)
                agent2.store_exp(exp)

        states = next_states                               # roll over states to next time step

        if act_i%2==0 and act_i < 30:
            agent.train()
            agent2.train()

    scores_deque.append(np.max(scores))
    scores_mean100.append(np.mean(scores_deque))
    scores_list.append(np.max(scores))
    if episode % 200 == 0:
        ax.clear()
        ax.plot(np.arange(1, len(scores_list) + 1), scores_list)
        ax.plot(np.arange(1, len(scores_mean100) + 1-50), scores_mean100[50:])

        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.draw()
        plt.pause(.001)

    # save the score list
    if episode % 50 == 0:
        with open("tmp_checkpoints/score.p", 'wb') as sf:
            pickle.dump([scores_list, scores_mean100], sf)

    #save weights
    if (episode % 2000 == 0) or ((episode % 200 == 0) and (episode<=1000)) \
            or (scores_mean100[-1] > max_score_mean100+0.01) or np.mean(scores_list[-25:])>max_score_mean25+0.02:
        print("agent_%06d_s100_%3.3f_s25_%3.3f" % (episode, scores_mean100[-1], np.mean(scores_list[-25:])))
        if (scores_mean100[-1] > max_score_mean100):
            max_score_mean100 = scores_mean100[-1]
        if (np.mean(scores_list[-25:]) > max_score_mean25):
            max_score_mean25 = np.mean(scores_list[-25:])
        d="tmp_checkpoints/agent_%06d_s100_%3.3f_s25_%3.3f/" % (episode, scores_mean100[-1], np.mean(scores_list[-25:]))
        if not os.path.exists(d):
            os.mkdir(d)
            os.mkdir(d + "1")
            os.mkdir(d + "2")
        agent.checkpoint(d + "1/")
        agent2.checkpoint(d + "2/")
        #with open(d+"/score.p", 'wb') as sf:
        #    pickle.dump([scores_list, scores_mean100], sf)


"""
    if (episode>200 and  episode % 25  == 0):
        print("episode "+str(episode)+ ", mean last_score: "+str(np.mean(scores_list[-25:])) )
        if coach_last_score>0.03 and np.mean(scores_list[-25:])<coach_last_score*0.7:
            coach_last_score*=0.7
            print("rolling back as score was previously "+ str(coach_last_score))
            agent.load_checkpoint("tmp_checkpoints/coach_last")
            sel= random.choice(["a", "c", "n"])
            if sel =="a":
                agent.pars["lr_act"] *=0.85
            elif sel == "c":
                agent.pars["lr_crit"] *= 0.85
            elif sel == "n":
                agent.pars["noise_in"] *= 1.2
        else:
            coach_last_score=np.mean(scores_list[-25:])
            agent.checkpoint("tmp_checkpoints/coach_last")
            if coach_last_score>0.03:
                agent.pars["lr_act"] *=1.01
                agent.pars["lr_crit"] *=1.01
                agent.pars["noise_in"] *=0.99


"""
wait = input("PRESS ENTER TO CONTINUE.")
env.close()

