import matplotlib.pyplot as plt
import numpy as np
import pickle
from ddpg_agent import ddpgAgent
from unityagents import UnityEnvironment
import sys

def scoreplot(picklename, figname, up_to=None):
    with open(picklename,'rb') as pf:
        scores_list, scores_mean100 = pickle.load(pf)

    if up_to is not None:
        scores_list=scores_list[:up_to]
        scores_mean100 = scores_mean100[:up_to]

    fig, ax = plt.subplots(figsize=(18, 6))


    ax.clear()
    ax.plot(np.arange(1, len(scores_list) + 1), scores_list, label="score")
    ax.plot(np.arange(1, len(scores_mean100) + 1 - 50), scores_mean100[50:], label="100 episodes mean")
    plt.grid()
    ax.legend()

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(figname, bbox_inches="tight")

def inspect_agent(agent_paths, seed):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    agent = ddpgAgent(agent_paths[0])
    agent2 = ddpgAgent(agent_paths[1])

    env = UnityEnvironment(file_name="../Tennis_Linux/Tennis.x86_64", seed=seed)

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

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.clear()
    ax.plot(np.arange(1, len(log_score) + 1), log_score, label="score")
    ax.plot([1, len(log_score) + 1], [np.mean(log_score), np.mean(log_score)], label="mean score")
    ax.legend()
    plt.grid()

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig("plots/score_inspect_seed_"+str(seed)+".jpg", bbox_inches="tight")


if __name__ == "__main__":

    scoreplot("checkpoints/final shot/score.p", "final_score.jpg", 15000)

    if len(sys.argv)>1:
        seed=int(sys.argv[1])
    else:
        seed=0
    agent_paths = ("checkpoints/final shot/agent_010975_s100_0.853_s25_1.332/2",
                   "checkpoints/final shot/agent_010975_s100_0.853_s25_1.332/2")
    inspect_agent(agent_paths,seed)




