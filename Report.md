## Udacity Deep Reinforcement Learning Nanodegree
# Continuous Control Project Report

### Learning Strategy

This project is a solution to the cooperative Tennis environment using 2 identical agents implementing [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971).  


The the next chapter presents a short overview of DDPG, which can be seen as an adaptation of Deep Q-Networks to the case of 
countinuous action space.
For a very basic overview od DQN and of its building blocks, please refer the report for my previous projects [qnet_navigation](https://github.com/csggnn/qnet_navigation/blob/master/Report.md)
For a more detailed description of DDPG please refer to the orignial paper, or if you want to dig deeper into reinforcement learning consider the 
[Udacity Deep Reinforcement Learning Nanodegree](https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

#### Intutition of Deep Deterministic Policy Gradients

**Deep Deterministic Policy Gradients** can be seen as an adaptation of DQN to the case of continuous action spaces.

In DQN a neural network is used as a function approximator of the representation of a Q function mapping each possible 
state in a continuous state space to set of Q values, one for each possible action. An agent feed a state observation to 
DQN and select the action corresponding to the highest Q value for that state.

If the action space is continuous, this approach no longer works, as a DQN can only produce a finite set of action value
 while the possible actions are infinite. 
 
DQN is a value method, it uses a neural network to model an action value function, producing an action value for each 
possible action in a state.
DDPG modifies DQN by using 2 networks: an actor network estimates the best action for an input state, while a critic 
network estimates the action value associated an input state, and the action selected by the actor.

The actor network uses the critic network to optimize its weights: weights are updated by gradient ascent, maximizing the
action value associated to the actions which the network would select for each state. This action value is obtained
from the critic network.

The critic network uses the action network as input, the action selected by the action network is fed as an input to the 
fully connected layers of the critic network which estimate the Q value associated to that specific action.
 
Experience replay is used as in DQN to decorrelate inputs: experiences, i.e. state-action-reward-new_state tuples 
observed by the agent are not directly used for training but are instead stored in a experience buffer. 
Learning occurs independently form actor's interaction with the environment, at every training itration a random batch of
experiences are drawn from experience memory. This approach has 2 main advantages:
- experiences can be used multiple times, speeding up learning
- experiences are no longer correlated in time, preventing instability

Differently for DQN, where probability values ase associated to possible actions DDPG algorithm is deterministic in its 
nature. Correlated noise is artificially added to action outputs in order to improve exploration.

Similarly to DQN, both the actor and the ctritic networks are doubled: each defines a local network which is actually 
trained, and a separated target network which is used for computing expected output. 

In its original formulation, the DQN local nework, used in training, is periodically copied over to the target network. 
Periodically updating its weights, the target network gradually improves its accuracy while still remaining decoupled
from the Local network and thus preventing instability.
Differently from DQN original algorithm, but similarly to some DQN implementations, DDPG uses soft target network updates.
After each update of the local networks (agent and critic), the target networks weights are updated as follows

**w_{target}=  (1-Tau) w_{target} + (Tau) w_{local}**

Where **Tau** is a soft update constant form 0 to 1 (typically low). The resulting target weights will be a temporally smoothed and delayed
vertion of the local networks weights. 

#### Solving the Tennis cooperative environment

In the Tennis envitonment, 2 agents are trained to bounce a ball, with the objective of keeping it in play
for as many exchanges as possible.

Each agent receives an observation of the state which represents the own and ball location with respect to its side of the net.
Each agentis can interact with the environment controlling its horizontal and vertical movement with respect to the net.
It has been verified that the actions performed by each actor are symmetric with respect to the network. (using the same 
action for both actor results in symmetric movement)

As both state observation and actions are symmetric, the same policy can be used to control any of the 2 agents, we are 
thus free of training just one single ddpg agent and using it alternatively for both players.

### Implementation

#### Preliminary notes 

This project uses the baseline implementation of DDPG, with limited modifications as described in the remainder of this reprot

#### Project structure

 - **ddpg_tennis_SOLUTION.py** is the main file of the project. It can train a DDPG agent on a set of environments, 
    display random runs of the environemnts and load a trained DDPG agent.
    currently, *ddpg_reacher_solution("show")* is called at line 202 to show the behavior of a trained agent solving the
    reacher environment. select *ddpg_reacher_solution("train")* to train a new agent.
 - **train_DQN_agent.py** is the main file used during development the QAgent class implemented in 
    q_agent.py and the file used for training the selected DQN agent.
    This file can be used to retrain the successful agent or to train new agents for the Banana Collection environment 
    as well as for the CartPole and LunarLanding environemnts.
 - **continuous_action_env.py**, **continuous_gym_env.py** and **reacher1_env.py**: wrappers to the gym and reacher 
    environments, they have been developed so that the DDPG agent can be run both gym and Reacher environments with the 
    same code. LunarLanderContinuous an Pendulum environments have been used to verify the ability of the code to solve 
    simple environments before passing to the more complex reacher task.
 - **ddpg_agent.py**: Agent with minor modifications with respect to the baseline implementation
 - **model.py**: Neural network models used by ddpg_agent.py
 - **param_optim.py** is a random parameter optimization script used to test several parameter configurations.

### Improvements and Results

#### Random actions

In its original implementation, a random noise of magnitude comparable to the magnitude of the agent's acions was added 
to guarantee proper exploration. This noise component has been found too large and impacting on performance. 
In the implementation used in this project:
 - noise is added to episodes only with probability p: while some episodes are run with noise, others follow the learned 
 deterministic policy. If noise is always added to actions, states requiring precise actions may never be reached.
 - noise has lower magnitude and is reduced over time: similar to decaying epsilon for epsilon greedy policies.

This modification has been effective in the solution of the 2 gym environments, but did not lead to a solution of the 
Reacher environment. 

Restults obtained with ddpg and modified noise after parameter optimization are shown in the following graph.

![score_graph](score_300.png)

#### "Experience first"

An Experience buffer is used to decorrelate sampled experiences and thus reduce instability. In the original 
implementation of ddpg samples were drawn form the experience buffer as soon as a whole batch was available.
When only few experience samples are available, these samples will be indeed be strongly correlated and stability may be
compromised. *"Experience First"* is a simple modification do ddpg consists in triggering learning only when the experiance 
buffer has collected a sufficient number of samples (in my implementation 1/5 of its capacity). 

*I do not claim having invented this modification. Some existing implementation of ddpg I have looked at waited for the 
experience buffer to be full before triggering learning. Unfortunately I have been looking at several implementations of
ddpg and can no longer find a reference to this code* 

With this simple modification, ddpg algorithm was able to reach reaching average score
values beyond 25 in 1000 episodes, although not solving the environment.

![score_graph](score_waitmem_1000.png)

Weights leading to these results have been saved and used as a starting point for 500 additional episodes of training, 
but ddpg revealed its instability. 

![score_graph](score_waitmem_1500.png)

#### Agressive discount factor

Discount factor measures the relevance of future rewards to current action, and it makes sense for it to depend on the 
compact that current action has on reward. In this specific environment, a short sequence of actions (2-3) can in all
states be taken to get to a positive reward from any starting state. This means the current state an the sequence of 
actions taken up to a given moment have very little influence on rewards which are distant in the future.

The discount fator used in previous tests was 0.99. A new discount factor of 0.92 has been tested without changing other
parameters. This simple modification lead to a significant improvement. With this modified parameter the environment 
could be solved in less than 500 episodes.
![score_graph](score_discount_092.png)

### Final configuration Setup ###

The parameters selected in the final configuration are a result of a preliminary optimization run on DDPG before 
introducing "Experience First". Only the discount factor has been modified after this optimization.
A new optimization run after the latest modifications will probabily find different optimal parameters.

The DDPG algoritm which solved the environment in 500 episodes uses the following parameters:
 - Replay buffer size : 100.000 experiences
 - batch size : 64 samples
 - Gamma (discount factor) : 0.92 (the original solution which did not solve the environment used Gamma=0.92)
 - Tau (soft update of target network) : 0.01
 - Learning Rate of Actor network : 0.00001 (reduced by factor of 10 w.r.t. original implementation, as a result of performance optimization)
 - Learning Rate of Critic network : 0.0001 (as above)
 - Weight decay : 0 (weight decay was found negatively impact training in the reacher environment)
 - Learn from a batch of samples in the replay buffer after every iteration (more relaxed learning frequncies have been tested but did not provide better results)
 - Both networks architectures use 400 + 300 nodes with relus as default. Actions are fed to the second layer in the Critic Network
 - Noise has been reduced by factor of 0.02 w.r.t original implementation. It is further reduced by factor 0.99999 at every action, 
   resulting in a total attenuation of 0.00027 after 500 episodes.

folder checkpoints/final/stores checkpoints for the final agent, along with agent parameters and score

### Future work

#### Correlated Experiences and Experience First

The most critical point in the evolution of my project was the introduction of *"Experience First"*.  Waiting for the 
replay buffer to be at least partially filled up was an arbitrary 
modification which marked the change from very poor learning to effective trainig, and set the starting point for a 
solution of the environment. 

The final implementation waits for 20% of the experience buffer to be filled before starting training. This 20% is an 
arbitrary value which has been kept as it showed good results, but different values can be tested.

2 gym environment have been also been tested and could be solved with this ddpg implementation before introducing 
*Experience First*. It could be interesting to see if *Experience First* has any impact on these environment. If it is 
not the case, it would be interesting to test the same environments with a very small experience buffer, so as to verify 
whether they are effected at all by correlation of experiences

#### Tuning and Testing

Although several modifications and improvements such as such as [Proritized experience replay](https://arxiv.org/abs/1511.05952) 
could be introduced to try to improve performance stability 
and convergence a most challenging task in DDPG is in my opinion th verification of wether improvements and modifications are actually 
the responsible of an improvement in performance, and whether this improvement is roboust to test runs with different 
seeds, small variations of parameters, environments.

[Deep Reinforcement Learning that Matters](https://arxiv.org/pdf/1709.06560.pdf) shows how performance id deep reinforrcement 
learning is strongly affected by random seeding, specific implementation choices, hyperparameters. These aspects could 
easily outweight algorithm modifications in determining the overall score.

Before introducing new modifications to the DDPG algorithm, my trivial modifications on random weighting, *experience first*
and agressive discount vactor should be toroughly validated, and a deeper understanding of the impact of modification of the current
parameters should be obtained. 

#### Dynamic parameter update

As shown in the baseline solution for this project and in my first solution using *Experience First*, one of the key 
problems in DDPG is stability: the learned policy does not simply get to a platoe, as it is mostly the case for supervised 
learning, but can instead decrease and never recover. 

As the state of a DDPG agent just consists of the weights in its neutal networks (and to some degree the experiences in its
experience buffer) training could incude a continuous assessment of performance.
When a performance loss is observed, an agent could be rolled back to a previous state, and training could be 
resuming learning with some modification of parameters (e.g. with a decreased learning rate)



