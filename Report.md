## Udacity Deep Reinforcement Learning Nanodegree
# Collaboration and Competition Project Report

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

#### Tennis cooperative environment

In the Tennis environment, 2 agents are trained to bounce a ball, with the objective of keeping it in play
for as many exchanges as possible.

2 actions for 2 players must be provided to the tennis evironment at every timestep, and the tennis environment returns 
and observation, a reward and a game end boolean for each of the 2 players. 

##### Observation and State #####

The observation for each player is actually the stack of the 3 last measurements. The current measurement for a player 
is an 8-dimensional vector representing the position and speed of the own racket and of the ball in the 2D tennis space.
Although the very last measurement, which includes speed, would capture most of the relevant information, the redundancy
of this observation representation may be beneficial in mapping the constant acceleration of gravity and 
impacts between the rackets and the ball.

The terms Observation and State are often used interchangeably in an environment where the state can be fully observed.
In this environment, as there are 2 players, i refer to observation as to the set of information that an agent can 
observe, and to state as to the whole information on the current situation of the environment. In practice, 
the state of the environment includes the position and speed of both rackets, while the observation of each player only 
includes position and speed of one racket.

##### Collaborative training #####

Several algorithms can be trained on this collaborative task, some options are the following:

1. **A single agent with full state**. A single is fed with the ensemble of states (and possibly last actions) of both players, and jointly selects
the actions of both players. This can be done with a single actor-critic network, with same critic but separate actors, 
 or with separate networks which possibly share the experience buffer.
 This option has been discarded as it does not seem fair. I want my agent to be able to play tennis collaboratively 
 without having to access the other player's obesrvation or action.
 
2. **A single agent with separate observations**. A single agent can be trained and used to take actions for both players, having access to the observation of only one
player at a time. The Agent can use the same actor critic network in both roles, and could chose to draw actions from 
the target actor network instead of from the local network for one of the two players.

3. **Separate agents sharing information**. 2 separate agents can be trained, and these can share some common 
information. These could for example share the same experience buffer (or just have identical experience buffers) or 
even a common critic network, or perhaps just a common target critic network.

4. **Completely separate agents**. Training 2 completely separate ddpg agents is also an options although exploiting the 
information available from the interaction of the agents should be beneficial.

In my experiments, I have explored using a single agent (option 2.), using the target actor network for one of the 
two players, and using separate agents with same experience buffer (option 3 shared experiences). None of the options 
stood out as superior to others, my final network has been obtained with 2 separate agents sharing experiences.

I have preferred the separate agent solution because I do not want my agent to learn a strategy which is compatible with
the strategy of different trained agents and does not rely on itself playing on both sides.
Actually, this inter-dependency could also happen when 2 separate agents are trained toghether, and there is no reason 
why each of the 2 agents should always play in the same role. 
Training N (2 or 3 for example) separate players and randomly selecting 2 players in each episode would also be a 
good strategy to lead to roboust playing styles which avoid inter-dependency.

To verify that my solution does not rely on the inter-dependency between agents, I will show [0.5 - 1.0] average score 
combining 2 agents which never actually played together during training.

### Implementation

#### Project structure

 - **ddpg_tennis_SOLUTION.py** is the main file of the project. It is used to train a DDPG agent on the tennis 
    environment and includes several training improvement ideas.
    - A *coach* algorithm (commented out) tried to stabilize learning by storing and reloading network state depending on
    score trend. The idea was to roll back to a previous state and reduce training rate every time a performance drop is 
    observed.
    - an *epsilon_greedy* noise approach. A very moderate correlated noise is added to actions to favour to balance
    exploration and exploitiation, but a strong additional noise would be added to some episodes with probability 
    noise_p, to prevent the agent from locking in a 0 reward state.
    - a *reward spread* mechanism to speed uo training in a reward sparse environment. In the tennis environment 
    reward is awarded with delay with respect to the action which is actually responsible for obtaining it, and is 
    awarded in a very limited number of actions. A local buffer stores the latest experiences. Whenever an experience 
    including a non-zero reward is received from the environment, this reward is spread among the N latest experiences.
 - **inspect_results.py** is the main file used to show the trained agent performance.
    A pair of agents can be loaded from saved checkpoints and tested on the TENNIS environment. 
    By default, **inspect_results.py** will load a pair of agents scoring on average >0.5 on the first 100 episodes of 
    TENNIS environmnent. This has been tested for seed options 0 and 1 as well as specifying no seed in my machine.
 - **linear_network.py** and  **critic_network.py** are customizable neural network configuration for actor (normal) and 
  critic (the additional action input can be fed at later layers). The most interesting feature is the ability of saving
   and loading a network to a checkpoint file, which includes not only the weights but also all the configuration 
   parameters 
 - **ddpg_agent.py**: Custom ddpg implementations including the following options:
   - [twin ddpg](https://spinningup.openai.com/en/latest/algorithms/td3.html]):
   a ddpg implementation which tries to address the possible overestimation of the Q function by the critic. 
   Twin ddpg uses 2 local and 2 target critic networks and uses the minimum among the Q values estimated by 
   the critic networks in training.
   - *lazy_actor* weight, adding cost to high action values in an attempt to obtain smoother, more natural playing 
   styles. This adds a positive term proportional to the action energy to the gradient computed in actor network 
   training.
 - **experience_replayer.py** the experience buffer used by the ddpg_agent, supporting a from of prioritized experience
   replay.
 
### Approach, Experimentation and Future Work

In the previous assignments, I have learned to my expense that tuning a baseline solution and performing the minimum 
number of modifications is a much safer and direct strategy for getting to a solution than going towards a custom 
implementation.

Scientific papers such as [Deep Reinforcement Learning that Matters](https://arxiv.org/pdf/1709.06560.pdf) 
confirm my experience showing how performance id deep reinforcement learning is strongly affected by random seeding, 
specific implementation choices, hyperparameters. These aspects could easily outweight algorithm modifications 
in determining the overall score.

Despite this, as this is the last assignment, I have decided to implement my own ddpg solution and to experiment with 
several ideas and modifications. I felt like trying to implement algorithms and custom modifications would probably 
delay my delivery and expose me to risks but would force me to go deeper in the algorithm and understand it better.
 
Experiments i have conducted included:

- using the same agent for both players
- using the local actor to play against the target actor
- using separate agents with shared experiences
- adding noise in different magnitude and configurations
- implementing a twin ddpg algorithm 
- implamenting a *coach* algorithm to aggress instability and recover form performance
- testing 3 layer networks and and tiny networks
- stripping past observations from the observation vectors (using 8 dimensional observation vectors)
- (adaptively) modifying learning rate
- spreading reward among last experiences to reduce reward sparsity
- constraining actor gradient to obtain smoother actions
- experimenting with a number of other parameters of the neural networks.

None of these ideas revealed itself to have a crucial impact on the performance of my agent, but most of them would have
required a much more significant research work and all of them were worth trying and are the starting point of my future 
work. 

#### Brute force solution

As my time for this project has come to an end, I have decided to suspend experimentation and settle on a brute force 
solution to obtain an agent (a pair of agents) which will satisfy the project requirements. 

To solve the tennis environment, I have trained a pair of vanilla ddpg agents adding frequent shots of strong correlated
noise to their actions, to favor exploration. The configuration used for this long training run is as follows:

The pair of ddpg agents used in this training run have the following configuration:
 - Replay buffer size : 100.000 experiences, used only when at least 20.000 are stored
 (For simplicity, the Replay Buffer is not shares but there are 2 identical replay buffers where experienced from both 
 agents are stores, one in each angent.)
 - batch size : 128 samples
 - Gamma (discount factor) : 0.99
 - Tau (soft update of target network) : 0.1
 - Learning Rate of Actor network : 0.0001 (Actor learing takes place every 2 critic learning runs) 
 - Learning Rate of Critic network : 0.0005 
 - Each Agent Learns from a batch of samples in the replay buffer after every 2 action iterations
 - Both networks architectures use 400 + 300 nodes with relus as default. Actions are fed to the second layer in the Critic Network
 - Filtered gaussian noise with standard deviaton 0.5 is added to each action in an episode with probability decaying from 1 to 1/20
 - Filtered gaussian noise with standard deviaton 0.02 is added to each action in all episodes

In the **delivery** release of this repository, ddpg_tennis_SOLUTION can be run to repeat training in the same configuration. 

This training run lasted several hours, and I have collected checkpoints of agents states acheving high average score.
Due to the instability of the algorithm and to the strong noise,  my agents did not score high for 100 consecutive 
episodes during training. this can be observed in the following figure

Having saved their state, I could anyways verify that my agents pass the 100 episodes average 
score requirement during test. The following figures show the per episode and average score of the selected pair of 
agents tested for 200 frames on the tennis environment with seeds 0 to 5. The agent consistently collects score >0.5.

### Future Work

Many potential directions for future work have been highlighted in the  **Approach, Experimentation and Future Work** paragraph. 
Exploiting the ability to save and restore weights as a form stabilization seems the most promising direction.

