# DDPG Tennis

**DDPG Tennis** is a PyTorch project implementing a reinforcement learning 
agents controlling paddles in a *Tennis Environment*. 

In a **Tennis Environment**, agents controlling tennis paddles try exchange a ball making as many 
passes as possible without letting it fall. The Tennis environment is a collaborative 
environment where the reward of each agent corresponds to the number of times it touches the ball.

This is a collaborative task, which can be tackled imposing different degrees of 
cooperation:
 1. 2 separate agents are trained independently.
 1. 2 single agent is trained using alternatively the 2 players state observation, and reward. 
    This apporach is viable if actions and state exhibit simmetry.
 1. 2 separate agents are used but share the same experience buffer. this approach can be used if observations are
    symmetrical (but actions might not be)
 1. a single agent receives the observation and reward of each player and decides for both actions.
 
 
## Installation

In order to run the code in this project, you will need a python3.6 environment with the 
following packages:
- numpy
- matplotlib
- pickle
- torch

You will also need to install the Unity Reacher Envronment: 
- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- [Win_32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- [Win_64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


## Usage

**WARNING**: project is not complete, for the moment only a random agent can be seen in ation running ddpg_tennis_SOLUTION.py

Simply run ddpg_tennis_SOLUTION to see 2 trained DDPG agents solving the Tennis environment.

The ddpg_reacher_SOULUTION simply calls *ddpg_reacher_solution("show", 500)*, which loads a DDPG solving the Reacher 
Environment after 500 training episodes.

Alternatively, you cn edit the main routine to call ddpg_tennis_solution("learn") and train a new DDPG agent. 
 
More details on the file structure, implementation choices and parameters in Report.md