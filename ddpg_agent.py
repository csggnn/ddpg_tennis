from collections import namedtuple
from linear_network import LinearNetwork
from critic_network import CriticNetwork
from experience_replayer import ExperienceReplayer
import torch
import numpy as np


class ddpgAgent:
    def __init__(self, state_shape=(784,), actor_layers=(400, 300),
                 critic_layers=(400, 300), seed=None, act_shape=(1,),
                 action_layer=-1, batch_size=128):
        lr_actor= 0.0001
        lr_critic=0.0001
        self.actor_loc = LinearNetwork(input_shape=state_shape, lin_layers=actor_layers, output_shape=act_shape,
                                        seed=seed)
        self.actor_tg = LinearNetwork(input_shape=state_shape, lin_layers=actor_layers, output_shape=act_shape,
                                        seed=seed)
        self.actor_optimizer = torch.optim.Adam(self.actor_loc.parameters(), lr=lr_actor)

        self.critic_loc_1 = CriticNetwork(input_shape=state_shape, lin_layers=critic_layers, output_shape=(1,),
                                         action_layer=action_layer, action_shape=act_shape, seed=seed)
        self.critic_loc_2 = CriticNetwork(input_shape=state_shape, lin_layers=critic_layers, output_shape=(1,),
                                         action_layer=action_layer, action_shape=act_shape, seed=seed)

        self.critic_tg_1 = CriticNetwork(input_shape=state_shape, lin_layers=critic_layers, output_shape=(1,),
                                         action_layer=action_layer, action_shape=act_shape, seed=seed)
        self.critic_tg_2 = CriticNetwork(input_shape=state_shape, lin_layers=critic_layers, output_shape=(1,),
                                         action_layer=action_layer, action_shape=act_shape, seed=seed)

        self.critic_optimizer1 = torch.optim.Adam(self.critic_loc_1.parameters(), lr=lr_critic)
        self.critic_optimizer2 = torch.optim.Adam(self.critic_loc_2.parameters(), lr=lr_critic)

        self.batch_size=batch_size
        self.mem = ExperienceReplayer(30000, wait_fill=0.02, default_prio=0.00001, epsilon=0.5)

        #self.exptuple = namedtuple("experience", "state action reward next_state done")

        self.train_count =0
    def store_exp(self, exp):
        self.mem.store(exp, exp["reward"])

    def get_action(self, state, noise_weight=0, last_noise_sample=None):
        action = self.actor_loc.forward_np(state)
        if noise_weight>0:
            noise_sample=0.1*np.random.rand(len(action))+0.9*last_noise_sample-0.05
            action = action/(1.0+noise_weight) + noise_sample*noise_weight/(1.0+noise_weight)
        else:
            noise_sample=None
        # should i add noise?
        return action, noise_sample


    def train(self):
        exp_batch =self.mem.draw(self.batch_size)
        if exp_batch is None:
            return

        states_t = torch.from_numpy(np.vstack([e["state"] for e in exp_batch])).float()
        next_states_t = torch.from_numpy(np.vstack([e["next_state"] for e in exp_batch])).float()
        rewards_t = torch.from_numpy(np.vstack([e["reward"] for e in exp_batch])).float()
        actions_t = torch.from_numpy(np.vstack([e["action"] for e in exp_batch])).float()
        dones_t = torch.from_numpy(np.vstack([np.uint8(e["done"]) for e in exp_batch])).float()

        # twin trick 1: add noise to
        actions_next = self.actor_tg.forward(next_states_t) # would add noise here?
        actions_next = torch.clamp(actions_next+torch.clamp(torch.randn(actions_next.shape)*0.001, -0.003, 0.003), -1.0, 1.0)


        # trick 2: avoid optimistic q: get the minimum of two targets.
        q_target_next = torch.min(self.critic_tg_1.forward(next_states_t, actions_next),
                                  self.critic_tg_2.forward(next_states_t, actions_next))

        # if done is true, does this mean that state is a terminal state or that next state is a terminal state?
        # must be next_state is a terminal state otherwise i could not have taken action from state.
        q_target = (1.0-dones_t)*q_target_next*0.9 + rewards_t

        q_expected_1 = self.critic_loc_1.forward(states_t, actions_t)
        q_expected_2 = self.critic_loc_2.forward(states_t, actions_t)
        # cost is difference between estimated Q value for next state and estimated Q value for current state + reward
        critic_error = torch.mean((q_target-q_expected_1)**2 + (q_target-q_expected_2)**2)

        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        critic_error.backward()
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()

        self.train_count = self.train_count+1

        if self.train_count>3:
            self.train_count=0

            self.actor_optimizer.zero_grad()
            self.critic_loc_1.eval()
            self.actor_loc.train()
            prev_actions = self.actor_loc.forward(states_t)
            # selecr action maximizing Q for current state.
            actor_q = -torch.mean(self.critic_loc_1.forward(states_t, act=prev_actions))

            actor_q.backward()
            self.actor_optimizer.step()
            self.critic_loc_1.train()
            self.actor_loc.eval()
            self.soft_update(self.actor_loc, self.actor_tg, 0.01)
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_loc_1, self.critic_tg_1, 0.01)
        self.soft_update(self.critic_loc_2, self.critic_tg_2, 0.01)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)







