from collections import namedtuple
from linear_network import LinearNetwork
from critic_network import CriticNetwork
from experience_replayer import ExperienceReplayer
import torch
import numpy as np
import pickle

def_pars={}
def_pars["erep_size"] = int(1e5)  # replay buffer size
def_pars["erep_fill"] = 0.2  # replay buffer size
def_pars["erep_eps"] = 1.0
def_pars["erep_def_prio"] = 0.001
def_pars["batch"] = 256  # minibatch size
def_pars["gamma"] = 0.95  # discount factor
def_pars["tau"] = 1e-1 # for soft update of target parameters
def_pars["lr_act"] = 3e-5  # learning rate of the actor
def_pars["lr_crit"] = 3e-5  # learning rate of the critic
def_pars["lazy_actor"] = 0.000
def_pars["train_act_every"] = 4 # learn only once every LEARN_EVERY actions
def_pars["actor_layers"] = (400, 300, 300)
def_pars["crit_layers"] = (400, 300, 300)
def_pars["act_input"] =-1


def_pars["noise_in"] = 0.00003

#def_pars["noise_dec"] = 0.999999
#def_pars["noise_start"] = 0.02
#def_pars["noise_stop"] = 0.0002


def_pars["twin"] = False
def_pars["ACTION_LAYER"] = -1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ddpgAgent:
    def __init__(self, state_size=(784,), act_size=(1,), seed=None, parameter_dict=None):

        if isinstance(state_size, str):
            checkpoint_fold=state_size
            with open(checkpoint_fold + "/pars.p", 'rb') as sf:
                self.pars=pickle.load(sf)
            self.initialize()
            self.load_checkpoint(checkpoint_fold)
            return

        else:
            if parameter_dict is None:
                self.pars=def_pars
            else:
                self.pars=parameter_dict
            self.pars["state_size"] = state_size
            self.pars["act_size"] = act_size
            self.pars["seed"] = seed
            self.initialize()
            return


    def initialize(self):

        self.net={}
        self.opt={}

        if self.pars["twin"]:
            self.twins=["_1", "_2"]
        else:
            self.twins=[""]

        for netname in ["act_loc", "act_tg"]:
            self.net[netname]=LinearNetwork(input_shape=self.pars["state_size"],
                                       lin_layers=self.pars["actor_layers"],
                                       output_shape= self.pars["act_size"],
                                       seed=self.pars["seed"], dropout_p=0.5)
        self.opt["act"] =  torch.optim.RMSprop(self.net["act_loc"].parameters(), lr=self.pars["lr_act"])


        for twin in self.twins:
            for netname in ["crit_loc", "crit_tg"]:
                self.net[netname+twin] = CriticNetwork(input_shape=self.pars["state_size"],
                                          lin_layers=self.pars["crit_layers"],
                                          output_shape=(1,),
                                          action_layer=self.pars["act_input"],
                                          action_shape=self.pars["act_size"],
                                          seed=self.pars["seed"], dropout_p=0.5)
            self.opt["crit"+twin] =torch.optim.RMSprop(self.net["crit_loc"+twin].parameters(), lr=self.pars["lr_crit"])

        if self.pars["erep_eps"]<1.0:
            self.mem = ExperienceReplayer(self.pars["erep_size"],
                                          wait_fill=self.pars["erep_fill"],
                                          default_prio=self.pars["erep_def_prio"],
                                          epsilon=self.pars["erep_eps"])
        else:
            self.mem = ExperienceReplayer(self.pars["erep_size"],
                                          wait_fill=self.pars["erep_fill"])

        self.train_cr_count =0
    def store_exp(self, exp):
        self.mem.store(exp)
        #self.mem.store(exp, exp["reward"])

    def get_action(self, state, noise_weight=0, last_noise_sample=None):
        action = self.net["act_loc"].forward_np(state)
        if noise_weight>0:
            noise_sample=0.1*(np.random.rand(len(action))-0.5)+0.9*last_noise_sample
            action = action/(1.0+noise_weight) + noise_sample*noise_weight/(1.0+noise_weight)
        else:
            noise_sample=None
        # should i add noise?
        return action, noise_sample

    def get_action_loc(self, state):
        action = self.net["act_loc"].forward_np(state)
        return action

    def get_action_tg(self, state):
        action = self.net["act_tg"].forward_np(state)
        return action


    def train(self):
        exp_batch =self.mem.draw(self.pars["batch"])
        if exp_batch is None:
            return

        states_t = torch.from_numpy(np.vstack([e["state"] for e in exp_batch])).float()
        next_states_t = torch.from_numpy(np.vstack([e["next_state"] for e in exp_batch])).float()
        rewards_t = torch.from_numpy(np.vstack([e["reward"] for e in exp_batch])).float()
        actions_t = torch.from_numpy(np.vstack([e["action"] for e in exp_batch])).float()
        dones_t = torch.from_numpy(np.vstack([np.uint8(e["done"]) for e in exp_batch])).float()


        actions_next = self.net["act_tg"].forward(next_states_t)
        # twin trick 1: add noise to inner action
        if self.pars["noise_in"]>0:
            n= self.pars["noise_in"]
            actions_next = torch.clamp(
                actions_next + torch.clamp(torch.randn(actions_next.shape) * n, -3*n, 3*n), -1.0, 1.0)


        # twin trick 2: avoid optimistic q: get the minimum of two targets.
        if self.pars["twin"]:
            q_target_next = torch.min(self.net["crit_tg_1"].forward(next_states_t, actions_next),
                                      self.net["crit_tg_2"].forward(next_states_t, actions_next))
        else:
            q_target_next = self.net["crit_tg"].forward(next_states_t, actions_next)

        q_target = (1.0-dones_t)*q_target_next*self.pars["gamma"] + rewards_t

        critic_error= 0
        for twin in self.twins:
            self.net["crit_loc"+twin].train()
            q_exp = self.net["crit_loc"+twin].forward(states_t, actions_t)
            critic_error+= torch.mean((q_target - q_exp) ** 2)
            self.net["crit_loc" + twin].zero_grad()
        # why not separate trainings?
        critic_error.backward()
        for twin in self.twins:
            self.opt["crit" + twin].step()

        self.train_cr_count = self.train_cr_count+1

        # twin trick 3: train critic more often then actor
        if self.train_cr_count >= self.pars["train_act_every"]:
            self.train_cr_count=0
            self.net["crit_loc"+self.twins[0]].eval()
            self.net["act_loc"].train()
            self.opt["act"].zero_grad()
            prev_actions = self.net["act_loc"].forward(states_t)
 #          Lazy trick: maximize expected return from critic but minimize actions.
            actor_q = torch.mean(prev_actions ** 2) * self.pars["lazy_actor"] \
                      - torch.mean(self.net["crit_loc" + self.twins[0]].forward(states_t, act=prev_actions))
            actor_q.backward()

            self.opt["act"].step()
            self.net["crit_loc"+self.twins[0]].train()
            self.net["act_loc"].eval()
            self.soft_update(self.net["act_loc"], self.net["act_tg"], self.pars["tau"])
        # ----------------------- update target networks ----------------------- #

        for tw in self.twins:
            self.soft_update(self.net["crit_loc"+tw], self.net["crit_tg"+tw], self.pars["tau"])

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

    def checkpoint(self, fname):
        for n in self.net:
            self.net[n].save_model(fname + "/"+n+".ckp", description="n.a.")
        with open(fname+"/pars.p", 'wb') as sf:
            pickle.dump(self.pars, sf)
        with open(fname+"/exp_rep", 'wb') as sf:
            pickle.dump((self.mem.memory, self.mem.prio_memory), sf)

    def load_checkpoint(self, fname):
        for n in self.net:
            self.net[n].load_model(fname + "/"+n+".ckp")
            self.net[n].train()
        with open(fname + "/exp_rep", 'rb') as sf:
            self.mem.memory,self.mem.prio_memory = pickle.load(sf)





