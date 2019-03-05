from collections import namedtuple


from pytorch_base_network import PyTorchBaseNetwork
from experience_replayer import ExperienceReplayer

class ddpgAgent:
    def __init__(self, state_shape=(784,), actor_layers=(128, 64),
                 critic_layers=(128, 64), seed=None, act_shape=(1,),
                 actin_layer=1, batch_size=64):
        self.actor_loc = PyTorchBaseNetwork(input_shape=state_shape, lin_layers=actor_layers, output_shape=act_shape,
                                        seed=seed)
        self.actor_tg = PyTorchBaseNetwork(input_shape=state_shape, lin_layers=actor_layers, output_shape=act_shape,
                                        seed=seed)
        self.critic_loc = PyTorchBaseNetwork(input_shape=state_shape, lin_layers=critic_layers, output_shape=(1,),
                                         actin_layer=1, actin_shape=act_shape, seed=seed)

        self.critic_tg = PyTorchBaseNetwork(input_shape=state_shape, lin_layers=critic_layers, output_shape=(1,),
                                         actin_layer=1, actin_shape=act_shape, seed=seed)
        self.batch_size=batch_size
        self.mem = ExperienceReplayer(10000)

        self.exptuple = namedtuple("experience", "state action reward new_state done")
    def store_exp(self, exp):
        self.mem.store(exp)

    def get_action(self, state):
        action = self.actor_loc.forward(state)
        # should i add noise?
        return action


    def train(self):
        exp_batch =self.mem.draw(self.batch_size)

        states = exp_batch.states
        actions = exp_batch.actions
        rewards = exp_batch.rewards
        next_states= exp_batch.next_states
        dones = exp_batch.dones

        actions_new = self.actor_loc.forward(next_states)

        q_target = (dones is False) *gamma* self.critic_tg.forward(states, act=actions_new) + rewards

        # minimize Q estimation error
        error = (self.critic_loc.forward(states, act=actions)- q_target)^2


        prev_actions = self.actor_loc.forward(states)


        # selecr action maximizing Q for current state.
        optimize = self.critic_loc.forward(states, act=prev_actions)








