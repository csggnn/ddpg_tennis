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

        expt=exptuple(exp.vector_observations, )
        self.mem.store(exp)


    def train(self):
        exp_batch =self.mem.draw(self.batch_size)

