from collections import namedtuple

import torch
import torch.nn.functional as tnn_functional
from torch import nn


def is_iterable(element):
    try:
        iterator = iter(element)
    except TypeError:
        return False
    else:
        return True


def squeeze_iterable(element):
    """if input is an iterable with a single element, return its value. """
    in_iterable = is_iterable(element)
    out_value = element
    out_iterable = in_iterable
    if in_iterable:
        if len(element) == 1:
            # this is the case in which i have a single element iterable.
            out_iterable = False
            out_value = element[0]
    return [out_value, out_iterable]


class CriticNetwork(nn.Module):
    """
    Basic PyTorch Network composed of Convolutional and linear layers.
    The Convolutional linear layers are positioned before the linear layer by default. The network only supports linear
    layers for the moment.
    """

    version = (0, 7)

    def __init__(self, input_shape=(784,), lin_layers=(400, 300), output_shape=(10,), action_shape=(1,), action_layer=-1, dropout_p=0,
                 seed=None):
        """ Network architecture initialization according to linear and convolutional layers features """
        super().__init__()
        if seed is not None:
            self.seed=torch.manual_seed(seed)
        self.pars_tuple = namedtuple('ctitic_net_pars_tuple', 'input_shape lin_layers output_shape action_shape action_layer dropout_p')
        if isinstance(input_shape, str):
            ckp=input_shape
            self.load_model(ckp)
        else:
            self.pars = self.pars_tuple(input_shape,lin_layers,output_shape, action_shape, action_layer, dropout_p)
            self.initialise()

    def load_model(self, ckp):
        saved = torch.load(ckp)
        if saved["version"] != self.version:
            raise ImportError(
                "PyTorchBaseNetwork is now at version " + self.version + " but model was saved at version " + saved[
                    'version'])
        print("loading network " + saved["description"])
        self.pars = self.pars_tuple(**saved["pars"])
        self.initialise()
        self.load_state_dict(saved["state_dict"])

    def initialise(self):
        [input_shape, iterable_input] = squeeze_iterable(self.pars.input_shape)
        [output_shape, _] = squeeze_iterable(self.pars.output_shape)
        if self.pars.dropout_p>0:
            self.dropout=nn.Dropout(p=self.pars.dropout_p)

        prev_layer_n = input_shape
        self.fc_layers = nn.ModuleList()
        self.act_layer=self.pars.action_layer
        if self.act_layer<0:
            self.act_layer = len(self.pars.lin_layers)-self.act_layer
        if (self.act_layer<0)|(self.act_layer>len(self.pars.lin_layers)):
            self.act_layer=len(self.pars.lin_layers)-1


        for curr_layer_i in range(len(self.pars.lin_layers)):
            curr_layer_n = self.pars.lin_layers[curr_layer_i]
            if curr_layer_i == self.act_layer:
                [act_n, _] = squeeze_iterable(self.pars.action_shape)
                prev_layer_n = prev_layer_n+act_n # action is an additional INPUT to this layer
            self.fc_layers.append(nn.Linear(prev_layer_n, curr_layer_n))
            self.fc_layers[curr_layer_i].weight.data.uniform_(-1.0/curr_layer_n, 1.0/curr_layer_n)
            self.fc_layers[curr_layer_i].bias.data.uniform_(-1.0 / curr_layer_n, 1.0 / curr_layer_n)
            prev_layer_n = curr_layer_n
        self.out_layer = nn.Linear(prev_layer_n, output_shape)
        self.out_layer.weight.data.uniform_(-0.03, 0.03)
        self.out_layer.bias.data.uniform_(-0.03, 0.03)

    def forward(self, x, act):
        """ Forward pass through the network, returns the output logits

        :param x(torch.FloatTensor): input or set of inputs to be processed by the network
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("x should be a Tensor but is of type " + str(type(x)))

        if not isinstance(x, torch.FloatTensor):
            raise TypeError("x should be a Tensor of Float but is of type " + str(x.type()))

        for fc_layer_i in range(len(self.fc_layers)):
            fc_layer=self.fc_layers[fc_layer_i]
            if fc_layer_i==self.act_layer:
                x=fc_layer(torch.cat((x,act), 1))
            else:
                x = fc_layer(x)
            x = tnn_functional.leaky_relu(x)
            if self.pars.dropout_p > 0:
                x = self.dropout(x)
        x = self.out_layer(x)
        return x

    def save_model(self, checkpoint_file, description=None):
        tosave = {"version": self.version, "pars": self.pars._asdict(),
                  "state_dict": self.state_dict(), "description": description}
        torch.save(tosave, checkpoint_file)


