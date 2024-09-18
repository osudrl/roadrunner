import torch.nn as nn

from nn.actor import Actor
from nn.base import *

class ResidualActor(Net, Actor):
    def __init__(self,
                 bounded, std,
                 base_in_dim, base_out_dim, base_layer,
                 res_state_in_dim, res_state_out_dim, res_state_layer,
                 res_perception_in_dim, res_perception_out_dim, res_perception_layer,
                 res_concat_layer, res_out_dim):
        """
        init blind policy layers
        load blind layers
        init residual layers
        load previous residual layers
        """
        super(ResidualActor, self).__init__()
        self.base_in_dim = base_in_dim
        self.base_out_dim = base_out_dim
        self.base_module = LSTMBase(in_dim=base_in_dim, layers=base_layer)
        self.base_mean   = nn.Linear(base_layer[-1], base_out_dim)

        self.res_state_in_dim = res_state_in_dim
        self.res_perception_in_dim = res_perception_in_dim
        self.res_state_layer = FFBase(in_dim=res_state_in_dim,
                                      layers=res_state_layer,
                                      nonlinearity='relu')
        self.res_perception_layer = FFBase(in_dim=res_perception_in_dim,
                                           layers=res_perception_layer,
                                           nonlinearity='relu')
        self.res_concat_layer = LSTMBase(in_dim=res_state_layer[-1]+res_perception_layer[-1]+base_out_dim,
                                         layers=res_concat_layer)
        self.res_mean_layer = nn.Linear(res_concat_layer[-1], res_out_dim)

        self.is_recurrent = True
        self.init_hidden_state()

        # Initialze actor
        Actor.__init__(self, latent=10, action_dim=res_out_dim, bounded=bounded, std=std,
                       learn_std=False, use_base_model=True)

    def forward(self, state, deterministic=True, update_normalization_param=False, return_log_prob=False):
        return self.actor_forward(state, deterministic=deterministic,
                                     update_normalization_param=update_normalization_param,
                                     return_log_prob=return_log_prob)

    def init_hidden_state(self, batch_size=1):
        self.res_concat_layer.init_hidden_state(batch_size=batch_size)
        self.base_module.init_hidden_state(batch_size=batch_size)

    def _base_forward(self, state):
        """
        seperate blind state and residual state
        do seprate inference on blind module and residual module
        add blind state and residual state
        [batch size, num steps, num states]
        """
        # Inference on base module
        a_base = self.base_mean(self.base_module._base_forward(state[...,:self.base_in_dim]))
        # Split base module input out of state
        state = state[...,self.base_in_dim:]
        # Inference on residual module
        s = self.res_state_layer._base_forward(state[...,:self.res_state_in_dim])
        p = self.res_perception_layer._base_forward(state[...,self.res_state_in_dim:])
        # Concat and add residual module output to base module output
        c = torch.cat((s, p, a_base), dim=-1)
        a_res = self.res_mean_layer(self.res_concat_layer._base_forward(c))
        a_res[...,:self.base_out_dim] += a_base
        return a_res

    def load_base(self, state_dict):
        """
        load blind module
        """
        pass

    def load_res(self, state_dict):
        """
        load residual module
        """
        pass
