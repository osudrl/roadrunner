import torch
import torch.nn as nn
import torch.nn.functional as F
import math as math

from nn.base import FFBase, LSTMBase, GRUBase, MixBase, LSTM_Concat_CNN_Base, LSTM_Add_CNN_Base

class Critic:
    def __init__(self, latent: int):
        """The base class for Value functions.

        Args:
            latent (int): Input size of last layer of Critic
        """
        self.critic_last_layer = nn.Linear(latent, 1)

    def critic_forward(self, state, update_norm=False):
        """Forward pass output value function result.

        Args:
            state (_type_): Critic input
            update_norm (bool, optional): Option to update normalization params. Defaults to False.

        Returns:
            float: Value of critic prediction
        """
        state = self.normalize_state(state, update_normalization_param=update_norm)
        x = self._base_forward(state)
        return self.critic_last_layer(x)

class FFCritic(FFBase, Critic):
    """
    A class inheriting from FF_Base and Critic
    which implements a feedforward value function.
    """
    def __init__(self, input_dim, layers):
        self.input_dim = input_dim
        self.layers = layers
        
        FFBase.__init__(self, in_dim=input_dim, layers=layers, nonlinearity='relu')
        Critic.__init__(self, latent=layers[-1])

    def forward(self, state, update_normalization_param=False):
        return self.critic_forward(state, update_norm=update_normalization_param)

class LSTMCritic(LSTMBase, Critic):
    """
    A class inheriting from LSTM_Base and Critic
    which implements a recurrent value function.
    """
    def __init__(self, input_dim, layers, use_privilege_critic=False):
        self.input_dim = input_dim
        self.layers = layers
        self.use_privilege_critic = use_privilege_critic

        LSTMBase.__init__(self, in_dim=input_dim, layers=layers)
        Critic.__init__(self, latent=layers[-1])
        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, state, update_normalization_param=False):
        return self.critic_forward(state, update_norm=update_normalization_param)

class GRUCritic(GRUBase, Critic):
    """
    A class inheriting from GRU_Base and V
    which implements a recurrent value function.
    """
    def __init__(self, input_dim, layers):
        self.input_dim = input_dim
        self.layers = layers

        GRUBase.__init__(self, in_dim=input_dim, layers=layers)
        Critic.__init__(self, latent=layers[-1])
        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, state, update_normalization_param=False):
        return self.critic_forward(state, update_norm=update_normalization_param)

class MixCritic(MixBase, Critic):
    def __init__(self,
                 input_dim, 
                 state_dim, 
                 nonstate_dim, 
                 lstm_layers, 
                 ff_layers,
                 nonstate_encoder_dim, 
                 nonstate_encoder_on):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.nonstate_dim = nonstate_dim
        self.lstm_layers = lstm_layers
        self.ff_layers = ff_layers
        self.nonstate_dim = nonstate_dim
        self.nonstate_encoder_on = nonstate_encoder_on

        MixBase.__init__(self,
                          in_dim=input_dim, 
                          state_dim=state_dim, 
                          nonstate_dim=nonstate_dim, 
                          lstm_layers=lstm_layers, 
                          ff_layers=ff_layers, 
                          nonstate_encoder_dim=nonstate_encoder_dim,
                          nonstate_encoder_on=nonstate_encoder_on)
        Critic.__init__(self, latent=ff_layers[-1])
        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, state, update_normalization_param=False):
        return self.critic_forward(state, update_norm=update_normalization_param)

class CNNLSTMCritic(LSTM_Concat_CNN_Base, Critic):
    """
    A class inheriting from LSTM_Concat_CNN_Base and Actor
    which implements a recurrent stochastic policy.
    """
    def __init__(self,
                 obs_dim,
                 state_dim,
                 layers,
                 bounded,
                 learn_std,
                 std,
                 image_shape,
                 image_channel):

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.layers = layers
        self.bounded = bounded
        self.learn_std = learn_std
        self.std = std
        self.image_shape = image_shape
        self.image_channel = image_channel

        LSTM_Concat_CNN_Base.__init__(self, state_dim=state_dim, layers=layers, 
                                      image_shape=image_shape, image_channel=image_channel)
        Critic.__init__(self, latent=layers[-1])
        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, state, update_normalization_param=False):
        return self.critic_forward(state, update_norm=update_normalization_param)

class CNNAddLSTMCritic(LSTM_Add_CNN_Base, Critic):
    """
    A class inheriting from LSTM_Add_CNN_Base and Actor
    which implements a recurrent stochastic policy.
    """
    def __init__(self,
                 obs_dim,
                 state_dim,
                 image_shape,
                 image_channel,
                 base_actor_layer):

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.image_shape = image_shape
        self.image_channel = image_channel
        self.base_actor_layer = base_actor_layer

        LSTM_Add_CNN_Base.__init__(self, state_dim=state_dim, base_actor_layers=base_actor_layer)
        Critic.__init__(self, latent=base_actor_layer[-1])

        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, state, update_normalization_param=False):
        return self.critic_forward(state, update_norm=update_normalization_param)
