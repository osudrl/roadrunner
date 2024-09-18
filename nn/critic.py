import torch
import torch.nn as nn
import torch.nn.functional as F
import math as math

from nn.base import FFBase, LSTMBase

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
    def __init__(self, input_dim, layers):
        self.input_dim = input_dim
        self.layers = layers

        LSTMBase.__init__(self, in_dim=input_dim, layers=layers)
        Critic.__init__(self, latent=layers[-1])
        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, state, update_normalization_param=False):
        return self.critic_forward(state, update_norm=update_normalization_param)
