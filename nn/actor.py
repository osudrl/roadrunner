import torch

import torch.nn as nn
import math as math

from nn.base import FFBase, LSTMBase

class Actor:
    def __init__(self,
                 latent: int,
                 action_dim: int,
                 bounded: bool,
                 learn_std: bool,
                 std: float):
        """The base class for actors. This class alone cannot be used for training, since it does
        not have complete model definition. normalize_state() and _base_forward() would be required
        to loaded to perform complete forward pass. Thus, child classes need to inherit
        this class with any model class in base.py.

        Args:
            latent (int): Input size for last action layer.
            action_dim (int): Action dim for last action layer.
            bounded (bool): Additional tanh activation after last layer.
            learn_std (bool): Option to learn std.
            std (float): Constant std.
        """
        self.action_dim = action_dim
        self.bounded    = bounded
        self.std        = torch.tensor(std)
        self.means      = nn.Linear(latent, action_dim)
        self.learn_std  = learn_std
        if self.learn_std:
            self.log_stds = nn.Linear(latent, action_dim)

    def _get_distrbution_params(self, input_state, update_normalization_param):
        """Perform a complete forward pass of the model and output mean/std for policy
        forward in stochastic_forward()

        Args:
            input_state (_type_): Model input
            update (bool): Option to update prenorm params. Defaults to False.

        Returns:
            mu: Model output, ie, mean of the distribution
            std: Optionally trainable param for distribution std. Default is constant.
        """
        state = self.normalize_state(input_state,
                                     update_normalization_param=update_normalization_param)
        latent = self._base_forward(state)
        mu = self.means(latent)
        if self.learn_std:
            std = torch.clamp(self.log_stds(latent), -3, 0.5).exp()
        else:
            std = self.std
        return mu, std

    def pdf(self, state):
        """Return Diagonal Normal Distribution object given mean/std from part of actor forward pass
        """
        mu, sd = self._get_distrbution_params(state, update_normalization_param=False)
        return torch.distributions.Normal(mu, sd)

    def log_prob(self, state, action):
        """Return the log probability of a distribution given state and action
        """
        log_prob = self.pdf(state=state).log_prob(action).sum(-1, keepdim=True)
        if self.bounded: # SAC, Appendix C, https://arxiv.org/pdf/1801.01290.pdf
            log_prob -= torch.log((1 - torch.tanh(state).pow(2)) + 1e-6).sum(-1, keepdim=True)
        return log_prob

    def actor_forward(self,
                state: torch.Tensor,
                deterministic=True,
                update_normalization_param=False,
                return_log_prob=False):
        """Perform actor forward in either deterministic or stochastic way, ie, inference/training.
        This function is default to inference mode.

        Args:
            state (torch.Tensor): Input to actor.
            deterministic (bool, optional): inference mode. Defaults to True.
            update_normalization_param (bool, optional): Toggle to update params. Defaults to False.
            return_log_prob (bool, optional): Toggle to return log probability. Defaults to False.

        Returns:
            Actions (deterministic or stochastic), with optional return on log probability.
        """
        mu, std = self._get_distrbution_params(state,
                                               update_normalization_param=update_normalization_param)
        if not deterministic or return_log_prob:
            # draw random samples for stochastic forward for training purpose
            dist = torch.distributions.Normal(mu, std)
            stochastic_action = dist.rsample()

        # Toggle bounded output or not
        if self.bounded:
            action = torch.tanh(mu) if deterministic else torch.tanh(stochastic_action)
        else:
            action = mu if deterministic else stochastic_action

        # Return log probability
        if return_log_prob:
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            if self.bounded:
                log_prob -= torch.log((1 - torch.tanh(action).pow(2)) + 1e-6).sum(-1, keepdim=True)
            return action, log_prob
        else:
            return action

class FFActor(FFBase, Actor):
    """
    A class inheriting from FF_Base and Actor
    which implements a feedforward stochastic policy.
    """
    def __init__(self,
                 obs_dim,
                 action_dim,
                 layers,
                 nonlinearity,
                 bounded,
                 learn_std,
                 std):

        # TODO, helei, make sure we have a actor example on what has to be included.
        # like the stuff below is useless to init, but has to inlcluded in order for saving checkpoint
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.layers = layers
        self.nonlinearity = nonlinearity
        self.bounded = bounded
        self.learn_std = learn_std
        self.std = std

        FFBase.__init__(self, in_dim=obs_dim, layers=layers, nonlinearity=nonlinearity)
        Actor.__init__(self,
                       latent=layers[-1],
                       action_dim=action_dim,
                       bounded=bounded,
                       learn_std=learn_std,
                       std=std)

    def forward(self, x, deterministic=True,
                update_normalization_param=False, return_log_prob=False):
        return self.actor_forward(x, deterministic=deterministic,
                                  update_normalization_param=update_normalization_param,
                                  return_log_prob=return_log_prob)

class LSTMActor(LSTMBase, Actor):
    """
    A class inheriting from LSTM_Base and Actor
    which implements a recurrent stochastic policy.
    """
    def __init__(self,
                 obs_dim,
                 action_dim,
                 layers,
                 bounded,
                 learn_std,
                 std):

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.layers = layers
        self.bounded = bounded
        self.learn_std = learn_std
        self.std = std

        LSTMBase.__init__(self, obs_dim, layers)
        Actor.__init__(self,
                       latent=layers[-1],
                       action_dim=action_dim,
                       bounded=bounded,
                       learn_std=learn_std,
                       std=std)

        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, x, deterministic=True,
                update_normalization_param=False, return_log_prob=False):
        return self.actor_forward(x, deterministic=deterministic,
                                  update_normalization_param=update_normalization_param,
                                  return_log_prob=return_log_prob)
