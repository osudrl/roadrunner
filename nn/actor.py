import torch

import torch.nn as nn
import math as math

from nn.base import FFBase, LSTMBase, GRUBase, MixBase, LSTM_Concat_CNN_Base, LSTM_Add_CNN_Base
from nn.base import FFConcatBase, FFLSTMConcatBase, LocomotionNet, LocomotionNetV2, LocomotionNetV3

class Actor:
    def __init__(self,
                 latent: int,
                 action_dim: int,
                 bounded: bool,
                 learn_std: bool,
                 std: float,
                 use_base_model: bool = False):
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
            use_base_model (bool, optional): Option to use use_base_model. Defaults to False.
        """
        self.action_dim = action_dim
        self.bounded    = bounded
        self.std        = nn.Parameter(torch.tensor(std), requires_grad=False) # Compatible to transfer to GPU
        self.means      = nn.Linear(latent, action_dim)
        self.learn_std  = learn_std
        self.use_base_model   = use_base_model
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
        if self.use_base_model:
            mu = self._base_forward(state)
        else:
            # Regular forward pass
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

class MixActor(MixBase, Actor):
    """
    A class inheriting from Mix_Base and Actor
    which implements a recurrent + FF stochastic policy.
    """
    def __init__(self,
                 obs_dim,
                 state_dim,
                 nonstate_dim,
                 action_dim,
                 lstm_layers,
                 ff_layers,
                 bounded,
                 learn_std,
                 std,
                 nonstate_encoder_dim,
                 nonstate_encoder_on):

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.nonstate_dim = nonstate_dim
        self.action_dim = action_dim
        self.lstm_layers = lstm_layers
        self.ff_layers = ff_layers
        self.bounded = bounded
        self.learn_std = learn_std
        self.std = std
        self.nonstate_encoder_dim = nonstate_encoder_dim
        self.nonstate_encoder_on = nonstate_encoder_on

        MixBase.__init__(self,
                          in_dim=obs_dim,
                          state_dim=state_dim,
                          nonstate_dim=nonstate_dim,
                          lstm_layers=lstm_layers,
                          ff_layers=ff_layers,
                          nonstate_encoder_dim=nonstate_encoder_dim,
                          nonstate_encoder_on=nonstate_encoder_on)
        Actor.__init__(self,
                       latent=ff_layers[-1],
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

    def latent_space(self, x):
        return self._latent_space_forward(x)

    def load_lstm_module(self, pretrained_actor, freeze_lstm = True):
        """Load LSTM module for Mix Actor

        Args:
            pretrained_actor: Previously trained actor
            freeze_lstm (bool, optional): Freeze the weights/bias in LSTM. Defaults to True.
        """
        for param_key in pretrained_actor.state_dict():
            if "lstm" in param_key:
                self.state_dict()[param_key].copy_(pretrained_actor.state_dict()[param_key])
        if freeze_lstm:
            for name, param in self.named_parameters():
                if "lstm" in name:
                    param.requires_grad = False

class GRUActor(GRUBase, Actor):
    """
    A class inheriting from GRU_Base and Actor
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

        GRUBase.__init__(self, obs_dim, layers)
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

class CNNLSTMActor(LSTM_Concat_CNN_Base, Actor):
    """
    A class inheriting from LSTM_Concat_CNN_Base and Actor
    which implements a recurrent stochastic policy.
    """
    def __init__(self,
                 obs_dim,
                 action_dim,
                 state_dim,
                 layers,
                 bounded,
                 learn_std,
                 std,
                 image_shape,
                 image_channel):

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.layers = layers
        self.bounded = bounded
        self.learn_std = learn_std
        self.std = std
        self.image_shape = image_shape
        self.image_channel = image_channel

        LSTM_Concat_CNN_Base.__init__(self, state_dim=state_dim, layers=layers, 
                                      image_shape=image_shape, image_channel=image_channel)
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

class CNNAddLSTMActor(LSTM_Add_CNN_Base, Actor):
    """
    A class inheriting from LSTM_Add_CNN_Base and Actor
    which implements a recurrent stochastic policy.
    """
    def __init__(self,
                 obs_dim,
                 action_dim,
                 state_dim,
                 bounded,
                 learn_std,
                 std,
                 image_shape,
                 image_channel,
                 base_actor_layer):

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.bounded = bounded
        self.learn_std = learn_std
        self.std = std
        self.image_shape = image_shape
        self.image_channel = image_channel
        self.base_actor_layer = base_actor_layer

        LSTM_Add_CNN_Base.__init__(self, state_dim=state_dim, base_actor_layers=base_actor_layer)
        Actor.__init__(self,
                       latent=base_actor_layer[-1],
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

    def load_base_actor(self, old_base_actor_dict):
        """Load base actor params

        Args:
            base_actor_dict: Previously trained actor
            freeze (bool, optional): Freeze the weights/bias in LSTM. Defaults to True.
        """
        for key in self.base_actor_lstm.state_dict():
            old_key = key.replace("base_actor_lstm.", "")
            self.base_actor_lstm.state_dict()[key].copy_(old_base_actor_dict[old_key])

        for key in self.means.state_dict():
            old_key = "means."+key
            self.means.state_dict()[key].copy_(old_base_actor_dict[old_key])

class FFConcatActor(FFConcatBase, Actor):
    """
    A class inheriting from FF_Base and Actor
    which implements a feedforward stochastic policy.
    """
    def __init__(self,
                 obs_dim,
                 state_dim,
                 map_dim,
                 action_dim,
                 state_layers,
                 map_layers,
                 concat_layers,
                 nonlinearity,
                 bounded,
                 learn_std,
                 std,
                 use_cnn):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_layers = state_layers
        self.map_layers = map_layers
        self.concat_layers = concat_layers
        self.nonlinearity = nonlinearity
        self.bounded = bounded
        self.learn_std = learn_std
        self.std = std
        self.use_cnn = use_cnn
        self.is_recurrent = False

        FFConcatBase.__init__(self, in_dim=obs_dim, state_dim=state_dim, map_dim=map_dim,
                              state_layers=state_layers, map_layers=map_layers,
                              concat_layers=concat_layers, nonlinearity=nonlinearity,
                              use_cnn=use_cnn)
        Actor.__init__(self,
                       latent=concat_layers[-1],
                       action_dim=action_dim,
                       bounded=bounded,
                       learn_std=learn_std,
                       std=std)

    def forward(self, x, deterministic=True,
                update_normalization_param=False, return_log_prob=False):
        return self.actor_forward(x, deterministic=deterministic,
                                  update_normalization_param=update_normalization_param,
                                  return_log_prob=return_log_prob)

    def latent_forward(self, x):
        return self._latent_space_forward(x)

    def load_base_model(self, base_model_dict):
        return super().load_base_model(base_model_dict)

class FFLSTMConcatActor(FFLSTMConcatBase, Actor):
    """
    A class inheriting from FF_Base and Actor
    which implements a feedforward stochastic policy.
    """
    def __init__(self,
                 obs_dim,
                 state_dim,
                 map_dim,
                 action_dim,
                 state_layers,
                 map_layers,
                 concat_layers,
                 nonlinearity,
                 bounded,
                 learn_std,
                 std,
                 use_cnn):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_layers = state_layers
        self.map_layers = map_layers
        self.concat_layers = concat_layers
        self.nonlinearity = nonlinearity
        self.bounded = bounded
        self.learn_std = learn_std
        self.std = std
        self.use_cnn = use_cnn
        self.is_recurrent = True

        FFLSTMConcatBase.__init__(self, in_dim=obs_dim, state_dim=state_dim, map_dim=map_dim,
                                  state_layers=state_layers, map_layers=map_layers,
                                  concat_layers=concat_layers, nonlinearity=nonlinearity,
                                  use_cnn=use_cnn)
        Actor.__init__(self,
                       latent=concat_layers[-1],
                       action_dim=action_dim,
                       bounded=bounded,
                       learn_std=learn_std,
                       std=std)
        
        self.init_hidden_state()

    def forward(self, x, deterministic=True,
                update_normalization_param=False, return_log_prob=False):
        return self.actor_forward(x, deterministic=deterministic,
                                  update_normalization_param=update_normalization_param,
                                  return_log_prob=return_log_prob)

    def latent_forward(self, x):
        return self._latent_space_forward(x)

class ResidualActor(LocomotionNet, Actor):
    """
    A class inheriting from LocomotioNet and Actor. It implements a residual policy on top of a 
    base policy, and it supports backprop throughout entire model once the base model loaded.
    """
    def __init__(self,
                 obs_dim,
                 state_dim,
                 map_dim,
                 action_dim,
                 base_action_dim,
                 state_layers,
                 map_layers,
                 concat_layers,
                 base_layers,
                 nonlinearity,
                 bounded,
                 learn_std,
                 std,
                 use_cnn,
                 link_base_model):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.base_action_dim = base_action_dim
        self.state_layers = state_layers
        self.map_layers = map_layers
        self.concat_layers = concat_layers
        self.base_layer = base_layers
        self.nonlinearity = nonlinearity
        self.bounded = bounded
        self.learn_std = learn_std
        self.std = std
        self.use_cnn = use_cnn
        self.link_base_model = link_base_model
        self.is_recurrent = True

        LocomotionNet.__init__(self, in_dim=obs_dim, state_dim=state_dim, map_dim=map_dim,
                                  state_layers=state_layers, map_layers=map_layers,
                                  concat_layers=concat_layers, nonlinearity=nonlinearity,
                                  use_cnn=use_cnn, base_layers=base_layers, action_dim=action_dim,
                                  link_base_model=link_base_model, base_action_dim=base_action_dim)
        Actor.__init__(self,
                       latent=concat_layers[-1],
                       action_dim=action_dim,
                       bounded=bounded,
                       learn_std=learn_std,
                       std=std,
                       use_base_model=True)
        
        self.init_hidden_state()

    def forward(self, x, deterministic=True,
                update_normalization_param=False, return_log_prob=False):
        return self.actor_forward(x, deterministic=deterministic,
                                  update_normalization_param=update_normalization_param,
                                  return_log_prob=return_log_prob)

    def load_encoder_patch(self):
        return super().load_encoder_patch()

    def get_perception_feature(self):
        return self._get_perception_feature()

class ResidualActorV2(LocomotionNetV2, Actor):
    """
    A class inheriting from LocomotioNet and Actor. It implements a residual policy on top of a 
    base policy, and it supports backprop throughout entire model once the base model loaded.
    """
    def __init__(self,
                 obs_dim,
                 state_dim,
                 map_dim,
                 action_dim,
                 map_input_layer_dim,
                 state_layers,
                 map_feature_layers,
                 concat_layers,
                 base_layers,
                 nonlinearity,
                 bounded,
                 learn_std,
                 std,
                 use_cnn,
                 link_base_model):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_layers = state_layers
        self.map_feature_layers = map_feature_layers
        self.map_input_layer_dim = map_input_layer_dim
        self.concat_layers = concat_layers
        self.base_layer = base_layers
        self.nonlinearity = nonlinearity
        self.bounded = bounded
        self.learn_std = learn_std
        self.std = std
        self.use_cnn = use_cnn
        self.link_base_model = link_base_model
        self.is_recurrent = True

        LocomotionNetV2.__init__(self,
                               obs_dim=obs_dim, state_dim=state_dim, map_dim=map_dim, map_input_layer_dim=map_input_layer_dim,
                               state_layers=state_layers, map_feature_layers=map_feature_layers,
                               concat_layers=concat_layers, nonlinearity=nonlinearity,
                               use_cnn=use_cnn, base_layers=base_layers, action_dim=action_dim,
                               link_base_model=link_base_model)
        Actor.__init__(self,
                       latent=concat_layers[-1],
                       action_dim=action_dim,
                       bounded=bounded,
                       learn_std=learn_std,
                       std=std,
                       use_base_model=True)
        
        self.init_hidden_state()

    def forward(self, x, deterministic=True,
                update_normalization_param=False, return_log_prob=False):
        return self.actor_forward(x, deterministic=deterministic,
                                  update_normalization_param=update_normalization_param,
                                  return_log_prob=return_log_prob)

    def get_perception_feature(self):
        return self._get_perception_feature()

class V3(LocomotionNetV3, Actor):
    def __init__(self,
                 obs_dim,
                 state_dim,
                 map_dim,
                 action_dim,
                 state_layers,
                 map_layers,
                 concat_layers,
                 nonlinearity,
                 bounded,
                 learn_std,
                 std,
                 use_cnn):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_layers = state_layers
        self.map_layers = map_layers
        self.concat_layers = concat_layers
        self.nonlinearity = nonlinearity
        self.bounded = bounded
        self.learn_std = learn_std
        self.std = std
        self.use_cnn = use_cnn
        self.is_recurrent = True

        LocomotionNetV3.__init__(self, obs_dim=obs_dim, state_dim=state_dim, map_dim=map_dim,
                                  state_layers=state_layers, map_layers=map_layers,
                                  concat_layers=concat_layers, nonlinearity=nonlinearity,
                                  use_cnn=use_cnn, action_dim=action_dim)
        Actor.__init__(self,
                       latent=concat_layers[-1],
                       action_dim=action_dim,
                       bounded=bounded,
                       learn_std=learn_std,
                       std=std,
                       use_base_model=False)
        
        self.init_hidden_state()

    def forward(self, x, deterministic=True,
                update_normalization_param=False, return_log_prob=False):
        return self.actor_forward(x, deterministic=deterministic,
                                  update_normalization_param=update_normalization_param,
                                  return_log_prob=return_log_prob)

    def get_perception_feature(self):
        return self._get_perception_feature()