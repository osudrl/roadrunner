import torch
import torch.nn as nn
from .resnet import CustomResNet, BasicBlock
from .encoder import Tarsus_Predictor_v1, Tarsus_Predictor_v2

def normc_fn(m):
    """
    This function multiplies the weights of a pytorch linear layer by a small
    number so that outputs early in training are close to zero, which means
    that gradients are larger in magnitude. This means a richer gradient signal
    is propagated back and speeds up learning (probably).
    """
    if m.__class__.__name__.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def create_layers(layer_fn, input_dim, layer_sizes):
    """
    This function creates a pytorch modulelist and appends
    pytorch modules like nn.Linear or nn.LSTMCell passed
    in through the layer_fn argument, using the sizes
    specified in the layer_sizes list.
    """
    ret = nn.ModuleList()
    ret += [layer_fn(input_dim, layer_sizes[0])]
    for i in range(len(layer_sizes)-1):
        ret += [layer_fn(layer_sizes[i], layer_sizes[i+1])]
    return ret

def get_activation(act_name):
    try:
        return getattr(torch, act_name)
    except:
        raise RuntimeError(f"Not implemented activation {act_name}. Please add in.")

def count_parameters(model):
    num = 0
    for p in model.parameters():
        if p.requires_grad:
            num += p.numel()
    return num


class Net(nn.Module):
    """
    The base class which all policy networks inherit from. It includes methods
    for normalizing states.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.is_recurrent = False

        # Params for nn-input normalization
        self.welford_state_mean = torch.zeros(1)
        self.welford_state_mean_diff = torch.ones(1)
        self.welford_state_n = 1

    def initialize_parameters(self):
        self.apply(normc_fn)
        if hasattr(self, 'critic_last_layer'):
            self.critic_last_layer.weight.data.mul_(0.01)

    def _base_forward(self, x):
        raise NotImplementedError

    def normalize_state(self, state: torch.Tensor, update_normalization_param=True):
        """
        Use Welford's algorithm to normalize a state, and optionally update the statistics
        for normalizing states using the new state, online.
        """

        if self.welford_state_n == 1:
            self.welford_state_mean = torch.zeros(state.size(-1)).to(state.device)
            self.welford_state_mean_diff = torch.ones(state.size(-1)).to(state.device)

        if update_normalization_param:
            if len(state.size()) == 1:  # if we get a single state vector
                state_old = self.welford_state_mean
                self.welford_state_mean += (state - state_old) / self.welford_state_n
                self.welford_state_mean_diff += (state - state_old) * (state - state_old)
                self.welford_state_n += 1
            else:
                raise RuntimeError  # this really should not happen
        return (state - self.welford_state_mean) / torch.sqrt(self.welford_state_mean_diff / self.welford_state_n)

    def copy_normalizer_stats(self, net):
        self.welford_state_mean      = net.welford_state_mean
        self.welford_state_mean_diff = net.welford_state_mean_diff
        self.welford_state_n         = net.welford_state_n

class FFBase(Net):
    """
    The base class for feedforward networks.
    """
    def __init__(self, in_dim, layers, nonlinearity='tanh'):
        super(FFBase, self).__init__()
        self.layers       = create_layers(nn.Linear, in_dim, layers)
        self.nonlinearity = get_activation(nonlinearity)

    def _base_forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = self.nonlinearity(layer(x))
        return x

class LSTMBase_(Net):
    """
    The base class for LSTM networks.
    """
    def __init__(self, in_dim, layers):
        super(LSTMBase, self).__init__()
        self.layers = layers
        for layer in self.layers:
            assert layer == self.layers[0], "LSTMBase only supports layers of equal size"
        self.lstm = nn.LSTM(in_dim, self.layers[0], len(self.layers))
        self.init_hidden_state()

    def init_hidden_state(self, **kwargs):
        self.hx = None

    def get_hidden_state(self):
        return self.hx[0], self.hx[1]

    def set_hidden_state(self, hidden, cells):
        self.hx = (hidden, cells)

    def _base_forward(self, x):
        dims = len(x.size())
        if dims == 1:  # if we get a single timestep (if not, assume we got a batch of single timesteps)
            x = x.view(1, 1, -1)
        elif dims == 3:
            self.init_hidden_state()

        x, self.hx = self.lstm(x, self.hx)

        if dims == 1:
            x = x.view(-1)

        return x

class LSTMBase(Net):
    """
    The base class for LSTM networks.
    """
    def __init__(self, in_dim, layers):
        super(LSTMBase, self).__init__()
        self.layers = create_layers(nn.LSTMCell, in_dim, layers)

    def init_hidden_state(self, batch_size=1):
        self.hidden = [torch.zeros(batch_size, l.hidden_size).to(next(self.layers.parameters()).device) for l in self.layers]
        self.cells  = [torch.zeros(batch_size, l.hidden_size).to(next(self.layers.parameters()).device) for l in self.layers]

    def get_hidden_state(self):
        hidden_numpy = [self.hidden[l].numpy() for l in range(len(self.layers))]
        cells_numpy = [self.cells[l].numpy() for l in range(len(self.layers))]

        return hidden_numpy, cells_numpy

    def set_hidden_state(self, hidden, cells):
        self.hidden = torch.FloatTensor(hidden)
        self.cells = torch.FloatTensor(cells)

    def _base_forward(self, x):
        dims = len(x.size())

        if dims == 3:  # if we get a batch of trajectories
            self.init_hidden_state(batch_size=x.size(1))

            y = []
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]

                y.append(x_t)
            x = torch.stack([x_t for x_t in y])
        else:
            if dims == 1:  # if we get a single timestep (if not, assume we got a batch of single timesteps)
                x = x.view(1, -1)

            for idx, layer in enumerate(self.layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]

            if dims == 1:
                x = x.view(-1)
        return x

class MixBase(Net):
    def __init__(self,
                 in_dim,
                 state_dim,
                 nonstate_dim,
                 lstm_layers,
                 ff_layers,
                 nonstate_encoder_dim,
                 nonlinearity='relu',
                 nonstate_encoder_on=True):
        """
        Base class for mixing LSTM and FF for actor.
        state1 -> LSTM ->
                          FF2 -> output
        state2 -> FF1   ->

        Args:
            in_dim (_type_): Model input size
            state_dim (_type_): Sub-input size to model for LSTM
            nonstate_dim (_type_): sub-input size for FF1
            lstm_layers (_type_): LSTM layers
            ff_layers (_type_): FF2 layers
            nonstate_encoder_dim (_type_): Layer for FF1
            nonlinearity (_type_, optional): Activation for FF1 and FF2.
                                             Defaults to torch.nn.functional.relu.
            nonstate_encoder_on (bool, optional): Use FF1 or not. Defaults to True.
        """
        assert state_dim + nonstate_dim == in_dim, "State and Nonstate Dimension Mismatch"
        super(MixBase, self).__init__()
        self.nonlinearity = nonlinearity
        self.state_dim = state_dim
        self.nonstate_encoder_on = nonstate_encoder_on

        # Construct model
        if nonstate_encoder_on: # use a FF encoder to encode commands
            nonstate_ft_dim = nonstate_encoder_dim # single layer encoder
            self.nonstate_encoder = FFBase(in_dim=nonstate_dim,
                                           layers=[nonstate_dim, nonstate_ft_dim],
                                           nonlinearity='relu')
        else:
            nonstate_ft_dim = nonstate_dim
        self.lstm = LSTMBase(in_dim=state_dim, layers=lstm_layers)
        self.ff = FFBase(in_dim=lstm_layers[-1]+nonstate_ft_dim,
                         layers=ff_layers,
                         nonlinearity='relu')
        self.latent_space = FFBase(in_dim=lstm_layers[-1], layers=ff_layers)

    def init_hidden_state(self, batch_size=1):
        self.lstm.init_hidden_state(batch_size=batch_size)

    def get_hidden_state(self):
        return self.lstm.get_hidden_state()

    def set_hidden_state(self, hidden, cells):
        self.lstm.set_hidden_state(hidden=hidden, cells=cells)

    def _base_forward(self, x):
        size = x.size()
        dims = len(size)
        if dims == 3: # for optimizaton with batch of trajectories
            state = x[:,:,:self.state_dim]
            nonstate = x[:,:,self.state_dim:]
            lstm_feature = self.lstm._base_forward(state)
            if self.nonstate_encoder_on:
                nonstate = self.nonstate_encoder._base_forward(nonstate)
            ff_input = torch.cat((lstm_feature, nonstate), dim=2)
        elif dims == 1: # for model forward
            state = x[:self.state_dim]
            nonstate = x[self.state_dim:]
            lstm_feature = self.lstm._base_forward(state)
            if self.nonstate_encoder_on:
                nonstate = self.nonstate_encoder._base_forward(nonstate)
            ff_input = torch.cat((lstm_feature, nonstate))
        ff_feature = self.ff._base_forward(ff_input)
        return ff_feature

    def _latent_space_forward(self, x):
        lstm_feature = self.lstm._base_forward(x)
        x = self.latent_space._base_forward(lstm_feature)
        return x

class GRUBase(Net):
    """
    The base class for GRU networks.
    NOTE: not maintained nor tested.
    """
    def __init__(self, in_dim, layers):
        super(GRUBase, self).__init__()
        self.layers = create_layers(nn.GRUCell, in_dim, layers)

    def init_hidden_state(self, batch_size=1):
        self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.layers]

    def _base_forward(self, x):
        dims = len(x.size())

        if dims == 3:  # if we get a batch of trajectories
            self.init_hidden_state(batch_size=x.size(1))

            y = []
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.layers):
                    h = self.hidden[idx]
                    self.hidden[idx] = layer(x_t, h)
                    x_t = self.hidden[idx]
                y.append(x_t)
            x = torch.stack([x_t for x_t in y])
        else:
            if dims == 1:  # if we get a single timestep (if not, assume we got a batch of single timesteps)
                x = x.view(1, -1)

            for idx, layer in enumerate(self.layers):
                h = self.hidden[idx]
                self.hidden[idx] = layer(x, h)
                x = self.hidden[idx]

            if dims == 1:
                x = x.view(-1)
        return x

class LSTM_Concat_CNN_Base(LSTMBase):
    """
    A generic class that concat(output of CNN, raw robot states) as LSTM inputs
    CNN often needs redefine, so here constructs the CNN entirely than inheritance.
    Inputs: Concat array with the flattened image indexed in the end
    """
    def __init__(self, state_dim, layers, image_shape=(32,32), image_channel=1):
        self.perception_in_size  = int(image_shape[0] * image_shape[1])
        self.perception_out_size = 16 # make sure matches the CNN output size
        self.img_width   = image_shape[0]
        self.img_height  = image_shape[1]
        self.img_channel = image_channel
        LSTMBase.__init__(self, in_dim=state_dim+self.perception_out_size, layers=layers)

        #NOTE: Taken from before, the following part will be modified per experiment
        self.cnn_net = nn.Sequential(
            nn.Conv2d(1,4,7,2,0), #bs , 4 , 10 , 10
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, 5, 2, 0), #bs , 8 5 5
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1, 0), # bs 16 1 1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

    def _base_forward(self, state):
        size = state.size()
        dims = len(size)

        if dims == 3: # for optimizaton with batch of trajectories
            # [num steps per episode, num episode, num state per step]
            num_traj = size[1]
            robot_state = state[:,:,:-self.perception_in_size]
            cnn_feature = []
            for traj_idx in range(num_traj):
                traj_data = state[: , traj_idx, -self.perception_in_size:]
                # print("batch dat shae ", traj_data.shape)
                perception_state = traj_data[-self.perception_in_size:].reshape(-1, self.img_channel, self.img_width, self.img_height)
                # print("per size ", perception_state.shape, "out size ", self.cnn_net.forward(perception_state).squeeze().shape)
                cnn_feature.append(self.cnn_net.forward(perception_state).squeeze()) # forward the CNN to get feature vector
            # print(num_traj, "feature ", len(cnn_feature))
            # print(cnn_feature)
            # Stack along num_traj index
            cnn_feature_out = torch.stack([out for out in cnn_feature]).reshape(-1, num_traj, self.perception_out_size)
            # print(cnn_feature_out)
            # print("robot ",robot_state.shape)
            # print("cnn ", cnn_feature_out.shape)
            x = torch.cat((robot_state, cnn_feature_out), dim=2) # concatenate feature vector with robot states
        elif dims == 1: # for model forward
            robot_state = state[:-self.perception_in_size]
            perception_state = state[-self.perception_in_size:].reshape(-1, self.img_channel, self.img_width, self.img_height)
            cnn_feature = self.cnn_net.forward(perception_state).squeeze() # forward the CNN to get feature vector
            x = torch.cat((robot_state, cnn_feature)) # concatenate feature vector with robot states

        return super()._base_forward(x)

class LSTM_Add_CNN_Base(Net):
    """
    A generic class that add(output of CNN, raw robot states) as LSTM inputs
    CNN often needs redefine, so here constructs the CNN entirely than inheritance.
    Inputs: Concat array with the flattened image indexed in the end
    """
    def __init__(self, state_dim, base_actor_layers, image_shape=(32,32), image_channel=1):
        super(LSTM_Add_CNN_Base, self).__init__()
        self.perception_in_size  = int(image_shape[0] * image_shape[1])
        self.perception_out_size = 16 # make sure matches the CNN output size
        self.img_width   = image_shape[0]
        self.img_height  = image_shape[1]
        self.img_channel = image_channel
        self.cnn_add = FFBase(in_dim=state_dim+self.perception_out_size,
                              layers=[64,base_actor_layers[-1]], nonlinearity='relu')
        self.base_actor_lstm = LSTMBase(in_dim=state_dim, layers=base_actor_layers)

        #NOTE: Taken from before, the following part will be modified per experiment
        self.cnn_net = nn.Sequential(
            nn.Conv2d(1,4,7,2,0), #bs , 4 , 10 , 10
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, 5, 2, 0), #bs , 8 5 5
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1, 0), # bs 16 1 1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

    def init_hidden_state(self, batch_size=1):
        self.base_actor_lstm.init_hidden_state(batch_size)

    def _base_forward(self, state):
        size = state.size()
        dims = len(size)

        if dims == 3: # for optimizaton with batch of trajectories
            # [num steps per episode, num episode, num state per step]
            num_traj = size[1]
            robot_state = state[:,:,:-self.perception_in_size]
            cnn_feature = []
            for traj_idx in range(num_traj):
                traj_data = state[: , traj_idx, -self.perception_in_size:]
                # print("batch dat shae ", traj_data.shape)
                perception_state = traj_data[-self.perception_in_size:].reshape(-1, self.img_channel, self.img_width, self.img_height)
                # print("per size ", perception_state.shape, "out size ", self.cnn_net.forward(perception_state).squeeze().shape)
                cnn_feature.append(self.cnn_net.forward(perception_state).squeeze()) # forward the CNN to get feature vector
            # print(num_traj, "feature ", len(cnn_feature))
            # print(cnn_feature)
            # Stack along num_traj index
            cnn_feature_out = torch.stack([out for out in cnn_feature]).reshape(-1, num_traj, self.perception_out_size)
            # print(cnn_feature_out)
            # print("robot ",robot_state.shape)
            # print("cnn ", cnn_feature_out.shape)
            x = torch.cat((robot_state, cnn_feature_out), dim=2) # concatenate feature vector with robot states
            x_cnn_robot = self.cnn_add._base_forward(x)
        elif dims == 1: # for model forward
            robot_state = state[:-self.perception_in_size]
            perception_state = state[-self.perception_in_size:].reshape(-1, self.img_channel, self.img_width, self.img_height)
            cnn_feature = self.cnn_net.forward(perception_state).squeeze() # forward the CNN to get feature vector
            x = torch.cat((robot_state, cnn_feature)) # concatenate feature vector with robot states
            x_cnn_robot = self.cnn_add._base_forward(x)

        base_lstm_feature = self.base_actor_lstm._base_forward(robot_state)
        x_final = x_cnn_robot + base_lstm_feature
        return x_final

class FFConcatBase(Net):
    """
    The base class for feedforward networks.
    """
    def __init__(self, in_dim, state_dim, map_dim,
                 state_layers, map_layers, concat_layers, nonlinearity='relu', use_cnn=False):
        assert state_dim + map_dim == in_dim, "State and Nonstate Dimension Mismatch"
        super(FFConcatBase, self).__init__()
        self.state_dim = state_dim
        self.nonlinearity = get_activation(nonlinearity)
        self.state_layers = FFBase(in_dim=state_dim, layers=state_layers, nonlinearity=nonlinearity)
        self.concat_layers = FFBase(in_dim=state_layers[-1]+map_layers[-1],
                                    layers=concat_layers, nonlinearity=nonlinearity)
        if use_cnn:
            self.map_layers = CNN_Encoder(img_dim=map_dim)
        else:
            self.map_layers = FFBase(in_dim=map_dim, layers=map_layers, nonlinearity=nonlinearity)

    def _base_forward(self, x):
        size = x.size()
        dims = len(size)
        if dims == 2: # for optimizaton with batch of trajectories
            state_feature = self.state_layers._base_forward(x[:,:self.state_dim])
            perception_feature = self.map_layers._base_forward(x[:,self.state_dim:])
            x_concat = torch.cat((state_feature, perception_feature), dim=1)
        elif dims == 1:
            state_feature = self.state_layers._base_forward(x[:self.state_dim])
            perception_feature = self.map_layers._base_forward(x[self.state_dim:])
            x_concat = torch.cat((state_feature, perception_feature))
        x_final = self.concat_layers._base_forward(x_concat)
        return x_final

    def _latent_space_forward(self, x):
        size = x.size()
        dims = len(size)
        if dims == 2: # for optimizaton with batch of trajectories
            perception_feature = self.map_layers._base_forward(x[:,self.state_dim:])
        elif dims == 1:
            perception_feature = self.map_layers._base_forward(x[self.state_dim:])
        return perception_feature

    def load_base_model(self, base_model_dict):
        """Load base model params
        Args:
            base_model_dict: Previously trained model partially
        """
        for key in self.state_layers.state_dict():
            self.state_layers.state_dict()[key].copy_(base_model_dict["state_layers."+key])
        for key in self.concat_layers.state_dict():
            self.concat_layers.state_dict()[key].copy_(base_model_dict["concat_layers."+key])

        for name, param in self.state_layers.named_parameters():
            param.requires_grad = False
        for name, param in self.concat_layers.named_parameters():
            param.requires_grad = False

class FFLSTMConcatBase(Net):
    """
    The base class for feedforward networks.
    """
    def __init__(self, in_dim, state_dim, map_dim,
                 state_layers, map_layers, concat_layers, nonlinearity='relu', use_cnn=False):
        assert state_dim + map_dim == in_dim, "State and Nonstate Dimension Mismatch"
        super(FFLSTMConcatBase, self).__init__()
        self.state_dim = state_dim
        self.state_layers = FFBase(in_dim=state_dim, layers=state_layers, nonlinearity=nonlinearity)
        self.nonlinearity = get_activation(nonlinearity)
        self.concat_layers = LSTMBase(in_dim=state_layers[-1]+map_layers[-1],
                                      layers=concat_layers)
        if use_cnn:
            self.map_layers = CNN_EncoderOld(img_dim=map_dim)
        else:
            self.map_layers = FFBase(in_dim=map_dim, layers=map_layers, nonlinearity=nonlinearity)

    def init_hidden_state(self, batch_size=1):
        self.concat_layers.init_hidden_state(batch_size=batch_size)

    def _base_forward(self, x):
        size = x.size()
        dims = len(size)
        if dims == 3: # for optimizaton with batch of trajectories
            state_feature = self.state_layers._base_forward(x[:,:,:self.state_dim])
            perception_feature = self.map_layers._base_forward(x[:,:,self.state_dim:])
            x_concat = torch.cat((state_feature, perception_feature), dim=2)
        elif dims == 1:
            state_feature = self.state_layers._base_forward(x[:self.state_dim])
            perception_feature = self.map_layers._base_forward(x[self.state_dim:])
            x_concat = torch.cat((state_feature, perception_feature))
        x_final = self.concat_layers._base_forward(x_concat)
        return x_final

    def _latent_space_forward(self, x):
        size = x.size()
        dims = len(size)
        if dims == 3: # for optimizaton with batch of trajectories
            perception_feature = self.map_layers._base_forward(x[:,:,self.state_dim:])
        elif dims == 1:
            perception_feature = self.map_layers._base_forward(x[self.state_dim:])
        return perception_feature

    def load_base_model(self, base_model_dict):
        """Load base model params
        Args:
            base_model_dict: Previously trained model partially
        """
        for key in self.state_layers.state_dict():
            self.state_layers.state_dict()[key].copy_(base_model_dict["state_layers."+key])
        for key in self.concat_layers.state_dict():
            self.concat_layers.state_dict()[key].copy_(base_model_dict["concat_layers."+key])

        for name, param in self.state_layers.named_parameters():
            param.requires_grad = False
        for name, param in self.concat_layers.named_parameters():
            param.requires_grad = False

class CNN_EncoderOld(nn.Module):
    def __init__(self, img_dim):
        super(CNN_EncoderOld, self).__init__()

        self.img_dim = [128,128]
        self.channel = 1
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, 16, 4),
            nn.ReLU(),
            nn.MaxPool2d((4, 4)),
            nn.Conv2d(4, 8, 4),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc_layers = [32,32]
        self.fc = nn.Linear(*self.fc_layers)

    def cnn_forward(self, x):
        x = x.reshape(-1, self.channel, *self.img_dim)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze()

    def _base_forward(self, x):
        size = x.size()
        dims = len(size)
        if dims == 3: # for optimizaton with batch of trajectories
            # [num steps per episode, num episode, num state per step]
            num_traj = size[1]
            cnn_feature = []
            for traj_idx in range(num_traj):
                traj_data = x[:, traj_idx, :]
                # print("traj_idx ", traj_idx, " traj shape ", traj_data.shape)
                # perception_state = traj_data.reshape(-1, self.channel, *self.img_dim)
                # print("perception size ", traj_data.shape, "out size ", self.cnn_forward(traj_data).shape)
                # print("unsqueeze shape ", self.cnn_forward(traj_data).reshape(-1, 1, 32).shape)
                # CNN forward and reshape to [num_step, num_traj=1, num_hidden_state]
                cnn_feature.append(self.cnn_forward(traj_data).reshape(-1, 1, self.fc_layers[0]))
            # print(num_traj, "feature ", len(cnn_feature))
            # print(cnn_feature)
            # Stack along the num_traj dimension
            cnn_feature_out = torch.concat([out for out in cnn_feature], dim=1)
        elif dims == 1: # for model forward
            cnn_feature_out = self.cnn_forward(x) # forward the CNN to get feature vector
        return cnn_feature_out

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 4, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 4, padding=1, stride=2),
        nn.ReLU(inplace=True)
    )

class CNN_Encoder(nn.Module):
    def __init__(self, img_dim):
        super(CNN_Encoder, self).__init__()

        self.img_dim = [128,128]
        self.channel = 1
        self.down1 = double_conv(1, 8)
        self.down2 = double_conv(8, 16)
        self.down3 = double_conv(16, 32)
        self.fc_layers = [7200, 32]
        self.fc = nn.Linear(*self.fc_layers)

    def cnn_forward(self, x):
        x = x.reshape(-1, self.channel, *self.img_dim)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.relu(x)
        return x.squeeze()

    def _base_forward(self, x):
        size = x.size()
        dims = len(size)
        if dims == 3: # for optimizaton with batch of trajectories
            # [num steps per episode, num episode, num state per step]
            num_traj = size[1]
            cnn_feature = []
            for traj_idx in range(num_traj):
                traj_data = x[:, traj_idx, :]
                # CNN forward and reshape to [num_step, num_traj=1, num_hidden_state]
                cnn_feature.append(self.cnn_forward(traj_data).reshape(-1, 1, self.fc_layers[1]))
            # Stack along the num_traj dimension
            cnn_feature_out = torch.concat([out for out in cnn_feature], dim=1)
        elif dims == 1: # for model forward
            cnn_feature_out = self.cnn_forward(x) # forward the CNN to get feature vector
        return cnn_feature_out

class LocomotionNet(Net):
    """
    This class uses LSTMBase as the backbone model to connecting with other types of output layers.
    LSTMBase is pretrained with the locomotion task.
    """
    def __init__(self, in_dim, state_dim, map_dim, action_dim, base_layers, base_action_dim,
                 state_layers, map_layers, concat_layers, nonlinearity='relu', use_cnn=False,
                 link_base_model=False):
        assert state_dim + map_dim == in_dim, "State and Nonstate Dimension Mismatch"
        super(LocomotionNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_cnn = use_cnn
        self.link_base_model = link_base_model
        self.nonlinearity = get_activation(nonlinearity)
        # Backbone model
        self.base_feature_layer = LSTMBase(in_dim=state_dim, layers=base_layers)
        self.base_output_layer = nn.Linear(base_layers[-1], base_action_dim)
        # Reesidual model with state, perception layers, and concat layers
        self.state_layers = FFBase(in_dim=state_dim, layers=state_layers, nonlinearity=nonlinearity)
        if self.use_cnn:
            # self.map_layers = CNN_Encoder(img_dim=map_dim)
            self.map_layers = CustomResNet(BasicBlock, [(2, 8), (2, 16)],
                                           norm_layer=torch.nn.GroupNorm)
        else:
            self.map_layers = FFBase(in_dim=map_dim, layers=map_layers, nonlinearity=nonlinearity)

        if self.link_base_model:
            concat_layers_in_dim = state_layers[-1]+map_layers[-1]+base_layers[-1]
            # concat_layers_in_dim = state_layers[-1]+base_layers[-1]
        else:
            concat_layers_in_dim = state_layers[-1]+map_layers[-1]
        self.concat_layers = LSTMBase(in_dim=concat_layers_in_dim, layers=concat_layers)
        self.means_layer = nn.Linear(concat_layers[-1], action_dim)

        self.use_encoder = False
        self.encoder_patch = None

    def load_encoder_patch(self):
        # self.encoder_patch = Tarsus_Predictor_v1(hidden_dim_ff=64, inp_dim=26, lstm_num_layers=2)
        # self.encoder_patch.load_state_dict(torch.load("./pretrained_models/encoder/model-2023-08-22_23_35_26.682826-10003631.pth", map_location=torch.device('cpu')))
        self.encoder_patch = Tarsus_Predictor_v2(hidden_dim_ff=128, inp_dim=26, lstm_num_layers=2)
        self.encoder_patch.load_state_dict(torch.load("pretrained_models/encoder/model-2023-08-22_23_49_43.118176-10010530_1.pth", map_location=torch.device('cpu')))
        for name, param in self.encoder_patch.named_parameters():
            param.requires_grad = False
        self.encoder_patch.reset_hidden_state()
        self.encoder_input_idx = [7,8,9,10,11,
                                  12,13,14,15,16,
                                  17,18,19,20,21,
                                  22,23,24,25,26,
                                  27,29,30,
                                  31,33,34]
        self.encoder_output_idx = [28, 32]
        self.use_encoder = True

    def init_hidden_state(self, batch_size=1):
        self.base_feature_layer.init_hidden_state(batch_size=batch_size)
        self.concat_layers.init_hidden_state(batch_size=batch_size)
        if self.encoder_patch:
            self.encoder_patch.reset_hidden_state(batch_size=batch_size)

    def _base_forward(self, x):
        size = x.size()
        dims = len(size)
        if dims == 3: # for optimizaton with batch of trajectories
            if self.use_encoder:
                # patch for left tarsus
                x[:, :, self.encoder_output_idx] = self.encoder_patch.forward(x[:, :, self.encoder_input_idx])
            state_feature = self.state_layers._base_forward(x[:,:,:self.state_dim])
            perception_feature = self.map_layers._base_forward(x[:,:,self.state_dim:])
            base_feature = self.base_feature_layer._base_forward(x[:,:,:self.state_dim])
            if self.link_base_model:
                x_concat = torch.cat((state_feature, perception_feature, base_feature), dim=2)
                # x_concat = torch.cat((state_feature, base_feature), dim=2)
            else:
                x_concat = torch.cat((state_feature, perception_feature), dim=2)
        elif dims == 1:
            if self.use_encoder:
                # patch for left tarsus
                x[self.encoder_output_idx] = self.encoder_patch.forward(x[self.encoder_input_idx].reshape(1,-1)).reshape(-1)
            state_feature = self.state_layers._base_forward(x[:self.state_dim])
            perception_feature = self.map_layers._base_forward(x[self.state_dim:])
            base_feature = self.base_feature_layer._base_forward(x[:self.state_dim])
            if self.link_base_model:
                x_concat = torch.cat((state_feature, perception_feature, base_feature))
                # x_concat = torch.cat((state_feature, base_feature))
            else:
                x_concat = torch.cat((state_feature, perception_feature))
        # Last layer operation to allow residual connection
        base_action = self.base_output_layer(base_feature)
        self.x_final = self.concat_layers._base_forward(x_concat)
        mean_action = self.means_layer(self.x_final)
        if dims == 3:
            mean_action[:,:,:base_action.shape[-1]] += base_action
            # meana_action = self.action_blend_layer._base_forward(torch.cat((mean_action, base_action), dim=2))
        elif dims == 1:
            mean_action[:base_action.shape[-1]] += base_action
            # meana_action = self.action_blend_layer._base_forward(torch.cat((mean_action, base_action)))
        self.perception_feature = perception_feature
        return mean_action

    def _get_perception_feature(self):
        return self.perception_feature

    def load_teacher_model(self, model_dict):
        for k,v in self.state_layers.state_dict().items():
            self.state_layers.state_dict()[k].copy_(model_dict["state_layers."+k])
        for k,v in self.concat_layers.state_dict().items():
            self.concat_layers.state_dict()[k].copy_(model_dict["concat_layers."+k])
        for k,v in self.base_feature_layer.state_dict().items():
            self.base_feature_layer.state_dict()[k].copy_(model_dict["base_feature_layer."+k])
        for k,v in self.base_output_layer.state_dict().items():
            self.base_output_layer.state_dict()[k].copy_(model_dict["base_output_layer."+k])

        # for name, param in self.state_layers.named_parameters():
        #     param.requires_grad = False
        # for name, param in self.concat_layers.named_parameters():
        #     param.requires_grad = False
        for name, param in self.base_feature_layer.named_parameters():
            param.requires_grad = False
        for name, param in self.base_output_layer.named_parameters():
            param.requires_grad = False

    def load_base_model(self, base_model_dict, freeze_backbone=True):
        """Load backbone model params

        Args:
            backbone_model_dict (dict): A dict of backbone model params
        """
        for key in self.base_feature_layer.state_dict():
            self.base_feature_layer.state_dict()[key].copy_(base_model_dict[key])
        for key in self.base_output_layer.state_dict():
            self.base_output_layer.state_dict()[key].copy_(base_model_dict["means."+key])

        # Freeze backbone model
        if freeze_backbone:
            for name, param in self.base_feature_layer.named_parameters():
                param.requires_grad = False
            for name, param in self.base_output_layer.named_parameters():
                param.requires_grad = False


class LocomotionNetV2(Net):
    """Updated version to allow perception feature based on state and image, isolated from pure robot state.
    """
    def __init__(self,
                 obs_dim, state_dim, map_dim, action_dim, map_input_layer_dim,
                 base_layers, state_layers, map_feature_layers, concat_layers,
                 nonlinearity='relu', use_cnn=False, link_base_model=False):
        assert state_dim + map_dim == obs_dim, "State and Nonstate Dimension Mismatch"
        super(LocomotionNetV2, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_cnn = use_cnn
        self.link_base_model = link_base_model
        self.nonlinearity = get_activation(nonlinearity)
        # Backbone model
        self.base_feature_layer = LSTMBase(in_dim=state_dim, layers=base_layers)
        self.base_output_layer = nn.Linear(base_layers[-1], 10)
        # Residual model with state, perception layers, and concat layers
        self.state_layers = FFBase(in_dim=state_dim, layers=state_layers, nonlinearity=nonlinearity)
        if self.use_cnn:
            self.perception_input_layer = CustomResNet(BasicBlock, [(2, 64), (2, 32)],
                                                       norm_layer=torch.nn.GroupNorm)
        else:
            self.perception_input_layer = nn.Linear(map_dim, map_input_layer_dim)
        self.perception_feature_layer = LSTMBase(in_dim=state_dim+map_input_layer_dim, layers=map_feature_layers)

        if self.link_base_model:
            concat_layers_in_dim = state_layers[-1]+map_feature_layers[-1]+10
        else:
            concat_layers_in_dim = state_layers[-1]+map_feature_layers[-1]
        self.concat_layers = FFBase(in_dim=concat_layers_in_dim, layers=concat_layers, nonlinearity=nonlinearity)
        self.means_layer = nn.Linear(concat_layers[-1], action_dim)

    def init_hidden_state(self, batch_size=1):
        self.base_feature_layer.init_hidden_state(batch_size=batch_size)
        self.perception_feature_layer.init_hidden_state(batch_size=batch_size)

    def _base_forward(self, x):
        size = x.size()
        dims = len(size)
        if dims == 3:
            state_feature = self.state_layers._base_forward(x[:,:,:self.state_dim])
            if self.use_cnn:
                map_out = self.perception_input_layer._base_forward(x[:,:,self.state_dim:])
            else:
                map_out = self.perception_input_layer(x[:,:,self.state_dim:])
            perception_input = torch.cat((x[:,:,:self.state_dim], map_out), dim=2)
            perception_feature = self.perception_feature_layer._base_forward(perception_input)
            base_feature = self.base_feature_layer._base_forward(x[:,:,:self.state_dim])
            base_action = self.base_output_layer(base_feature)
            if self.link_base_model:
                x_concat = torch.cat((state_feature, perception_feature, base_action), dim=2)
            else:
                x_concat = torch.cat((state_feature, perception_feature), dim=2)
        elif dims == 1:
            state_feature = self.state_layers._base_forward(x[:self.state_dim])
            if self.use_cnn:
                map_out = self.perception_input_layer._base_forward(x[self.state_dim:])
            else:
                map_out = self.perception_input_layer(x[self.state_dim:])
            perception_input = torch.cat((x[:self.state_dim], map_out))
            perception_feature = self.perception_feature_layer._base_forward(perception_input)
            base_feature = self.base_feature_layer._base_forward(x[:self.state_dim])
            base_action = self.base_output_layer(base_feature)
            if self.link_base_model:
                x_concat = torch.cat((state_feature, perception_feature, base_action))
            else:
                x_concat = torch.cat((state_feature, perception_feature))
        # Last layer operation to allow residual connection
        self.x_final = self.concat_layers._base_forward(x_concat)
        mean_action = self.means_layer(self.x_final)
        if dims == 3:
            mean_action[:,:,:base_action.shape[-1]] += base_action
        elif dims == 1:
            mean_action[:base_action.shape[-1]] += base_action
        self.perception_feature = perception_feature
        return mean_action

    def _get_perception_feature(self):
        return self.perception_feature

    def load_teacher_model(self, model_dict):
        for k,v in self.state_layers.state_dict().items():
            self.state_layers.state_dict()[k].copy_(model_dict["state_layers."+k])
        for k,v in self.concat_layers.state_dict().items():
            self.concat_layers.state_dict()[k].copy_(model_dict["concat_layers."+k])
        for k,v in self.perception_feature_layer.state_dict().items():
            self.perception_feature_layer.state_dict()[k].copy_(model_dict["perception_feature_layer."+k])
        for k,v in self.base_feature_layer.state_dict().items():
            self.base_feature_layer.state_dict()[k].copy_(model_dict["base_feature_layer."+k])
        for k,v in self.base_output_layer.state_dict().items():
            self.base_output_layer.state_dict()[k].copy_(model_dict["base_output_layer."+k])

        # for name, param in self.state_layers.named_parameters():
        #     param.requires_grad = False
        # for name, param in self.concat_layers.named_parameters():
        #     param.requires_grad = False
        # for name, param in self.perception_feature_layer.named_parameters():
        #     param.requires_grad = False
        for name, param in self.base_feature_layer.named_parameters():
            param.requires_grad = False
        for name, param in self.base_output_layer.named_parameters():
            param.requires_grad = False

    def load_base_model(self, base_model_dict, freeze_backbone=True):
        """Load backbone model params

        Args:
            backbone_model_dict (dict): A dict of backbone model params
        """
        for key in self.base_feature_layer.state_dict():
            self.base_feature_layer.state_dict()[key].copy_(base_model_dict[key])
        for key in self.base_output_layer.state_dict():
            self.base_output_layer.state_dict()[key].copy_(base_model_dict["means."+key])

        # Freeze backbone model
        if freeze_backbone:
            for name, param in self.base_feature_layer.named_parameters():
                param.requires_grad = False
            for name, param in self.base_output_layer.named_parameters():
                param.requires_grad = False

class LocomotionNetV3(Net):
    def __init__(self, obs_dim, state_dim, map_dim, action_dim,
                 state_layers, map_layers, concat_layers, nonlinearity='relu', use_cnn=False):
        assert state_dim + map_dim == obs_dim, "State and Nonstate Dimension Mismatch"
        super(LocomotionNetV3, self).__init__()
        self.state_dim = state_dim
        self.state_layers = FFBase(in_dim=state_dim, layers=state_layers, nonlinearity=nonlinearity)
        self.nonlinearity = get_activation(nonlinearity)
        self.concat_layers = LSTMBase(in_dim=state_layers[-1]+map_layers[-1],
                                      layers=concat_layers)
        if use_cnn:
            self.map_layers = CustomResNet(BasicBlock, [(2, 8), (2, 16)],
                                           norm_layer=torch.nn.GroupNorm)
        else:
            self.map_layers = FFBase(in_dim=map_dim, layers=map_layers, nonlinearity=nonlinearity)

    def init_hidden_state(self, batch_size=1):
        self.concat_layers.init_hidden_state(batch_size=batch_size)

    def _base_forward(self, x):
        size = x.size()
        dims = len(size)
        if dims == 3: # for optimizaton with batch of trajectories
            state_feature = self.state_layers._base_forward(x[:,:,:self.state_dim])
            perception_feature = self.map_layers._base_forward(x[:,:,self.state_dim:])
            x_concat = torch.cat((state_feature, perception_feature), dim=2)
        elif dims == 1:
            state_feature = self.state_layers._base_forward(x[:self.state_dim])
            perception_feature = self.map_layers._base_forward(x[self.state_dim:])
            x_concat = torch.cat((state_feature, perception_feature))
        x_final = self.concat_layers._base_forward(x_concat)
        self.perception_feature = perception_feature
        return x_final

    def _get_perception_feature(self):
        return self.perception_feature


if __name__ == '__main__':
    image_size = int(128*128)
    robot_state = 43
    obs_dim = int(image_size+robot_state)
    # net = LSTM_Concat_CNN_Base(state_dim=43, layers=[64,64], image_shape=[32, 32])
    # net = LSTM_Add_CNN_Base(state_dim=43, base_actor_layers=[64,32])
    # net = FFConcatBase(in_dim=robot_state+image_size, state_dim=robot_state, map_dim=image_size,
    #                    state_layers=[32,32], map_layers=[32, 32], concat_layers=[32], use_cnn=True)
    # net = FFLSTMConcatBase(in_dim=robot_state+image_size, state_dim=robot_state, map_dim=image_size,
    #                        state_layers=[32,32], map_layers=[32, 32], concat_layers=[32], use_cnn=True)
    # print(count_parameters(net))
    # print(net)
    # if hasattr(net, 'init_hidden_state'):
    #     net.init_hidden_state()
    # net.eval() # call this for the batchnorm if single data
    # x = torch.ones((obs_dim))
    # print(net._base_forward(x).shape)
    # x = torch.rand((200, 1, obs_dim))
    # print(net._base_forward(x).shape)
    # print(net.state_dict().keys())

    net = LocomotionNet(in_dim=robot_state+image_size, state_dim=robot_state, map_dim=image_size,
                        state_layers=[32,32], map_layers=[32, 32], concat_layers=[32], use_cnn=True,
                        base_layers=[64,64], action_dim=10)
    # net = LocomotionNetV2(obs_dim=robot_state+image_size, state_dim=robot_state, map_dim=image_size,
    #                     state_layers=[128], map_feature_layers=[64], concat_layers=[128,64], use_cnn=True,
    #                     base_layers=[64,64], action_dim=10, map_input_layer_dim=64, link_base_model=False)
    print(net)
    if hasattr(net, 'init_hidden_state'):
        net.init_hidden_state()
    net.eval() # call this for the batchnorm if single data
    x = torch.ones((obs_dim))
    print(f"input shape {x.shape}")
    out = net._base_forward(x)
    print(f"output shape {out.shape}")
    x = torch.rand((200, 1, obs_dim))
    print(f"input shape {x.shape}")
    out = net._base_forward(x)
    print(f"output shape {out.shape}")
    # print(net.state_dict().keys())

    # net = CNN_Encoder(img_dim=[128,128])
    # print(net)
    # net.eval() # call this for the batchnorm if single data
    # x = torch.ones((128*128))
    # print(net._base_forward(x).shape)
    # print(net.state_dict().keys())
    # net.cuda()

    print("total parameters", count_parameters(net))

    # from torch.profiler import profile, record_function, ProfilerActivity
    # net.cuda()
    # x = x.to("cuda:0")
    # with profile(activities=[ProfilerActivity.CPU],
    #         profile_memory=True, record_shapes=True) as prof:
    #     net(x)
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))