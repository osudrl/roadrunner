import torch
import torch.nn as nn

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

class LSTMBase(Net):
    """
    The base class for LSTM networks.
    """
    def __init__(self, in_dim, layers):
        super().__init__()
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
            x = x.view(1, -1)
        elif dims == 3:
            self.init_hidden_state()

        x, self.hx = self.lstm(x, self.hx)

        if dims == 1:
            x = x.view(-1)

        return x


class LSTMBase_(Net):
    """
    (DEPRECATED) Will be removed in future. Use this class only for compatibility with old models.
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

def get_activation(act_name):
    try:
        return getattr(torch, act_name)
    except:
        raise RuntimeError(f"Not implemented activation {act_name}. Please add in.")
