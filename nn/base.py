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

def get_activation(act_name):
    try:
        return getattr(torch, act_name)
    except:
        raise RuntimeError(f"Not implemented activation {act_name}. Please add in.")


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
