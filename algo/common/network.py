import logging

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class BaseLSTM(nn.Module):
    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        else:
            self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

    def forward(self, s):
        raise NotImplementedError


class Actor_LSTM(BaseLSTM):
    def __init__(self, args):
        super(Actor_LSTM, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.lstm_hidden_dim),
        )

        self.lstm = nn.LSTM(input_size=args.lstm_hidden_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s_b = s['base']
        # s_b: [batch_size, seq_len, arg.state_dim]

        # s_p = s['privilege']
        s_b = self.fc1(s_b)
        # s_p: [batch_size, seq_len, hidden_dim]

        s, (self.hidden_state, self.cell_state) = self.lstm(s_b, (self.hidden_state, self.cell_state))
        # s: [batch_size, seq_len, hidden_size_lstm]

        mean = torch.tanh(self.mean_layer(s))
        # mean: [batch_size, seq_len, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_LSTM(BaseLSTM):
    def __init__(self, args):
        super(Critic_LSTM, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(args.privilege_state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.lstm_hidden_dim),
        )

        self.lstm = nn.LSTM(input_size=args.lstm_hidden_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.value_layer = nn.Sequential(
            nn.Linear(args.lstm_hidden_dim, args.lstm_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.lstm_hidden_dim, 1)
        )

    def forward(self, s):
        s_p = s['privilege']
        # s_p: [batch_size, seq_len, hidden_dim]

        s_p = self.fc1(s_p)
        # s_p: [batch_size, seq_len, hidden_dim]

        s, (self.hidden_state, self.cell_state) = self.lstm(s_p, (self.hidden_state, self.cell_state))
        # s: [batch_size, seq_len, hidden_size_lstm]

        value = self.value_layer(s)
        # mean: [batch_size, seq_len, action_dim]

        return value


class Actor_LSTM_v2(BaseLSTM):
    def __init__(self, args):
        super(Actor_LSTM_v2, self).__init__()

        state_dim = sum([sum(shape) for shape in args.state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s = torch.cat(list(s.values()), dim=-1)
        # s: [batch_size, seq_len, *]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [batch_size, seq_len, hidden_size_lstm]

        mean = self.mean_layer(s)
        # mean: [batch_size, seq_len, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_LSTM_v2(BaseLSTM):
    def __init__(self, args):
        super(Critic_LSTM_v2, self).__init__()

        state_dim = sum([sum(shape) for shape in args.state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.value_layer = nn.Sequential(
            nn.Linear(args.lstm_hidden_dim, 1)
        )

    def forward(self, s):
        s = torch.cat(list(s.values()), dim=-1)
        # s: [batch_size, seq_len, *]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [batch_size, seq_len, hidden_size_lstm]

        value = self.value_layer(s)
        # mean: [batch_size, seq_len, action_dim]

        return value


class Actor_LSTM_v3(BaseLSTM):
    def __init__(self, args):
        super(Actor_LSTM_v3, self).__init__()

        self.fc1 = nn.Linear(11, 32)

        self.lstm = nn.LSTM(input_size=args.state_dim + self.fc1.out_features - 6, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s = s['base']
        # s: [batch_size, seq_len, hidden_dim]

        s_base_orient_base_yaw = s[..., :5]
        # s_base_orient_base_yaw: [batch_size, seq_len, 5]

        s_motor = s[..., :-6]
        # s_motor: [batch_size, seq_len, arg.state_dim-6]

        s_cmd_encoding = s[..., -6:]
        # s_cmd_encoding: [batch_size, seq_len, 6]

        s_latent = torch.cat((s_base_orient_base_yaw, s_cmd_encoding), dim=-1)
        # s_latent: [batch_size, seq_len, 11]

        s_latent = self.fc1(s_latent)
        # s_cmd_encoding: [batch_size, seq_len, 32]

        s = torch.cat((s_motor, s_latent), dim=-1)
        # s: [batch_size, seq_len, arg.state_dim + 32 - 6]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [batch_size, seq_len, hidden_size_lstm]

        mean = self.mean_layer(s)
        # mean: [batch_size, seq_len, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_LSTM_v3(BaseLSTM):
    def __init__(self, args):
        super(Critic_LSTM_v3, self).__init__()

        self.fc1 = nn.Linear(11, 32)

        self.lstm = nn.LSTM(input_size=args.privilege_state_dim + self.fc1.out_features - 6,
                            hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.value_layer = nn.Sequential(
            nn.Linear(args.lstm_hidden_dim, 1)
        )

    def forward(self, s):
        s = s['privilege']
        # s: [batch_size, seq_len, hidden_dim]

        s_base_orient_base_yaw = s[..., :5]
        # s_base_orient_base_yaw: [batch_size, seq_len, 5]

        s_motor = s[..., :-6]
        # s_motor: [batch_size, seq_len, arg.state_dim-6]

        s_cmd_encoding = s[..., -6:]
        # s_cmd_encoding: [batch_size, seq_len, 6]

        s_latent = torch.cat((s_base_orient_base_yaw, s_cmd_encoding), dim=-1)
        # s_latent: [batch_size, seq_len, 11]

        s_latent = self.fc1(s_latent)
        # s_cmd_encoding: [batch_size, seq_len, 32]

        s = torch.cat((s_motor, s_latent), dim=-1)
        # s: [batch_size, seq_len, arg.state_dim - 6 + 32]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [batch_size, seq_len, hidden_size_lstm]

        value = self.value_layer(s)
        # mean: [batch_size, seq_len, action_dim]

        return value


class Actor_FF(nn.Module):
    def __init__(self, args):
        super(Actor_FF, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.action_dim),
        )

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'fc1'):
                orthogonal_init(self.fc1, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s_b = s['base']
        # s_b: [batch_size, seq_len, arg.state_dim]

        mean = self.fc1(s_b)
        # mean: [batch_size, seq_len, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_FF(nn.Module):
    def __init__(self, args):
        super(Critic_FF, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(args.privilege_state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1),
        )

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'fc1'):
                orthogonal_init(self.fc1, gain=0.01)

    def forward(self, s):
        s_b = s['privilege']
        # s_b: [batch_size, seq_len, arg.state_dim]

        value = self.fc1(s_b)
        # mean: [batch_size, seq_len, 1]

        return value


class Actor_FF_v2(nn.Module):
    def __init__(self, args):
        super(Actor_FF_v2, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Linear(args.hidden_dim, args.action_dim),
        )

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'fc1'):
                orthogonal_init(self.fc1, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s_b = s['base']
        # s_b: [batch_size, seq_len, arg.state_dim]

        mean = self.fc1(s_b)
        # mean: [batch_size, seq_len, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_FF_v2(nn.Module):
    def __init__(self, args):
        super(Critic_FF_v2, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(args.privilege_state_dim, args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Linear(args.hidden_dim, 1),
        )

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'fc1'):
                orthogonal_init(self.fc1, gain=0.01)

    def forward(self, s):
        s_b = s['privilege']
        # s_b: [batch_size, seq_len, arg.state_dim]

        value = self.fc1(s_b)
        # mean: [batch_size, seq_len, 1]

        return value
