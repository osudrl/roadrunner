import torch
import torch.nn as nn

class Tarsus_Predictor_v1(nn.Module):
    def __init__(self, hidden_dim_ff, inp_dim, lstm_num_layers):
        super(Tarsus_Predictor_v1, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_ff, hidden_dim_ff),
        )

        self.lstm = nn.LSTM(input_size=hidden_dim_ff,
                            hidden_size=hidden_dim_ff,
                            num_layers=lstm_num_layers, batch_first=False)

        self.out_layers = nn.Sequential(
            nn.Linear(hidden_dim_ff, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.Linear(16, 2)
        )

    def reset_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        # if batch_size is None:
        #     self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        #     self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        # else:
        #     self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        #     self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        self.hx = None

    def forward(self, inp):
        inp = self.fc_layers(inp)

        # inp, (self.hidden_state, self.cell_state) = self.lstm(inp, (self.hidden_state, self.cell_state))
        inp, self.hx = self.lstm(inp, self.hx)

        out = self.out_layers(inp)

        return out


class Tarsus_Predictor_v2(nn.Module):
    def __init__(self, inp_dim, hidden_dim_ff, lstm_num_layers):
        super(Tarsus_Predictor_v2, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_ff, hidden_dim_ff),
            nn.Linear(hidden_dim_ff, hidden_dim_ff),
        )

        self.lstm = nn.LSTM(input_size=hidden_dim_ff,
                            hidden_size=hidden_dim_ff,
                            num_layers=lstm_num_layers, batch_first=False)

        self.out_layers = nn.Sequential(
            nn.Linear(hidden_dim_ff, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Linear(64, 2)
        )

    def reset_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        # if batch_size is None:
        #     self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        #     self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        # else:
        #     self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        #     self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        self.hx = None

    def forward(self, inp):
        inp = self.fc_layers(inp)

        # inp, (self.hidden_state, self.cell_state) = self.lstm(inp, (self.hidden_state, self.cell_state))
        inp, self.hx = self.lstm(inp, self.hx)

        out = self.out_layers(inp)

        return out


class Tarsus_Predictor_v3(nn.Module):
    def __init__(self, inp_dim, hidden_dim_ff, lstm_num_layers):
        super(Tarsus_Predictor_v3, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_ff, hidden_dim_ff),
        )

        self.lstm = nn.LSTM(input_size=hidden_dim_ff,
                            hidden_size=hidden_dim_ff,
                            num_layers=lstm_num_layers, batch_first=True)

        self.out_layers_1 = nn.Sequential(
            nn.Linear(hidden_dim_ff, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        self.out_layers_2 = nn.Sequential(
            nn.Linear(hidden_dim_ff, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def reset_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        else:
            self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

    def forward(self, inp):
        inp = self.fc_layers(inp)

        inp, (self.hidden_state, self.cell_state) = self.lstm(inp, (self.hidden_state, self.cell_state))

        out_1 = self.out_layers_1(inp)
        out_2 = self.out_layers_2(inp)

        return torch.cat((out_1, out_2), dim=-1)

class Tarsus_Predictor_v6(nn.Module):
    def __init__(self, inp_dim, hidden_dim_ff, lstm_num_layers, inp_min, inp_max):
        super(Tarsus_Predictor_v6, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_ff, hidden_dim_ff),
        )

        self.lstm = nn.LSTM(input_size=hidden_dim_ff,
                            hidden_size=hidden_dim_ff,
                            num_layers=lstm_num_layers, batch_first=True)

        self.out_layers_1 = nn.Sequential(
            nn.Linear(hidden_dim_ff, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        self.out_layers_2 = nn.Sequential(
            nn.Linear(hidden_dim_ff, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        if inp_min is not None and inp_max is not None:
            self.inp_min = torch.tensor(inp_min)
            self.inp_max = torch.tensor(inp_max)

    def to(self, device):
        if hasattr(self, 'inp_min') and hasattr(self, 'inp_max'):
            self.inp_min = self.inp_min.to(device)
            self.inp_max = self.inp_max.to(device)
        return super().to(device)

    @staticmethod
    def scale_value(value, src_min, src_max, trg_min, trg_max):
        return (value - src_min) / (src_max - src_min) * (trg_max - trg_min) + trg_min

    def reset_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        else:
            self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

    def forward(self, inp):
        inp = self.fc_layers(inp)

        inp, (self.hidden_state, self.cell_state) = self.lstm(inp, (self.hidden_state, self.cell_state))

        out_1 = self.out_layers_1(inp)
        out_2 = self.out_layers_2(inp)

        out = torch.cat((out_1, out_2), dim=-1)

        if hasattr(self, 'inp_min') and hasattr(self, 'inp_max'):
            out = self.scale_value(out, -1.0, 1.0, self.inp_min, self.inp_max)

        return out