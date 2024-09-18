import numpy as np
import torch
import torch.nn as nn

from algo.common.network import BaseLSTM


class Tarsus_Predictor_v2(BaseLSTM):
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
                            num_layers=lstm_num_layers, batch_first=True)

        self.out_layers = nn.Sequential(
            nn.Linear(hidden_dim_ff, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Linear(64, 2)
        )

    def forward(self, inp):
        inp = self.fc_layers(inp)
        inp, (self.hidden_state, self.cell_state) = self.lstm(inp, (self.hidden_state, self.cell_state))
        out = self.out_layers(inp)
        return out


class TarsusPatchWrapper(BaseLSTM):
    def __init__(self, actor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = actor
        self.encoder_patch = Tarsus_Predictor_v2(hidden_dim_ff=128, inp_dim=26, lstm_num_layers=2)
        self.encoder_patch.load_state_dict(
            torch.load("pretrained_models/encoder/model-2023-08-22_23_49_43.118176-10010530_1.pth",
                       map_location=torch.device('cpu')))
        self.encoder_patch.eval()
        self.encoder_input_idx = [8, 9, 10, 11, 12,  # left motor positions
                                  13, 14, 15, 16, 17,  # right motor positions
                                  18, 19, 20, 21, 22,  # left motor velocities
                                  23, 24, 25, 26, 27,  # right motor velocities
                                  28,  # left shin position
                                  #  29,    # left tarsus position (to be predicted)
                                  30,  # right shin position
                                  31,  # right tarsus position
                                  32,  # left shin velocity
                                  # 33,     # left tarsus velocity (to be predicted)
                                  34,  # right shin velocity
                                  35,  # right tarsus velocity
                                  ]
        self.encoder_output_idx = [
            29,  # left tarsus position
            33  # left tarsus velocity
        ]

    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        self.encoder_patch.init_hidden_state(device, batch_size)
        return self.actor.init_hidden_state(device, batch_size)

    def __call__(self, *args, **kwargs):
        # manually defer this to .forward, as we cannot inherit from nn.Module
        return self.forward(*args, **kwargs)

    def forward(self, s, *args, **kwargs):
        # before = s['base'][..., self.encoder_output_idx]
        for k in s.keys():
            s[k][..., self.encoder_output_idx] = self.encoder_patch.forward(s[k][..., self.encoder_input_idx])

        # after = s['base'][..., self.encoder_output_idx]
        # #
        # print("Before: ", before, "After: ", after, "Diff: ", np.square(after - before).mean())

        return self.actor(s, *args, **kwargs)
