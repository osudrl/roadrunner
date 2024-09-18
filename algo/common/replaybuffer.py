from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class ReplayBuffer:
    def __init__(self, args, buffer_size, num_cassie=1):
        self.args = args
        self.buffer_size = buffer_size
        self.num_cassie = num_cassie
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = dict(
            a=torch.zeros(self.buffer_size, self.num_cassie, self.args.action_dim, dtype=torch.float32),
            r=torch.zeros(self.buffer_size, self.num_cassie, dtype=torch.float32),
            dw=torch.ones(self.buffer_size, self.num_cassie, dtype=torch.bool),
            active=torch.zeros(self.buffer_size, self.num_cassie, dtype=torch.bool))

        self.buffer['s'] = OrderedDict()
        for k, shape in self.args.state_dim.items():
            self.buffer['s'][k] = torch.zeros(self.buffer_size, self.num_cassie, *shape, dtype=torch.float32)

        self.count = 0
        self.s_last = dict()
        # self.v_last = []
        self.ep_lens = []

    # Expecting all child of s in dictionary

    def store_transition(self, s, a, r, dw, active):
        for k in s.keys():
            self.buffer['s'][k][self.count] = s[k]
        self.buffer['a'][self.count] = a
        self.buffer['r'][self.count] = r
        # self.buffer['v'][self.count] = v
        self.buffer['dw'][self.count] = dw
        self.buffer['active'][self.count] = active

        self.count += 1

    def store_last_state(self, s):
        for k in s.keys():
            assert k not in self.s_last, \
                f"s_last already contains {k}. store_last_state may be called more than once for same key"
            self.s_last[k] = s[k]

        # self.v_last.append(v)

        self.ep_lens.append(self.count)

    def merge(self, replay_buffer):
        if self.is_full():
            return

        ep_len = replay_buffer.count
        for k in replay_buffer.buffer.keys():
            if k == 's': continue
            D = replay_buffer.buffer[k].size()[2:]
            replay_buffer.buffer[k] = replay_buffer.buffer[k][:ep_len].transpose(0, 1).reshape(-1, 1, *D)

        for k in replay_buffer.buffer['s'].keys():
            D = replay_buffer.buffer['s'][k].size()[2:]
            replay_buffer.buffer['s'][k] = replay_buffer.buffer['s'][k][:ep_len].transpose(0, 1).reshape(-1, 1, *D)

        # Remaining buffer size
        rem_size = self.buffer_size - self.count

        # Buffer to be merged
        curr_size = ep_len * replay_buffer.num_cassie

        if curr_size > rem_size:
            # Only some episodes can be merged
            self.ep_lens.extend([replay_buffer.count] * (rem_size // ep_len))

            if rem_size % replay_buffer.count != 0:
                self.ep_lens.append(rem_size % ep_len)

            curr_size = rem_size
        else:
            # All episodes can be merged
            self.ep_lens.extend([replay_buffer.count] * replay_buffer.num_cassie)

        for k in self.buffer['s'].keys():
            self.buffer['s'][k][self.count:self.count + curr_size] = replay_buffer.buffer['s'][k][:curr_size]
            num_ep_taken = min(int(np.ceil(rem_size / ep_len)), replay_buffer.num_cassie)
            if k not in self.s_last:
                self.s_last[k] = replay_buffer.s_last[k][:num_ep_taken]
            else:
                self.s_last[k] = torch.cat((self.s_last[k], replay_buffer.s_last[k][:num_ep_taken]), dim=0)

        self.buffer['a'][self.count:self.count + curr_size] = replay_buffer.buffer['a'][:curr_size]
        self.buffer['r'][self.count:self.count + curr_size] = replay_buffer.buffer['r'][:curr_size]
        self.buffer['dw'][self.count:self.count + curr_size] = replay_buffer.buffer['dw'][:curr_size]
        self.buffer['active'][self.count:self.count + curr_size] = replay_buffer.buffer['active'][:curr_size]

        self.count += curr_size

    def is_full(self):
        return self.count >= self.buffer_size

    @staticmethod
    def get_adv(v, v_next, r, dw, active, args):
        # Calculate the advantage using GAE
        adv = torch.zeros_like(r, device=r.device)
        gae = 0
        with torch.no_grad():
            deltas = r + args.gamma * v_next * ~dw - v

            for t in reversed(range(r.size(1))):
                gae = deltas[:, t] + args.gamma * args.lamda * gae
                adv[:, t] = gae
            v_target = adv + v
            if args.use_adv_norm:
                mean = adv[active].mean()
                std = adv[active].std() + 1e-8
                adv = (adv - mean) / std
        return adv, v_target

    @staticmethod
    def unfold(x, size, step):
        return x.unfold(dimension=0, size=min(x.size(0), size), step=step).permute(0, -1, *torch.arange(1, x.dim()))

    #
    # @staticmethod
    # def pad_sequence(x, length):
    #     return F.pad(x, [0] * (x.dim() - 2) * 2 + [0, length], value=0)

    @staticmethod
    def merge_from_multiple_env(args, replay_buffers):
        replay_buffer = ReplayBuffer(args, sum(rb.buffer_size for rb in replay_buffers.values()))

        for rb in replay_buffers.values():
            for k in rb.buffer.keys():
                if k == 's': continue
                replay_buffer.buffer[k][replay_buffer.count:replay_buffer.count + rb.count] = rb.buffer[k]

            for k in rb.buffer['s'].keys():
                replay_buffer.buffer['s'][k][replay_buffer.count:replay_buffer.count + rb.count] = rb.buffer['s'][k]

                if k not in replay_buffer.s_last:
                    replay_buffer.s_last[k] = rb.s_last[k]
                else:
                    replay_buffer.s_last[k] = (
                        torch.cat((replay_buffer.s_last[k], rb.s_last[k]), dim=0)
                    )

            replay_buffer.count += rb.count

            replay_buffer.ep_lens.extend(rb.ep_lens)

        return replay_buffer

    @staticmethod
    def create_buffer(replay_buffers, args, value_function, device):
        with torch.inference_mode():
            replay_buffer = ReplayBuffer.merge_from_multiple_env(args, replay_buffers)
            s = replay_buffer.buffer['s']
            s_last = replay_buffer.s_last
            # v_label_last = replay_buffer.v_last
            a = replay_buffer.buffer['a']
            r = replay_buffer.buffer['r']
            # v_label = replay_buffer.buffer['v']
            dw = replay_buffer.buffer['dw']
            active = replay_buffer.buffer['active']
            ep_lens = replay_buffer.ep_lens

            # Send to device and remove singleton dim
            for k in s.keys():
                s[k] = s[k].to(device).squeeze(1)
                s_last[k] = s_last[k].to(device)
            a = a.to(device).squeeze(1)
            r = r.to(device).squeeze(1)
            dw = dw.to(device).squeeze(1)
            active = active.to(device).squeeze(1)
            active_seq = torch.ones(r.size(0), dtype=torch.bool, device=device)

            # Put s_last at the end of each state in an episode
            for k in s.keys():
                s[k] = s[k].split(ep_lens)
                s[k] = F.pad(pad_sequence(s[k], batch_first=True), (0, 0, 0, 1))
                s[k][np.arange(len(ep_lens)), ep_lens] = s_last[k]

            r = r.split(ep_lens)
            r = pad_sequence(r, padding_value=0, batch_first=True)

            dw = dw.split(ep_lens)
            dw = pad_sequence(dw, padding_value=1, batch_first=True)

            active = active.split(ep_lens)
            active = pad_sequence(active, padding_value=0, batch_first=True)

            active_seq = active_seq.split(ep_lens)
            active_seq = pad_sequence(active_seq, padding_value=0, batch_first=True)

            if hasattr(value_function, 'init_hidden_state'):
                value_function.init_hidden_state(device=device, batch_size=len(ep_lens))

            v = value_function.forward(s).squeeze(-1)
            v = pad_sequence(v, padding_value=0, batch_first=True)

            # Compute v_next (exclude first state, include last state), v (exclude last state, include first state)
            v_next = v[:, 1:]
            v = v[:, :-1]

            # Mask out inactive transitions
            v[~active] = 0
            v_next[~active] = 0

            # Compute advantages
            adv, v_target = ReplayBuffer.get_adv(v, v_next, r, dw, active, args)

            for k in s.keys():
                s[k] = s[k][:, :-1][active_seq]
            adv = adv[active_seq]
            v_target = v_target[active_seq]

            ep_lens = torch.tensor(ep_lens, dtype=torch.long, device=device)
            buffer = dict(s=s, a=a, adv=adv, v_target=v_target, ep_lens=ep_lens, active=active[active_seq])

            return buffer
