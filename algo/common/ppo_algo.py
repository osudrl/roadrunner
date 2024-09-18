import copy

import numpy as np
import torch.nn.functional as F
import tqdm
from torch.distributions import kl_divergence
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from algo.common.network import *
from algo.common.replaybuffer import ReplayBuffer
from algo.common.utils import mirror_tensor, get_sliding_windows


class PPO_algo:
    def __init__(self, args, device, mirror_dict):
        self.args = args
        self.device = device
        self.mirror_dict = mirror_dict

        # self.actor = Actor_LSTM_Depth(args)
        # self.critic = Critic_LSTM_Depth(args)
        self.actor = Actor_LSTM_v2(args)
        self.critic = Critic_LSTM_v2(args)
        # self.actor = Actor_Transformer(args)
        # self.critic = Critic_Transformer(args)

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        if self.args.set_adam_eps:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_a, eps=self.args.eps)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_c, eps=self.args.eps)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_c)

        self.actor_old = copy.deepcopy(self.actor)
        self.actor_old.eval()

        self.num_thread = torch.get_num_threads()

        self.pbar_epoch = tqdm.tqdm(total=args.num_epoch, desc='Training epoch',
                                    position=np.count_nonzero(self.args.num_cassie_prob) + 2, colour='green')

    def update(self, replay_buffers, total_steps, check_kl):
        torch.set_num_threads(self.num_thread)

        losses = []
        kls = []

        # Create buffer
        buffer = ReplayBuffer.create_buffer(replay_buffers, self.args, self.critic, self.device)

        windows, active_seq = get_sliding_windows(ep_lens=buffer['ep_lens'].tolist(), buffer_size=buffer['a'].size(0))

        num_batches = active_seq.size(0)

        active_seq = active_seq.to(self.device)

        self.actor_old.load_state_dict(self.actor.state_dict())

        for i in range(self.args.num_epoch):
            self.pbar_epoch.update(1)

            early_stop = False
            sampler = BatchSampler(SubsetRandomSampler(range(num_batches)), self.args.mini_batch_size, False)
            for index in sampler:

                if hasattr(self.actor, 'init_hidden_state'):
                    self.actor.init_hidden_state(device=self.device, batch_size=len(index))

                if hasattr(self.critic, 'init_hidden_state'):
                    self.critic.init_hidden_state(device=self.device, batch_size=len(index))

                if hasattr(self.actor_old, 'init_hidden_state'):
                    self.actor_old.init_hidden_state(device=self.device, batch_size=len(index))

                # Get active_seq mask for selected indices
                _active_seq = active_seq[index]

                # Get transitions for selected indices
                s = buffer['s'].copy()
                for k in s.keys():
                    s[k] = s[k][windows[index]]

                a = buffer['a'][windows[index]]
                # a: [batch_size, seq_len, args.action_dim]

                adv = buffer['adv'][windows[index]]
                # adv: [batch_size, seq_len]

                v_target = buffer['v_target'][windows[index]]
                # v_target: [batch_size, seq_len]

                active = buffer['active'][windows[index]]
                # active: [batch_size, seq_len]

                _active_seq &= active

                with torch.inference_mode():
                    dist = self.actor_old.pdf(s)
                    a_logprob = dist.log_prob(a)

                # Forward pass
                try:
                    dist_now = self.actor.pdf(s)
                except ValueError as ve:
                    mean, std = self.actor.forward(s)
                    torch.save({'s': s, 'a': a, 'mean': mean, 'std': std, 'adv': adv, 'v_target': v_target,
                                'actor_old_state_dict': self.actor_old.state_dict(),
                                'actor_state_dict': self.actor.state_dict(),
                                'optimizer_actor_state_dict': self.optimizer_actor.state_dict()},
                               f'training_logs/training_error_{self.args.run_name}.pt')
                    logging.error(
                        f"Non-finite values detected in training. Saved to training_logs/training_error_{self.args.run_name}.pt'")
                    raise ve

                values_now = self.critic(s).squeeze(-1)

                ratios = (dist_now.log_prob(a).sum(-1) - a_logprob.sum(-1)).exp()

                del a_logprob

                # actor loss
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * adv
                entropy_loss = - self.args.entropy_coef * dist_now.entropy().sum(-1)
                actor_loss = -torch.min(surr1, surr2)

                actor_loss = actor_loss[_active_seq].mean()
                entropy_loss = entropy_loss[_active_seq].mean()
                critic_loss = (0.5 * F.mse_loss(values_now, v_target, reduction='none'))[_active_seq].mean()

                if self.args.use_mirror_loss:
                    s_mirrored = {}
                    for k in s.keys():
                        if k == 'depth':
                            s_mirrored[k] = s[k].flip(dims=[-1])
                            continue
                        s_mirrored[k] = mirror_tensor(s[k], self.mirror_dict['state_mirror_indices'][k])
                    with torch.no_grad():
                        mirrored_a, _ = self.actor.forward(s_mirrored)
                    target_a = mirror_tensor(mirrored_a, self.mirror_dict['action_mirror_indices'])
                    mirror_loss = (0.5 * F.mse_loss(dist_now.mean, target_a, reduction='none'))[_active_seq].mean()

                with torch.inference_mode():
                    kl = kl_divergence(dist_now, dist).sum(-1)[_active_seq].mean()
                    kls.append(kl.item())

                if self.args.kl_check and check_kl and kl > self.args.kl_threshold:
                    logging.warning(f'Early stopping at epoch {i} due to reaching max kl.')
                    early_stop = True
                    break

                log = {'epochs': i, 'actor_loss': actor_loss.item(), 'entropy_loss': entropy_loss.item(),
                       'critic_loss': critic_loss.item(), 'num_batches': num_batches,
                       'kl_divergence': kl, 'active_count': _active_seq.sum().item(), 'active_shape': _active_seq.shape}

                if self.args.use_mirror_loss:
                    log['mirror_loss'] = mirror_loss.item()
                    losses.append((log['actor_loss'], log['entropy_loss'], log['mirror_loss'], log['critic_loss']))
                else:
                    losses.append((log['actor_loss'], log['entropy_loss'], log['critic_loss']))

                # sampler.set_description(str(log))

                # Check for error
                if not (all([torch.isfinite(s[k]).all() for k in s.keys()])
                        and torch.isfinite(a).all()
                        and torch.isfinite(adv).all()
                        and torch.isfinite(v_target).all()):
                    torch.save({'s': s, 'a': a, 'adv': adv, 'v_target': v_target, 'values_now': values_now,
                                'actor_old_state_dict': self.actor_old.state_dict(),
                                'actor_state_dict': self.actor.state_dict(),
                                'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
                                'surr1': surr1, 'surr2': surr2, 'entropy_loss': entropy_loss, 'actor_loss': actor_loss,
                                'critic_loss': critic_loss, 'kl': kl, 'active_seq': _active_seq},
                               f'training_logs/training_error_{self.args.run_name}.pt')
                    raise RuntimeError(
                        f"Non-finite values detected in training. Saved to training_logs/training_error_{self.args.run_name}.pt'")

                # Update
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()

                if self.args.use_mirror_loss:
                    (actor_loss + entropy_loss + mirror_loss).backward()
                else:
                    (actor_loss + entropy_loss).backward()
                critic_loss.backward()

                if self.args.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.args.grad_clip)

                if any([((~param.grad.isfinite()).any()).item() for param in self.actor.parameters() if
                        param.grad is not None]):

                    # collect gradients in array
                    gradients = []
                    for param in self.actor.parameters():
                        if param.grad is None:
                            continue
                        gradients.append(param.grad)

                    torch.save({'s': s, 'a': a, 'adv': adv, 'v_target': v_target, 'values_now': values_now,
                                'actor_old_state_dict': self.actor_old.state_dict(),
                                'actor_state_dict': self.actor.state_dict(),
                                'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
                                'gradients': gradients,
                                'surr1': surr1, 'surr2': surr2, 'entropy_loss': entropy_loss, 'actor_loss': actor_loss,
                                'critic_loss': critic_loss, 'kl': kl, 'active_seq': _active_seq},
                               f'training_logs/training_error_{self.args.run_name}.pt')
                    # raise RuntimeError(
                    #     f"Non-finite values detected in gradients. Saved to training_logs/training_error_{self.args.run_name}.pt'")
                    logging.warning(
                        f"Non-finite values detected in gradients. Saved to training_logs/training_error_{self.args.run_name}.pt'")
                    early_stop = True
                    break

                self.optimizer_actor.step()
                self.optimizer_critic.step()

                del s, ratios, surr1, surr2, adv, actor_loss, entropy_loss, critic_loss, values_now, v_target, _active_seq, dist_now,

                if self.args.empty_cuda_cache and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            if early_stop:
                break

        if self.args.use_lr_decay:
            self.lr_decay(total_steps)

        if self.args.use_mirror_loss:
            a_loss, e_loss, m_loss, c_loss = zip(*losses)
            a_loss, e_loss, m_loss, c_loss = np.mean(a_loss), np.mean(e_loss), np.mean(m_loss), np.mean(c_loss)
        else:
            a_loss, e_loss, c_loss = zip(*losses)
            a_loss, e_loss, c_loss = np.mean(a_loss), np.mean(e_loss), np.mean(c_loss)
            m_loss = None

        kl = np.mean(kls)

        del losses, kls, buffer

        if self.args.empty_cuda_cache and self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.pbar_epoch.reset()

        return a_loss, e_loss, m_loss, c_loss, kl, num_batches, i

    def lr_decay(self, total_steps):
        lr_a_now = self.args.lr_a * (1 - total_steps / self.args.max_steps)
        lr_c_now = self.args.lr_c * (1 - total_steps / self.args.max_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
