import numpy as np
import os
import ray
import torch
import torch.optim as optim

from copy import deepcopy
from time import time
from torch.distributions import kl_divergence

from algo.util.worker import AlgoWorker
from util.mirror import mirror_tensor

@ray.remote
class DistillOptim(AlgoWorker):
    def __init__(self,
                 actor,
                 critic,
                 mirror_dict,
                 use_offline_data,
                 teacher_actor=None,
                 a_lr=1e-4,
                 c_lr=1e-4,
                 eps=1e-6,
                 entropy_coeff=0,
                 grad_clip=0.01,
                 mirror=0,
                 clip=0.2,
                 save_path=None,
                 **kwargs):
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print(f"Using cuda:1 GPU for optimization.")
                self.device = torch.device("cuda:1")
            else:
                print(f"Using cuda:0 GPU for optimization.")
                self.device = torch.device("cuda:0")
        else:
            self.device = "cpu"
        print(f"Optimizer uses device: {self.device}")
        AlgoWorker.__init__(self, actor=actor, critic=critic, device=self.device)
        self.old_actor = deepcopy(actor).to(self.device)
        if teacher_actor is None:
            raise RuntimeError("Need to pass a teacher actor for optimizer.")
        self.teacher_actor = teacher_actor.to(self.device)
        self.teacher_actor.eval()
        self.actor_optim   = optim.Adam(self.actor.parameters(), lr=a_lr, eps=eps)
        self.critic_optim  = optim.Adam(self.critic.parameters(), lr=c_lr, eps=eps)
        self.action_loss = torch.nn.MSELoss(reduction='sum')
        self.perception_loss = torch.nn.MSELoss(reduction='sum')
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip
        self.mirror    = mirror
        self.clip = clip
        self.save_path = save_path
        self.use_offline_data = use_offline_data
        # Unpack mirror dict
        self.state_mirror_indices  = mirror_dict['state_mirror_indices']
        self.action_mirror_indices = mirror_dict['action_mirror_indices']
        if self.state_mirror_indices is not None:
            self.state_mirror_indices = torch.tensor(self.state_mirror_indices).to(self.device)
        if self.action_mirror_indices is not None:
            self.action_mirror_indices = torch.tensor(self.action_mirror_indices).to(self.device)

    def optimize(self,
                 memory,
                 epochs=4,
                 batch_size=32,
                 kl_thresh=0.02,
                 recurrent=False,
                 verbose=False):
        """
        Does a single optimization step given buffer info

        Args:
            memory (Buffer): Buffer object of rollouts from experience collection phase of PPO
            epochs (int): optimization epochs
            batch_size (int): optimization batch size
            kl_thresh (float): threshold for max kl divergence
            recurrent (bool): Buffer samples for recurrent policy or not
            state_mirror_indices(list): environment-specific list of mirroring information
            state_mirror_indices(list): environment-specific list of mirroring information
            verbose (bool): verbose logger output
        """
        torch.set_num_threads(1)
        kls, a_loss, c_loss, m_loss = [], [], [], []
        done = False
        for epoch in range(epochs):
            epoch_start = time()
            if not self.use_offline_data:
                for batch in memory.sample(batch_size=batch_size,
                                           recurrent=recurrent):
                    # Transfer batch to GPU
                    for k in ['teacher_states', 'states', 'mask']:
                        batch[k] = batch[k].to(device=self.device, non_blocking=True)
                    kl, losses = self._update_policy(batch)
            else:
                # Transfer batch to GPU
                batch = memory
                for k,v in batch.items():
                    batch[k] = v.to(device=self.device, non_blocking=True)
                # NOTE: reconstruct states from nonperception states and depth
                # Offline data is stored as nonperception states and depth to reduce file size
                batch['states'] = torch.cat((batch['nonperception_states'],
                                             batch['depth'].reshape(batch['depth'].shape[0],
                                                                    batch['depth'].shape[1],
                                                                    -1)), dim=2)
                kl, losses = self._update_policy(memory)
            kls    += [kl]
            a_loss += [losses[0]]
            c_loss += [losses[1]]
            m_loss += [losses[2]]

            #     if max(kls) > kl_thresh:
            #         print(f"\t\tbatch had kl of {max(kls)} (threshold {kl_thresh}), stopping " \
            #               f"optimization early.")
            #         done = True
            #         break

            if verbose:
                print(f"\t\tepoch {epoch+1:2d} in {(time() - epoch_start):3.2f}s, " \
                      f"kl {np.mean(kls):6.5f}, actor loss {np.mean(a_loss):6.3f}, " \
                      f"critic loss {np.mean(c_loss):6.3f}")

            # if done:
            #     break
        return np.mean(a_loss), np.mean(c_loss), np.mean(m_loss), np.mean(kls)

    def retrieve_parameters(self):
        """
        Function to return parameters for optimizer copies of actor and critic
        """
        return list(self.actor.parameters()), list(self.critic.parameters())

    def _update_policy(self, batch):
        self.actor.train(True)
        num_states = batch['mask'].sum()
        student_actions = self.actor(batch['states'], deterministic=True) * batch['mask']
        student_latent = self.actor.get_perception_feature() * batch['mask']
        if not self.use_offline_data:
            teacher_actions = self.teacher_actor(batch['teacher_states'], deterministic=True) * batch['mask']
            teacher_latent = self.teacher_actor.get_perception_feature() * batch['mask']
        else:
            teacher_actions = batch['actions'] * batch['mask']
            teacher_latent = batch['perception_feature'] * batch['mask']
        actor_loss  = self.action_loss(student_actions, teacher_actions) / num_states
        latent_loss = self.perception_loss(student_latent, teacher_latent) / num_states

        # if not (torch.isfinite(states).all() and torch.isfinite(actions).all() \
        #         and torch.isfinite(returns).all() and torch.isfinite(advantages).all():
        #     torch.save({"states": states,
        #                 "actions": actions,
        #                 "returns": returns,
        #                 "advantages": advantages,
        #                 "actor_loss": actor_loss}, os.path.join(self.save_path, "training_error.pt"))
        #     raise RuntimeError(f"Optimization experiences non-finite values, please check locally"
        #                        f" saved file at training_error.pt for further diagonose.")

        self.actor_optim.zero_grad()
        (actor_loss).backward()

        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
        self.actor_optim.step()
        return 0, (actor_loss.item(), latent_loss.item(), 0)
