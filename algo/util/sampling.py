import numpy as np
import ray
import torch

from time import time
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
from algo.util.worker import AlgoWorker
from util.mirror import mirror_tensor

class Buffer:
    """
    Generic Buffer class to hold samples for PPO.
    Note: that it is assumed that trajectories are stored
    consecutively, next to each other. The list traj_idx stores the indices where individual
    trajectories are started.

    Args:
        discount (float): Discount factor

    Attributes:
        states (list): List of stored sampled observation states
        actions (list): List of stored sampled actions
        rewards (list): List of stored sampled rewards
        values (list): List of stored sampled values
        returns (list): List of stored computed returns
        advantages (list): List of stored computed advantages
        ep_returns (list): List of trajectories returns (summed rewards over whole trajectory)
        ep_lens (list): List of trajectory lengths
        size (int): Number of currently stored states
        traj_idx (list): List of indices where individual trajectories start
        buffer_read (bool): Whether or not the buffer is ready to be used for optimization.
        additional_info_keys (list): List of keys for additional information stored in the buffer.
    """
    def __init__(self, discount: float = 0.99, gae_lambda: float = 0.95):
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clear()

    def __len__(self):
        return self.size

    @property
    def num_trajs(self):
        return len(self.traj_idx) - 1

    def clear(self):
        """
        Clear out/reset all buffer values. Should always be called before starting new sampling iteration
        """
        self.states     = []
        self.actions    = []
        self.rewards    = []
        self.values     = []
        self.returns    = []
        self.advantages = []

        self.ep_returns = []
        self.ep_lens = []

        self.size = 0

        self.traj_idx = [0]
        self.buffer_ready = False

        # Clear additional info if it exists
        if 'additional_info_keys' in self.__dict__:
            for key in self.additional_info_keys:
                if key in self.__dict__:
                    self.__dict__[key] = []
        self.additional_info_keys = []

    def push(self,
             state: np.ndarray,
             action: np.ndarray,
             reward: np.ndarray,
             value: np.ndarray):
        """
        Store new PPO state (state, action, reward, value, termination)

        Args:
            state (numpy vector):  observation
            action (numpy vector): policy action
            reward (numpy vector): reward
            value (numpy vector): value function value
            return (numpy vector): return
        """
        self.states  += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values  += [value]

        self.size += 1

    def push_additional_info(self, info_dict: dict):
        """
        Store any additional information about the current state, and they will be used later in
        optimization. For example, Buffer can store privilege information for the current state.

        Args:
            info_dict (dict): Dictionary containing additional information about the current state
        """
        for key, value in info_dict.items():
            # Add a buffer attribute for the new key
            if key not in self.__dict__.keys():
                self.__dict__[key] = []
                self.additional_info_keys.append(key)
            self.__dict__[key] += [value]

    def end_trajectory(self, terminal_value: float = 0.0):
        """
        Finish a stored trajectory, i.e. calculate return for each step by adding a termination
        value to the last state and backing up return based on discount factor.

        Args:
            terminal_value (float): Estimated value at the final state in the trajectory. Used to
                                    back up and calculate returns for the whole trajectory
        """
        self.traj_idx += [self.size]
        rewards = np.array(self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]).squeeze(axis=1)
        values = self.values[self.traj_idx[-2]:self.traj_idx[-1]] + [terminal_value]

        # Compute GAE + returns
        returns = []
        gae = 0
        for t in range(len(values) - 2, -1, -1):
            delta = rewards[t] + self.discount * values[t + 1] - values[t]
            gae = delta + self.discount * self.gae_lambda * gae
            returns.insert(0, gae + values[t])

        self.returns += returns

        self.ep_returns += [np.sum(rewards)]
        self.ep_lens    += [len(rewards)]

    def _finish_buffer(self, state_mirror_idx = None):
        """
        Get a buffer ready for optimization by turning each list into torch Tensor. Also calculate
        mirror states and normalized advantages. Must be called before sampling from the buffer for
        optimization. While make "buffer_ready" variable true.

        Args:
            mirror (function pointer): Pointer to the state mirroring function that while mirror
                                       observation states
        """
        with torch.no_grad():
            self.states  = torch.Tensor(np.array(self.states))
            self.actions = torch.Tensor(np.array(self.actions))
            self.rewards = torch.Tensor(np.array(self.rewards))
            self.returns = torch.Tensor(np.array(self.returns))
            self.values  = torch.Tensor(np.array(self.values))
            # Finish additional info
            for key in self.additional_info_keys:
                self.__dict__[key] = torch.Tensor(np.array(self.__dict__[key]))

            # Mirror states in needed
            if state_mirror_idx is not None:
                self.mirror_states = mirror_tensor(self.states, state_mirror_idx)

            # Calculate and normalize advantages
            a = self.returns - self.values
            a = (a - a.mean()) / (a.std() + 1e-4)
            self.advantages = a
            self.buffer_ready = True

    def sample(self,
               batch_size: int = 64,
               recurrent: bool = False,
               mirror_state_idx = None):
        """
        Returns a randomly sampled batch from the buffer to be used for optimization. If "recurrent"
        is true, will return a random batch of trajectories to be used for backprop through time.
        Otherwise will return randomly selected states from the buffer.

        Args:
            batch_size (int): Size of the batch. If recurrent is True then the number of
                              trajectories to return. Otherwise is the number of states to return.
            recurrent (bool): Whether to return a recurrent batch (trajectories) or not
            mirror (function pointer): Pointer to the state mirroring function. If is None, the no
                                       mirroring will be done.

        Returns: Batch data in the form of a dictionary with keys "states", "actions", "returns"...
        """
        if not self.buffer_ready:
            self._finish_buffer(mirror_state_idx)

        batch = {}
        if recurrent:
            random_indices = SubsetRandomSampler(range(len(self.traj_idx)-1))
            sampler = BatchSampler(random_indices, batch_size, drop_last=False)

            for traj_indices in sampler:
                # Data format in [num_steps_per_traj, num_trajs, num_states_per_step]
                states     = [self.states[self.traj_idx[i]:self.traj_idx[i+1]]     for i in traj_indices]
                actions    = [self.actions[self.traj_idx[i]:self.traj_idx[i+1]]    for i in traj_indices]
                returns    = [self.returns[self.traj_idx[i]:self.traj_idx[i+1]]    for i in traj_indices]
                advantages = [self.advantages[self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]
                traj_mask  = [torch.ones_like(r, dtype=torch.bool) for r in returns]

                batch['states']     = pad_sequence(states,     batch_first=False)
                batch['actions']    = pad_sequence(actions,    batch_first=False)
                batch['returns']    = pad_sequence(returns,    batch_first=False)
                batch['advantages'] = pad_sequence(advantages, batch_first=False)
                batch['mask']       = pad_sequence(traj_mask,  batch_first=False)

                for k in self.additional_info_keys:
                    batch[k] = pad_sequence([self.__dict__[k][self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices], batch_first=False)

                if mirror_state_idx is None:
                    batch['mirror_states'] = torch.ones_like(batch['returns'])
                else:
                    mirror_states = [self.mirror_states[self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]
                    batch['mirror_states'] = pad_sequence(mirror_states, batch_first=False)

                yield batch
        else:
            random_indices = SubsetRandomSampler(range(self.size))
            sampler = BatchSampler(random_indices, batch_size, drop_last=True)

            for i, idxs in enumerate(sampler):
                batch['states']     = self.states[idxs]
                batch['actions']    = self.actions[idxs]
                batch['returns']    = self.returns[idxs]
                batch['advantages'] = self.advantages[idxs]
                batch['mask']       = torch.ones_like(batch['returns'], dtype=torch.bool)

                for k in self.additional_info_keys:
                    batch[k] = self.__dict__[k][idxs]

                if mirror_state_idx is None:
                    batch['mirror_states'] = torch.ones_like(batch['returns'])
                else:
                    batch['mirror_states'] = self.mirror_states[idxs]

                yield batch

    def __add__(self, buf2):
        new_buf = Buffer(self.discount, self.gae_lambda)
        new_buf.states      = self.states + buf2.states
        new_buf.actions     = self.actions + buf2.actions
        new_buf.rewards     = self.rewards + buf2.rewards
        new_buf.values      = self.values + buf2.values
        new_buf.returns     = self.returns + buf2.returns
        new_buf.advantages  = self.advantages + buf2.advantages
        new_buf.ep_returns  = self.ep_returns + buf2.ep_returns
        new_buf.ep_lens     = self.ep_lens + buf2.ep_lens

        new_buf.traj_idx    = self.traj_idx + [self.size + i for i in buf2.traj_idx[1:]]
        new_buf.size        = self.size + buf2.size

        # Add operation for additional info
        for key in self.additional_info_keys:
            assert key in buf2.additional_info_keys,\
                   f"Buffers do not have the same additional info keys, as {key}."
            new_buf.__dict__[key] = []
            new_buf.additional_info_keys.append(key)
            new_buf.__dict__[key] = self.__dict__[key] + buf2.__dict__[key]
        return new_buf


@ray.remote
class AlgoSampler(AlgoWorker):
    """
    Worker for sampling experience for training algorithms

    Args:
        actor: actor pytorch network
        critic: critic pytorch network
        env_fn: environment constructor function
        gamma: discount factor

    Attributes:
        env: instance of environment
        gamma: discount factor
        dynamics_randomization: if dynamics_randomization is enabled in environment
    """
    def __init__(self, actor, critic, env_fn, gamma, gae_lambda, worker_id: int):
        torch.set_num_threads(1)
        self.gamma  = gamma
        self.gae_lambda  = gae_lambda
        self.env    = env_fn()
        self.worker_id = worker_id

        if hasattr(self.env, 'dynamics_randomization'):
            self.dynamics_randomization = self.env.dynamics_randomization
        else:
            self.dynamics_randomization = False

        AlgoWorker.__init__(self, actor, critic)

    def sample_traj(self,
                    max_traj_len: int = 300,
                    do_eval: bool = False,
                    update_normalization_param: bool=False):
        """
        Function to sample experience

        Args:
            max_traj_len: maximum trajectory length of an episode
            do_eval: if True, use deterministic policy
            update_normalization_param: if True, update normalization parameters
        """
        start_t = time()
        # Toggle models to eval mode
        self.actor.eval()
        self.critic.eval()
        memory = Buffer(self.gamma, self.gae_lambda)
        with torch.no_grad():
            state = torch.Tensor(self.env.reset())
            done = False
            value = 0
            traj_len = 0

            if hasattr(self.actor, 'init_hidden_state'):
                self.actor.init_hidden_state()
            if hasattr(self.critic, 'init_hidden_state'):
                self.critic.init_hidden_state()

            while not done and traj_len < max_traj_len:
                state = torch.Tensor(state)
                action = self.actor(state,
                                    deterministic=do_eval,
                                    update_normalization_param=update_normalization_param)
                if do_eval:
                    # If is evaluation, don't need critic value
                    value = 0.0
                else:
                    value = self.critic(state).numpy()

                # If environment has additional info, push additional info to buffer
                if hasattr(self.env, 'get_info_dict'):
                    memory.push_additional_info(self.env.get_info_dict())

                next_state, reward, done, _ = self.env.step(action.numpy())

                memory.push(state.numpy(), action.numpy(), np.array([reward]), value)

                state = next_state
                traj_len += 1

            value = (not done) * self.critic(torch.Tensor(state)).numpy()
            memory.end_trajectory(terminal_value=value)

        return memory, traj_len / (time() - start_t), self.worker_id
