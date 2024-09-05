import numpy as np
import torch

def train_normalizer(env_fn, policy, min_timesteps, max_traj_len=1000, noise=0.5):
    with torch.no_grad():
        env = env_fn()
        env.dynamics_randomization = False

        total_t = 0
        while total_t < min_timesteps:
            state = env.reset()
            done = False
            timesteps = 0

            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

            while not done and timesteps < max_traj_len:
                state = torch.Tensor(state)
                if noise is None:
                    action = policy.forward(state, update_normalization_param=True, deterministic=False).numpy()
                else:
                    action = policy.forward(state, update_normalization_param=True).numpy() + np.random.normal(0, noise, size=policy.action_dim)
                state, _, done, _ = env.step(action)
                timesteps += 1
                total_t += 1