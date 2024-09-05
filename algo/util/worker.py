from copy import deepcopy

class AlgoWorker:
    """
        Generic template for a worker (sampler or optimizer) for training algorithms. This worker
        mainly holds the actor and critic networks, and has a function to sync the networks with
        other workers.

        Args:
            actor: actor pytorch network
            critic: critic pytorch network
    """
    def __init__(self, actor, critic):
        self.actor = deepcopy(actor)
        self.critic = deepcopy(critic)

    def sync_policy(self, new_actor_params, new_critic_params, input_norm=None):
        """
        Function to sync the actor and critic parameters with new parameters.

        Args:
            new_actor_params (torch dictionary): New actor parameters to copy over
            new_critic_params (torch dictionary): New critic parameters to copy over
            input_norm (int): Running counter of states for normalization
        """
        for p, new_p in zip(self.actor.parameters(), new_actor_params):
            p.data.copy_(new_p)

        for p, new_p in zip(self.critic.parameters(), new_critic_params):
            p.data.copy_(new_p)

        if input_norm is not None:
            self.actor.welford_state_mean, self.actor.welford_state_mean_diff, self.actor.welford_state_n = input_norm
            self.critic.copy_normalizer_stats(self.actor)