# Algo
This folder holds all of the policy optimization code. Currently only PPO is implemented, but any other future algorithms will be here as well. We use [Ray](https://docs.ray.io/en/latest/) to handle parallelization of sampling.

## PPO Training Arguments
There are many arguments available to you to set the hyperparameters of PPO however you choose. A brief description of each argument:
- `prenorm`: Whether to do prenormalization or not. Prenormalization gathers some samples before training to get an estimate on the statistics of the input distribution in order to normalize the inputs for better/more stable learning. This can help the beginning of learning since intially the network normalizatino parameters are set to zero, though this is not strictly necessary.
- `prenormalize-steps`: The number of steps to gather when doing prenormalization
- `update-norm`: Whether to update the input normalization parameters during training. If this is true, the networks will use the data gathered during training to update their running estimate of the mean and std dev of the inputs for input normalization.
- `num-steps`: The number of samples to collect each iteration.
- `num-eval-eps`: The number of episodes collected during evaluation. This specifies how many evaluation trajectories (trajectories using the deterministic policy) will be gathered for logging/metric purposes.
- `eval-freq`: How often to collect evaluation metrics. To save time, we often don't collect evaluation trajectories every iteration. The training reward/episode length is usually a good enough signal for how the training is doing, so a dense evaluation signal is often not necessary. This argument specifies how often evaluation is done, i.e. every 50 iterations if set to 50.
- `discount`: The discount factor to use when calculating the returns.
- `gae-lambda`: The lambda bias-variance tradeoff hyperparameter for Generalized Advantage Estimation (GAE).
- `a-lr`: The actor policy learning rate. Note that we use ADAM for optimization, so this is just the initial learning rate that we set, ADAM will adapt as it sees fit during training.
- `c-lr`: The critic learning rate
- `eps`: The ADAM optimizer epsilon value. This is added to the denominator to improve numerical stability.
- `kl`: The PPO KL divergence threshold. If during policy optimization we find that the KL divergence between the new policy and the old policy exceeds this threshold we stop optimization early.
- `clip`: Clipping value for the clipped PPO objective. During policy optimization, the log probability ratio will be clipped to (1 +- clip)
- `grad-clip`: The gradient clipping value, i.e. the maximum allowed gradient norm when taking an update step.
- `batch-size`: Minibatch size to use during optimization
- `epochs`: Number of epochs to optimize for each iteration. How many times to loop over the buffer each iteration.
- `mirror`: The mirror loss coefficient (see below for details on mirror loss). Set to 0 if you don't want to use mirroring. Can set to be greater than 1 if want to more heavily weight the mirror loss.
- `entropy-coeff`: The coefficient of entropy loss in the actor loss used in optimization. This is usually unused and just set to 0, we have not found that entropy loss makes a large difference in training.
- `workers`: The number of workers to use for parallel sampling.
- `backprop-workers`: The number of workers to use for backprop (i.e what to use for pytorch.set_num_threads()). If set to -1 then will automatically decide the best number of workers to the fastest backprop time.
- `redis`: Ray redis address
- `previous`: Path to a previously trained policy to bootstrap from. Will load this as the initial policy instead of a randomly initialized network. Does not overwrite the previous policy.
- `save-freq`: How often to save the model. The best model seen so far will always be saved no matter what, but this can be used to save a copy of the policy every `x` iterations. Set to -1 to disable this saving.

Also note that there are some algorithm wide arguments, which apply to all algos and are defined in [`train.py`](../train.py). See [here](../README.md#L43) for details.

## AlgoWorker Class
Training algorithms use the generic `AlgoWorker` template class which just holds onto a copy of both the actor and critic, along with a function for syncing it's own copy with new network parameters.

## Sampling
Sampling is implemented in [`sampling.py`](util/sampling.py). It contains the `AlgoSampler` class, which is a ray remote class and is intended to be used with all future algorithms as well. It's main function is the [`sample_traj`](util/sampling.py#L287) function which will sample a trajectory with it's current policy. You can control what the maximum allowed trajectory length is, whether this is an evaluation trajectory or not (i.e. evaluate with deterministic policy or not), and wheter to update the policy's normalization parameters with the arguments `max_traj_len`, `do_eval`, and `update_normalization_param`. The worker will save the data at each step (state, action, reward, and critic value) into a custom `Buffer` object and output it at the end along with some timing information.

The `Buffer` object is meant to hold samples for policy optimization.  `push` is used by the `AlgoSampler` to push new states into the buffer. `push_additional_info` can be used to store additional information about the current state to be used later in optimization. For example, "privileged" information for use in the critic later or something can be added here. Note that at the end of each sampled trajectory, `sample_traj` calls `Buffer.end_trajectory`, which takes in a terminal value and calculates the reward for each step by backing up return using the discount factor. Note that we use [Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf), and the gae values are calculated here as well. Both the discount factor and the gae lambda parameter are set at the construction of the `Buffer` object. `Buffer` also has a `sample` function that will return a randomly sampled batch and is used by the optimization. You can specify the batch size and whether the batch is recurrent or not. If the batch is recurrent, the function will sample whole *trajectories* instead of states. Note that this function will add zero padding to make all the trajectories the same length. In this recurrent case remember to adjust the batch size. When training FF networks a batch size of 10000 is reasonble, but for RNNs this is more equivalent to batch size of 32 trajectories (assuming trajectories are 300 steps long).

`Buffer` also overloads the "+" addition operator, so you can add two buffer objects together to combine their states.

Note that `Buffer` stores individual trajectories consecutively, i.e. "next to each other". Info is stored as a list of lists (and as only 2D Tensors when they are converted to Tensors in `_finish_buffer`). There is not "trajectory" dimension. Instead there is the `self.traj_idx` list which stores the indicies at which each trajectory starts/ends. The reason for this is that due to termination conditions sampled trajectories may be of different lengths. Rather than having a set 3rd dimension with a set trajectory length and then padding where necessary, we instead choose to only pad when we randomly sample batches from the buffer.

## Optimization
Our PPO implementation is in [`ppo.py`](ppo.py). It operates through the [`PPO`](ppo.py#L268) class which internally holds multiple `AlgoSamplers` and a [`PPOOptim`](ppo.py#L20) object for optimization.

`PPOOptim` splits the policy optimization into two functions, the high level `optimize` function and the worker function `_update_policy` which it internally calls. `optimize` is what the `PPO` class actually interacts with and takes in a `Buffer` object of sampled states and optimization parameters. We do minibatch sampling and multiple epochs. So for example we may specify 5 epochs and a batch size of 32, which means that we split the buffer randomly into pieces 32 size big, do a policy update with each one, then repeat the process 5 times. `optimize` handles splitting the buffer into random minibatches which then get passed to `_update_policy` to actually calculate the gradients and take a step. We follow the generic PPO update rule with a clipped objective and clipped gradient. There is also a KL divergence threshold: if after a minibatch update the policy has changed too much from the original policy (exceeded the KL threshold) we stop optimization early and don't do the rest of the minibatches/epochs.

You'll see some `mask` variables around in the policy optimization functions. This is the mask to deal with the zero padding that is used when optimizing recurrent policies (we have to sample whole trajectories in this case, and use zero padding to make sure all trajectories are of the same length). We don't want any of the padded states/actions to affect optimization, so all of the ratio, losses, and KL computations are masked to not include those elements.

## Mirror Loss
To encourage more symmetric motions to be learned during training we utilize a "mirror loss". Details can be read in the original [paper](https://www.cs.ubc.ca/~van/papers/2019-MIG-symmetry/index.html), but the main idea is that the policy should output "mirrorred" actions when we input "mirror" states. For example, let's say we input state `a` into the policy, and get out action `u`. Then, if we input the mirror of state `a`, `mirror(a)`, we should get out `mirror(u)`. "Mirroring" in this case means reflecting about the saggital plane (i.e. swapping the left and right arm/leg joint positions) and that is a self-inverse function (i.e. the mirror of the mirror state is just the original state itself, `mirror(mirror(a)) = a`). See the [envs](../env/README.md) for further details of the state mirroring is dealt with.

The mirror loss itself functions as follows: During policy optimization we have all of the states and their corresponding actions. We also have the mirror states. We then pass the mirror states into the policy to get the mirror actions. The mirror loss is then just the MSE loss between the originally sampled actions and the mirror of the mirror actions (since they should be equal).

## Logging
[`log.py`](util/log.py) takes care of all of creating directories/files to save all of the training logs. Policies will be saved in a directory `logdir/env-name/run-name/timestamp/` where `logdir` and `run-name` are user set arguments (see [`train.py`](../train.py)) and `env-name` is the name of the environment. This directory will contain all of the saved policy and critic `.pt` files as well as a `experiment.info` and `experiment.pkl` file. These files contain all of the training and environment arguments use and can be used to both recreate an identical env for evaluation as well as train an identical policy with the exact same hyperparameters. `experiment.info` is a human readable text file of all the arguments while `experiment.pkl` is a pickle to actually load in python.

The log directory will also contain training logs. We use [Tensorboard](https://www.tensorflow.org/tensorboard/get_started) for all of our training logging. We log evaluation stats (like test/train return and episode length), optimization stats (losses, KL div), as well as timing stats (sample time, optimization time, parallelization overhead). We highly suggest users use [Weight & Biases](https://wandb.ai/site) (or wandb) to view these tensorboard logs. This allows users to view the tensorboard graphs from anywhere via just a webpage, which is particularly useful when training on a remote machine like vlab. Create an account and then login on your machine with `wandb login` (you should already have wandb installed via the conda env created by the setup script). See [`train.py`](../train.py) and the home directory readme for wandb logging arguments when training.

If wandb is down or you need to use raw tensorboard for some reason you can always launch tensorboard manually with
```
tensorboard --logdir=/path/to/policy/logs --port=8888
```
(you can specify any open port number you want). Then you can just goto `localhost:8888` in your web browser to see the tensorboard output. If on vlab we recommend starting a [tmux](https://github.com/tmux/tmux/wiki) or [screen](https://linuxize.com/post/how-to-use-linux-screen/) session on the login node, and then start the tensorboard process with that so it will stay open. You can then port forward the vlab port to your local machine with
```
ssh -NfL localhost:8888:localhost:8888 user@ssh-iam.intel-research.net
```
Then you can goto `localhost:8888` on your local machine and view the vlab tensorboard process. Unfortunately you need to manually kill the ssh process when you are done (use `ps aux | grep -NfL` to search for the process and then use `kill -9` to kill it).