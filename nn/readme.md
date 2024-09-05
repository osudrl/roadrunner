# NN
This folder contains all of the neural network definitions we use to create our actor and critic networks. They are mostly made up of Pytorch's inbuilt `nn.Module` object.

## Base
[`base.py`](base.py) defines the super basic `Net` class, that enforces a `_base_forward` function as well as normalization functions. These are used to make up the `FFBase`, `LSTMBase` classes that will actually implement the `_base_forward` function. These are basically implementations of feedforward, LSTM archiectures that are compatibile with our code, and used as the basic building blocks to construct our actor and critic.

Note that `LSTMBase` uses Pytorch inbuilt `nn.LSTM` object, which only supports layers of equal size. For example the LSTM NN cannot have one layer of size 128 and then the second layer be 64, both layers must either be 128 or 64. This is usually not a problem, but if layers of non-equal sizes are absolutely needed users can use the depracted `LSTMBase_` class instead. This uses individual Pytorch `nn.LSTMCell` to build the LSTM ourselves. However, since we need to implement the inner loop ourselves this is much slower than `nn.LSTM` for both the forward pass and backprop, so only use this if absolutely needed.

Any new types of networks (such as CNN, transformers, etc.) that are needed will go here in the future.

## Actor
[`actor.py`](actor.py) defines the `Actor` base class which is meant to define a stochastic actor. It enforces a standard deviation (`self.std` and is given as input during construction) as well as a final linear layer to output means for each action dimension. Note that `std` can be either a single float which will be used for all action dimensions or a list/numpy array of individual std devs to use for each dimension. There is also an option to learn std devs, in which case an extra linear layer is used to output std devs for each action dimension. See [`nn_factory.py`](../util/nn_factory.py) and the [readme](../util/readme.md) for more details on how to use the NN construction.

The `Actor` main functions are `actor_forward` and `log_prob`. `actor_forward` performs a forward pass (either deterministic or stochastic depending on the arguments) utilizing the `_base_forward` function from the base NN classes.
`log_prob` computes the log probability of the current policy outputting the given action given the state and is used in optimization.

The actual policy actor classes that you will interact with `FFActor`, `LSTMActor`. These simply overload the `forward` function (that they inherit from Pytorch's `nn.Module`) using `actor_forward` so that you can call their forward pass with just `policy(state)`.

## Critic
[`critic.py`](critic.py) defines the `Critic` base class are the actual critic classes `FFCritic`, `LSTMCritic` and functions pretty much the same as the actor classes. In the critic case things are a bit simpler since there is not stochastic representation to take care of, the classes just need to implement a forward pass to output a single float of the value function estimate.