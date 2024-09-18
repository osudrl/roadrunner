import torch

from nn.actor import FFActor, LSTMActor, MixActor
from nn.critic import FFCritic, LSTMCritic, MixCritic
from nn.base import FFBase, LSTMBase, MixBase
from util.colors import FAIL, ENDC, OKGREEN

state_size = 50
traj_length = 100
num_traj = 30
state_batch_size = (num_traj, traj_length, state_size)
x = torch.zeros((state_size,))
x_batch = torch.zeros(state_batch_size)
ff_layers = [128, 128]
lstm_layers = [64, 64]
action_size = 10

def test_base_module(base_partial):

    if base_partial == FFBase:
        ff_base = FFBase(in_dim=state_size, layers=ff_layers)
        assert ff_base._base_forward(x).size(dim=0) == ff_layers[-1], f"{ff_base.__class__.__name__} output wrong size."
        print("Passed FFBase")
    elif base_partial == LSTMBase:
        lstm_base = LSTMBase(in_dim=state_size, layers=lstm_layers)
        lstm_base.init_hidden_state()
        assert lstm_base._base_forward(x).size(dim=0) == lstm_layers[-1], f"{lstm_base.__class__.__name__} output wrong size."
        lstm_base.init_hidden_state()
        assert lstm_base._base_forward(x_batch).size(dim=0) == num_traj, f"{lstm_base.__class__.__name__} num traj output wrong size."
        assert lstm_base._base_forward(x_batch).size(dim=1) == traj_length, f"{lstm_base.__class__.__name__} traj length output wrong size."
        assert lstm_base._base_forward(x_batch).size(dim=2) == lstm_layers[-1], f"{lstm_base.__class__.__name__} output wrong size."
        print("Passed LSTMBase")
    elif base_partial == MixBase:
        mix_base = MixBase(in_dim=state_size, state_dim=40, nonstate_dim=10, lstm_layers=lstm_layers,
                            ff_layers=ff_layers, nonstate_encoder_dim=10)
        mix_base.init_hidden_state()
        assert mix_base._base_forward(x).size(dim=0) == ff_layers[-1], f"{lstm_base.__class__.__name__} output wrong size."
        mix_base.init_hidden_state()
        assert mix_base._base_forward(x_batch).size(dim=0) == num_traj, f"{lstm_base.__class__.__name__} num traj output wrong size."
        assert mix_base._base_forward(x_batch).size(dim=1) == traj_length, f"{lstm_base.__class__.__name__} traj length output wrong size."
        assert mix_base._base_forward(x_batch).size(dim=2) == ff_layers[-1], f"{lstm_base.__class__.__name__} output wrong size."
        print("Passed MixBase")
    else:
        raise RuntimeError(f"No such base module exists for {base_partial().__class__.__name__}")

def test_actor_module(actor_partial):

    if actor_partial == FFActor:
        ff_actor = FFActor(obs_dim=state_size, action_dim=action_size, layers=ff_layers,
                           bounded=False, learn_std=False, std=0.1, nonlinearity='tanh')
        action = ff_actor.forward(x, deterministic=False, update_normalization_param=False)
        assert action.size(dim=0) == action_size, f"{ff_actor.__class__.__name__} output wrong size."
        test_actor_forward(ff_actor)
        print("Pass forward test for FF Actor")
    elif actor_partial == LSTMActor:
        lstm_actor = LSTMActor(obs_dim=state_size, action_dim=action_size, layers=lstm_layers,
                               bounded=False, learn_std=False, std=0.1)
        action = lstm_actor.forward(x, deterministic=False, update_normalization_param=False)
        assert action.size(dim=0) == action_size, f"{lstm_actor.__class__.__name__} output wrong size."
        action = lstm_actor.forward(x_batch, deterministic=False, update_normalization_param=False)
        assert lstm_actor.forward(x_batch).size(dim=0) == num_traj, f"{lstm_actor.__class__.__name__} num traj output wrong size."
        assert lstm_actor.forward(x_batch).size(dim=1) == traj_length, f"{lstm_actor.__class__.__name__} traj length output wrong size."
        assert lstm_actor.forward(x_batch).size(dim=2) == action_size, f"{lstm_actor.__class__.__name__} output wrong size."
        test_actor_forward(lstm_actor)
        print("Pass forward test for LSTM Actor")
    elif actor_partial == MixActor:
        mix_actor = MixActor(obs_dim=state_size, state_dim=40, nonstate_dim=10, action_dim=action_size,
                             lstm_layers=lstm_layers, ff_layers=ff_layers, bounded=True,
                             learn_std=False, std=0.1, nonstate_encoder_dim=10, nonstate_encoder_on=True)
        action = mix_actor.forward(x, deterministic=False, update_normalization_param=False)
        assert action.size(dim=0) == action_size, f"{mix_actor.__class__.__name__} output wrong size."
        action = mix_actor.forward(x_batch, deterministic=False, update_normalization_param=False)
        assert mix_actor.forward(x_batch).size(dim=0) == num_traj, f"{mix_actor.__class__.__name__} num traj output wrong size."
        assert mix_actor.forward(x_batch).size(dim=1) == traj_length, f"{mix_actor.__class__.__name__} traj length output wrong size."
        assert mix_actor.forward(x_batch).size(dim=2) == action_size, f"{mix_actor.__class__.__name__} output wrong size."
        test_actor_forward(mix_actor)
        print("Pass forward test for Mix Actor")
    else:
        raise RuntimeError(f"No such base module exists for {actor_partial().__class__.__name__}")

def test_critic_module(critic_partial):

    if critic_partial == FFCritic:
        ff_critic = FFCritic(input_dim=state_size, layers=ff_layers)
        action = ff_critic.forward(x, update_normalization_param=False)
        assert action.size(dim=0) == 1, f"{ff_critic.__class__.__name__} output wrong size."
        print("Passed FF Critic")
    elif critic_partial == LSTMCritic:
        lstm_critic = LSTMCritic(input_dim=state_size, layers=lstm_layers)
        action = lstm_critic.forward(x, update_normalization_param=False)
        assert action.size(dim=0) == 1, f"{lstm_critic.__class__.__name__} output wrong size."
        action = lstm_critic.forward(x_batch, update_normalization_param=False)
        assert lstm_critic.forward(x_batch).size(dim=0) == num_traj, f"{lstm_critic.__class__.__name__} num traj output wrong size."
        assert lstm_critic.forward(x_batch).size(dim=1) == traj_length, f"{lstm_critic.__class__.__name__} traj length output wrong size."
        assert lstm_critic.forward(x_batch).size(dim=2) == 1, f"{lstm_critic.__class__.__name__} output wrong size."
        print("Passed LSTM Critic")
    elif critic_partial == MixCritic:
        mix_critic = MixCritic(input_dim=state_size, state_dim=40, nonstate_dim=10,
                            lstm_layers=lstm_layers, ff_layers=ff_layers, nonstate_encoder_dim=10, nonstate_encoder_on=True)
        action = mix_critic.forward(x, update_normalization_param=False)
        assert action.size(dim=0) == 1, f"{mix_critic.__class__.__name__} output wrong size."
        action = mix_critic.forward(x_batch, update_normalization_param=False)
        assert mix_critic.forward(x_batch).size(dim=0) == num_traj, f"{mix_critic.__class__.__name__} num traj output wrong size."
        assert mix_critic.forward(x_batch).size(dim=1) == traj_length, f"{mix_critic.__class__.__name__} traj length output wrong size."
        assert mix_critic.forward(x_batch).size(dim=2) == 1, f"{mix_critic.__class__.__name__} output wrong size."
        print("Passed Mix Critic")
    else:
        raise RuntimeError(f"No such base module exists for {critic_partial().__class__.__name__}")

def test_actor_forward(actor):
    # This essentially tests shared method for actors, regardless of actor type.
    if actor.is_recurrent: # reset actor
        actor.init_hidden_state()
    a1 = actor.forward(x, deterministic=False, update_normalization_param=False)
    a2 = actor.forward(x, deterministic=False, update_normalization_param=False)
    assert not torch.equal(a1, a2), "Stochastic forward result in the same output!"

    log_prob = actor.log_prob(state=x, action=torch.zeros((action_size)))
    assert not torch.isnan(log_prob), "Log prob is Nan!"
    assert torch.isreal(log_prob), "Log prob is not real number!"
    _, log_prob = actor.forward(x, deterministic=False, update_normalization_param=False,
                                   return_log_prob=True)
    assert not torch.isnan(log_prob), "Log prob is Nan!"
    assert torch.isreal(log_prob), "Log prob is not real number!"

    if actor.is_recurrent: # batch mode
        log_prob = actor.log_prob(state=x_batch, action=torch.zeros((action_size)))
        assert not torch.any(torch.isnan(log_prob)), "Log prob is Nan!"
        assert torch.any(torch.isreal(log_prob)), "Log prob is not real number!"
        _, log_prob = actor.forward(x_batch, deterministic=False, update_normalization_param=False,
                                    return_log_prob=True)
        assert not torch.any(torch.isnan(log_prob)), "Log prob is Nan!"
        assert torch.any(torch.isreal(log_prob)), "Log prob is not real number!"
        actor.init_hidden_state()

def test_nn():
    # Insert any module into this list to enable test. Can define customized test as well.
    # Some base modules require specific forward pass tests.
    # base_modules = [FFBase, LSTMBase, MixBase]
    # actor_modules = [FFActor, LSTMActor, MixActor]
    # critic_modules = [FFCritic, LSTMCritic, MixCritic]

    # for m in base_modules:
    #     test_base_module(m)
    # print(f"{OKGREEN}Passed all NN base tests! \u2713{ENDC}")

    # for a in actor_modules:
    #     test_actor_module(a)
    # print(f"{OKGREEN}Passed all NN actor tests! \u2713{ENDC}")

    # for c in critic_modules:
    #     test_critic_module(c)
    # print(f"{OKGREEN}Passed all NN critic tests! \u2713{ENDC}")
    # print(f"{OKGREEN}Passed all NN tests! \u2713{ENDC}")
    
    
    from nn.actor import CNNLSTMActor, CNNAddLSTMActor
    image_size = 32
    robot_state = 43
    obs_dim = int(image_size**2+robot_state)
    
    net = CNNLSTMActor(obs_dim=42+1054,
                action_dim=10,
                state_dim=42,
                layers=[64,64],
                bounded=False,
                learn_std=False,
                std=0.13,
                image_shape=[32,32],
                image_channel=1)
    net = CNNAddLSTMActor(obs_dim=obs_dim,
                action_dim=10,
                state_dim=robot_state,
                base_actor_layer=[64,64],
                bounded=False,
                learn_std=False,
                std=0.13,
                image_shape=[32,32],
                image_channel=1)
    print(net)
    net.eval() # call this for the batchnorm if single data
    x = torch.ones((obs_dim))
    print(net.forward(x).shape)
    x = torch.ones((5, 8, obs_dim))
    print(net.forward(x).shape)
