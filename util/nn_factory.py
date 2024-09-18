import argparse
import torch
import os

from nn.critic import FFCritic, LSTMCritic, GRUCritic, CNNLSTMCritic, CNNAddLSTMCritic
from nn.actor import FFActor, LSTMActor, GRUActor, MixActor, CNNLSTMActor, CNNAddLSTMActor
from nn.actor import FFConcatActor
from types import SimpleNamespace
from util.colors import FAIL, WARNING, ENDC

def nn_factory(args, env=None):
    """The nn_factory initializes a model class (actor, critic etc) by args (from saved pickle file
    or fresh new training). More cases can be added here to support different class types and init
    methods.

    Args:
        args (Namespace): Arguments for model class init.
        env (Env object, optional): Env object to get any env-relevant info to
            initialize modules. Defaults to None.

    Returns: actor and critic
    """
    # Unpack args with iterators
    layers = [int(x) for x in args.layers.split(',')]
    if args.std_array != "":
        args.std = args.std_array
        std = [float(x) for x in args.std.split(',')]
        assert len(std) == args.action_dim,\
               f"{FAIL}Std array size {len(std)} mismatch with action size {args.action_dim}.{ENDC}"
    else:
        std = args.std

    # Construct module class
    if args.arch == 'lstm':
        policy = LSTMActor(args.obs_dim,
                            args.action_dim,
                            std=std,
                            bounded=args.bounded,
                            layers=layers,
                            learn_std=args.learn_stddev)
        critic = LSTMCritic(args.obs_dim, layers=layers)
    elif args.arch == 'gru':
        policy = GRUActor(args.obs_dim,
                        args.action_dim,
                        std=std,
                        bounded=args.bounded,
                        layers=layers,
                        learn_std=args.learn_stddev)
        critic = GRUCritic(args.obs_dim, layers=layers)
    elif args.arch == 'ff':
        policy = FFActor(args.obs_dim,
                        args.action_dim,
                        std=std,
                        bounded=args.bounded,
                        layers=layers,
                        learn_std=args.learn_stddev,
                        nonlinearity=args.nonlinearity)
        critic = FFCritic(args.obs_dim, layers=layers)
    elif args.arch == 'mix':
        policy = MixActor(obs_dim=args.obs_dim,
                          state_dim=env.keywords['state_dim'],
                          nonstate_dim=env.keywords['nonstate_dim'],
                          action_dim=args.action_dim,
                          lstm_layers=layers,
                          ff_layers=layers,
                          bounded=args.bounded,
                          learn_std=args.learn_stddev,
                          std=std,
                          nonstate_encoder_dim=args.nonstate_encoder_dim,
                          nonstate_encoder_on=args.nonstate_encoder_on)
        critic = LSTMCritic(input_dim=args.obs_dim, layers=layers)
    elif args.arch == 'cnnconcat':
        policy = CNNLSTMActor(obs_dim=args.obs_dim,
                            action_dim=args.action_dim,
                            state_dim=env.state_dim,
                            layers=layers,
                            bounded=args.bounded,
                            learn_std=args.learn_stddev,
                            std=args.std,
                            image_shape=[32,32],
                            image_channel=1)
        critic = CNNLSTMCritic(obs_dim=args.obs_dim,
                            state_dim=env.state_dim,
                            layers=layers,
                            bounded=args.bounded,
                            learn_std=args.learn_stddev,
                            std=args.std,
                            image_shape=[32,32],
                            image_channel=1)
    elif args.arch == 'cnnadd':
        policy = CNNAddLSTMActor(obs_dim=args.obs_dim,
                                 action_dim=args.action_dim,
                                 state_dim=env.state_dim,
                                 base_actor_layer=[64,64],
                                bounded=args.bounded,
                                learn_std=args.learn_stddev,
                                std=args.std,
                                image_shape=[32,32],
                                image_channel=1)
        critic = CNNAddLSTMCritic(obs_dim=args.obs_dim,
                                 state_dim=env.state_dim,
                                 base_actor_layer=[64,64],
                                image_shape=[32,32],
                                image_channel=1)
        if args.load_actor:
            actor_dict = torch.load(os.path.join(args.load_actor, "actor.pt"))
            policy.load_base_actor(old_base_actor_dict=actor_dict['model_state_dict'])
    elif args.arch == 'ffconcat':
        use_cnn = False if args.load_actor == "" else True
        policy = FFConcatActor(obs_dim=args.obs_dim,
                                action_dim=args.action_dim,
                                state_dim=env.state_dim,
                                map_dim=env.map_dim,
                                state_layers=[64,32],
                                map_layers=[64,32],
                                concat_layers=[32],
                                bounded=args.bounded,
                                learn_std=args.learn_stddev,
                                std=args.std,
                                nonlinearity=args.nonlinearity,
                                use_cnn=use_cnn)
        critic = FFCritic(input_dim=args.obs_dim,
                        layers=[64,64])
        if args.load_actor:
            actor_dict = torch.load(os.path.join(args.load_actor, "actor.pt"))
            policy.load_base_model(base_model_dict=actor_dict['model_state_dict'])
    elif args.arch == 'fflstmconcat':
        from nn.actor import FFLSTMConcatActor
        policy = FFLSTMConcatActor(obs_dim=args.obs_dim,
                                    action_dim=args.action_dim,
                                    state_dim=env.state_dim,
                                    map_dim=env.map_dim,
                                    state_layers=[64,32],
                                    map_layers=[64,32],
                                    concat_layers=[32],
                                    bounded=args.bounded,
                                    learn_std=args.learn_stddev,
                                    std=args.std,
                                    nonlinearity=args.nonlinearity,
                                    use_cnn=False)
        if hasattr(args, 'use_privilege_critic'):
            critic = LSTMCritic(input_dim=env.privilege_obs_size,
                                layers=[64,64],
                                use_privilege_critic=args.use_privilege_critic)
        else:
            critic = LSTMCritic(input_dim=args.obs_dim,
                                layers=[64,64])
    elif args.arch == 'fflstmconcatcnn':
        from nn.actor import FFLSTMConcatActor
        policy = FFLSTMConcatActor(obs_dim=args.obs_dim,
                                    action_dim=args.action_dim,
                                    state_dim=env.state_dim,
                                    map_dim=env.map_dim,
                                    state_layers=[64,32],
                                    map_layers=[64,32],
                                    concat_layers=[32],
                                    bounded=args.bounded,
                                    learn_std=args.learn_stddev,
                                    std=args.std,
                                    nonlinearity=args.nonlinearity,
                                    use_cnn=True)
        critic = LSTMCritic(input_dim=args.obs_dim,
                            layers=[64,64])
        if args.load_actor:
            actor_dict = torch.load(os.path.join(args.load_actor, "actor.pt"))
            policy.load_base_model(base_model_dict=actor_dict['model_state_dict'])
    elif args.arch == 'residual':
        from nn.actor import ResidualActor
        # Load the base actor always
        assert args.load_actor != "", "load-actor must be specified for residual actor"
        base_actor_dict = torch.load(os.path.join(args.load_actor, "actor.pt"))
        # Extract base layer from the saved model dict
        base_layer = []
        for k,v in base_actor_dict['model_state_dict'].items():
            if 'weight_hh' in k:
                base_layer.append(v.shape[1])
        policy = ResidualActor(obs_dim=args.obs_dim,
                                action_dim=args.action_dim,
                                state_dim=env.state_dim,
                                map_dim=env.map_dim,
                                base_action_dim=base_actor_dict['action_dim'],
                                # state_layers=[64,32],
                                # map_layers=[64,32],
                                # concat_layers=[32],
                                # state_layers=[64],
                                # map_layers=[256, 128],
                                # concat_layers=[64, 64],
                                state_layers=[64],
                                map_layers=[256, 128],
                                concat_layers=[128, 64],
                                # state_layers=[64],
                                # map_layers=[256, 128],
                                # concat_layers=[128, 128],
                                base_layers=base_layer,
                                bounded=args.bounded,
                                learn_std=args.learn_stddev,
                                std=args.std,
                                nonlinearity=args.nonlinearity,
                                use_cnn=False,
                                link_base_model=args.link_base_model)
        critic = LSTMCritic(input_dim=env.privilege_obs_size if args.use_privilege_critic else args.obs_dim,
                            layers=[128,128],
                            use_privilege_critic=args.use_privilege_critic)
        policy.load_base_model(base_model_dict=base_actor_dict['model_state_dict'])
    elif args.arch == 'residual-cnn':
        from nn.actor import ResidualActor
        # Load the base actor always
        assert args.load_actor != "", "load-actor must be specified for residual actor"
        base_actor_dict = torch.load(os.path.join(args.load_actor, "actor.pt"))
        # Extract base layer from the saved model dict
        base_layer = []
        for k,v in base_actor_dict['model_state_dict'].items():
            if 'weight_hh' in k and 'concat' not in k:
                base_layer.append(v.shape[1])
        policy = ResidualActor(obs_dim=args.obs_dim,
                                action_dim=args.action_dim,
                                state_dim=env.state_dim,
                                map_dim=env.map_dim,
                                state_layers=[64,32],
                                map_layers=[64,32],
                                concat_layers=[32],
                                base_layers=base_layer,
                                bounded=args.bounded,
                                learn_std=args.learn_stddev,
                                std=args.std,
                                nonlinearity=args.nonlinearity,
                                use_cnn=True,
                                link_base_model=args.link_base_model)
        critic = LSTMCritic(input_dim=env.privilege_obs_size if args.use_privilege_critic else args.obs_dim,
                            layers=[128,128],
                            use_privilege_critic=args.use_privilege_critic)
        # TODO: for student training, try to make this robust
        # Load previously trained student
        # Load teacher
        # Load base actor
        # policy.load_teacher_model(model_dict=base_actor_dict['model_state_dict'])
    elif args.arch == 'residual-v2':
        from nn.actor import ResidualActorV2
        # Load the base actor always
        assert args.load_actor != "", "load-actor must be specified for residual actor"
        base_actor_dict = torch.load(os.path.join(args.load_actor, "actor.pt"))
        # Extract base layer from the saved model dict
        base_layer = []
        for k,v in base_actor_dict['model_state_dict'].items():
            if 'weight_hh' in k:
                base_layer.append(v.shape[1])
        policy = ResidualActorV2(obs_dim=args.obs_dim,
                                action_dim=args.action_dim,
                                state_dim=env.state_dim,
                                map_dim=env.map_dim,
                                # state_layers=[128],
                                # map_feature_layers=[64],
                                # map_input_layer_dim=64,
                                # concat_layers=[128, 64],
                                state_layers=[64],
                                map_feature_layers=[64],
                                map_input_layer_dim=64,
                                concat_layers=[64],
                                base_layers=base_layer,
                                bounded=args.bounded,
                                learn_std=args.learn_stddev,
                                std=args.std,
                                nonlinearity=args.nonlinearity,
                                use_cnn=False,
                                link_base_model=args.link_base_model)
        critic = LSTMCritic(input_dim=env.privilege_obs_size if args.use_privilege_critic else args.obs_dim,
                            layers=[128,128],
                            use_privilege_critic=args.use_privilege_critic)
        policy.load_base_model(base_model_dict=base_actor_dict['model_state_dict'])
    elif args.arch == 'v3':
        from nn.actor import V3
        policy = V3(obs_dim=args.obs_dim,
                                action_dim=args.action_dim,
                                state_dim=env.state_dim,
                                map_dim=env.map_dim,
                                state_layers=[128, 64],
                                map_layers=[32],
                                concat_layers=[64, 64],
                                bounded=args.bounded,
                                learn_std=args.learn_stddev,
                                std=args.std,
                                nonlinearity=args.nonlinearity,
                                use_cnn=False)
        critic = LSTMCritic(input_dim=env.privilege_obs_size if args.use_privilege_critic else args.obs_dim,
                            layers=[128,128],
                            use_privilege_critic=args.use_privilege_critic)
    else:
        raise RuntimeError(f"Arch {args.arch} is not included, check the entry point.")

    return policy, critic

def load_checkpoint(model, model_dict: dict):
    """Load saved checkpoint (as dict) into a model definition. This process varies by use case ,
    but here tries to load all saved attributes from dict into the empty (or no-empty) model class.

    Args:
        model_dict (dict): A saved dict contains required attributes to initialize a model class.
        model: A model class, ie actor, critic, cnn etc. Thsi is not a direct nn.module, but a
               customized wrapper class with use-base dependent attributes.
    """
    # Create dict to check that all actor attributes are set
    model_vars = set()
    for var in vars(model):
        if var[0] != "_":
            model_vars.add(var)
    for key, val in model_dict.items():
        if key == "model_state_dict":
            # Hotfix for loading base model
            if hasattr(model, 'std'):
                val['std'] = getattr(model, 'std')
            model.load_state_dict(val)
        elif hasattr(model, key):
            # avoid loading private attributes
            if not key.startswith('_'):
                # Hotfix for std no saved as nn parameter
                if type(getattr(model, key)) == torch.nn.Parameter:
                    setattr(model, key, torch.nn.Parameter(val))
                else:
                    setattr(model, key, val)
        else:
            if key == 'model_class_name':
                pass
            else:
                print(
                    f"{FAIL}{key} in saved model dict, but model {model.__class__.__name__} "
                    f"has no such attribute.{ENDC}")
        model_vars.discard(key)
    # Double check that all model attributes are set
    if len(model_vars) != 0:
        miss_vars = ""
        for var in model_vars:
            if not var.startswith('_'):
                miss_vars += var + " "
        print(f"{WARNING}WARNING: Model attribute(s) {miss_vars}were not set.{ENDC}")

def save_checkpoint(model, model_dict: dict, save_path: str):
    """Save a checkpoint by dict from a model class.

    Args:
        model: Any model class
        model_dict (dict): Saved dict.
        save_path (str): Saving path.
    """
    # Loop thru keys to make sure get any updates from model class
    # Excludes private attributes starting with "_"
    for key in vars(model):
        if not key.startswith('_'):
            model_dict[key] = getattr(model, key)
    torch.save(model_dict | {'model_state_dict': model.state_dict()},
                save_path)

def add_nn_parser(parser: argparse.ArgumentParser | SimpleNamespace | argparse.Namespace):
    args = {
        "std" : (0.13, "Action noise std dev"),
        "bounded" : (False, "Whether or not actor policy has bounded output"),
        "layers" : ("256,256", "Hidden layer size for actor and critic"),
        "arch" : ("ff", "Actor/critic NN architecture"),
        "learn-stddev" : (False, "Whether or not to learn action std dev"),
        "nonlinearity" : ("relu", "Actor output layer activation function"),
        "std-array" : ("", "An array repsenting action noise per action."),
        "load-actor" : ("", "Load a previously trained actor."),
        "use_privilege_critic" : (False, "Whether or not to use asymmetric actor critic."),
        "link_base_model" : (False, "Concat last layer feature from base model to the actor."),
    }
    if isinstance(parser, argparse.ArgumentParser):
        nn_group = parser.add_argument_group("NN arguments")
        for arg, (default, help_str) in args.items():
            if isinstance(default, bool):   # Arg is bool, need action 'store_true' or 'store_false'
                nn_group.add_argument("--" + arg, action=argparse.BooleanOptionalAction)
            else:
                nn_group.add_argument("--" + arg, default = default, type = type(default), help = help_str)
        nn_group.set_defaults(bounded=False)
        nn_group.set_defaults(learn_stddev=False)
        nn_group.set_defaults(use_privilege_critic=False)
        nn_group.set_defaults(link_base_model=False)
    elif isinstance(parser, (SimpleNamespace, argparse.Namespace)):
        for arg, (default, help_str) in args.items():
            arg = arg.replace("-", "_")
            if not hasattr(parser, arg):
                setattr(parser, arg, default)
    else:
        raise RuntimeError(f"{FAIL}nn_factory add_nn_args got invalid object type when trying " \
                           f"to add nn arguments. Input object should be either an " \
                           f"ArgumentParser or a SimpleNamespace.{ENDC}")

    return parser