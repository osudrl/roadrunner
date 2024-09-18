import argparse
import torch

from nn.critic import FFCritic, LSTMCritic
from nn.actor import FFActor, LSTMActor
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
    elif args.arch == 'ff':
        policy = FFActor(args.obs_dim,
                        args.action_dim,
                        std=std,
                        bounded=args.bounded,
                        layers=layers,
                        learn_std=args.learn_stddev,
                        nonlinearity=args.nonlinearity)
        critic = FFCritic(args.obs_dim, layers=layers)
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
            model.load_state_dict(val)
        elif hasattr(model, key):
            # avoid loading private attributes
            if not key.startswith('_'):
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
    }
    if isinstance(parser, argparse.ArgumentParser):
        nn_group = parser.add_argument_group("NN arguments")
        for arg, (default, help_str) in args.items():
            if isinstance(default, bool):   # Arg is bool, need action 'store_true' or 'store_false'
                nn_group.add_argument("--" + arg, default = default, action = "store_" + \
                                    str(not default).lower(), help = help_str)
            else:
                nn_group.add_argument("--" + arg, default = default, type = type(default), help = help_str)
    elif isinstance(parser, SimpleNamespace) or isinstance(parser, argparse.Namespace):
        for arg, (default, help_str) in args.items():
            arg = arg.replace("-", "_")
            if not hasattr(parser, arg):
                setattr(parser, arg, default)
    else:
        raise RuntimeError(f"{FAIL}nn_factory add_nn_args got invalid object type when trying " \
                           f"to add nn arguments. Input object should be either an " \
                           f"ArgumentParser or a SimpleNamespace.{ENDC}")

    return parser