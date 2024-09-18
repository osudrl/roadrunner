import argparse

from functools import partial
from importlib import import_module
from types import SimpleNamespace
from util.colors import FAIL, ENDC, WARNING

def env_factory(env_name: str, env_args: list | SimpleNamespace | argparse.Namespace):
    """
    Function to handle creating environment objects. Takes in a string of the name of the
    environment to create along with either a list of unhandled command line arguments or a
    Namespace of arguments to be passed to environment initialzation. Returns an environment partial
    function that can be called to instantiate a new environment object.

    Args:
        env_name (str): The name of the environment to create.
        env_args (list, SimpleNamespace, argparse.Namespace): Either a list of unhandled command
            line arguments (basically the second output of argparse.parse_known_args) or a Namespace
            of arguments. Both will be handled and used in environment creation. Any argument that
            is not in the list/Namespace but is defined in the environment's "add_env_args" function
            will be set to the default value as specified in the "add_env_args" function.

    Returns:
        partial function: Returns a partial function that will return a new environment object when
            called. For example, will return env_fn, and new env objects can be made with "env_fn()".
            The intention of this is mainly to be used in parallel settings, when the same single
            env partial function can be passed to multiple workers so each can create their own env
            object with the same initialization parameters.
    """

    try:
        env_module = import_module(f"env.tasks.{env_name.lower()}.{env_name.lower()}")
        env_class = getattr(env_module, env_name)
        if isinstance(env_args, list):
            env_argparse = argparse.ArgumentParser()
            env_argparse = add_env_parser(env_argparse)
            env_args, non_env_args = env_argparse.parse_known_args(env_args)
            if len(non_env_args) > 0:
                print(f"{WARNING}env_factory got non env args {non_env_args}.{ENDC}")
        elif isinstance(env_args, SimpleNamespace) or isinstance(env_args, argparse.Namespace):
            env_args = add_env_parser(env_name, env_args)
        else:
            raise RuntimeError(f"{FAIL}env_factory must receive either a list of un-parsed args " \
                               f"or a SimpleNamespace of already parsed args.{ENDC}")
        return partial(env_class, **vars(env_args))
    except:
        raise RuntimeError(f"Cannot locate env with name {env_name}.\n"
                           f"Check if modules names are aligned exactly the same from class to folder names.")

def add_env_parser(env_name, parser, is_eval: bool = False):
    """
    Function to add handling of arguments relevant to an environment construction. Handles both
    the case where the input is an argument parser (in which case it will use `add_argument`) and
    the case where the input is just a Namespace (in which it will just add to the namespace with
    the default values) Note that arguments that already exist in the namespace will not be
    overwritten. To add new arguments if needed, they can just be added to the `args` dictionary
    which should map arguments to the tuple pair (default value, help string).

    Args:
        parser (argparse.ArgumentParser or SimpleNamespace, or argparse.Namespace): The argument
            parser or Namespace object to add arguments to
        is_eval (bool): Whether the environment is being used for evaluation with new env args passed in
            command line. If True it will supress loading default env args.

    Returns:
        argparse.ArgumentParser or SimpleNamespace, or argparse.Namespace: Returns the same object
            as the input but with added arguments.
    """
    try:
        module = import_module(f"env.tasks.{env_name.lower()}.{env_name.lower()}")
        args = getattr(module, env_name).get_env_args()
        if isinstance(parser, argparse.ArgumentParser):
            env_group = parser.add_argument_group("Env arguments")
            for arg, (default, help_str) in args.items():
                # Supress loading default env args if eval called with command line args. However, still
                # want to use DR and state noise default to off
                if is_eval and arg not in ("dynamics-randomization", "state-noise"):
                    if isinstance(default, bool):   # Arg is bool, need action 'store_true' or 'store_false'
                        env_group.add_argument("--" + arg, action=argparse.BooleanOptionalAction,
                                            default = argparse.SUPPRESS)
                    elif isinstance(default, list): # Arg is list, need to use `nargs`
                        env_group.add_argument("--" + arg, nargs=len(default), default=argparse.SUPPRESS,
                                            type=type(default[0]), help=help_str)
                    else:
                        env_group.add_argument("--" + arg, default = argparse.SUPPRESS,
                                            type = type(default), help = help_str)
                else:
                    if isinstance(default, bool):   # Arg is bool, need action 'store_true' or 'store_false'
                        env_group.add_argument("--" + arg, action=argparse.BooleanOptionalAction, default = default)
                    elif isinstance(default, list): # Arg is list, need to use `nargs`
                        env_group.add_argument("--" + arg, nargs=len(default), default=default,
                                            type=type(default[0]), help=help_str)
                    else:
                        env_group.add_argument("--" + arg, default = default, type = type(default), help = help_str)

            # If eval set DR to false by default
            if is_eval:
                env_group.set_defaults(dynamics_randomization=False)
        elif isinstance(parser, (SimpleNamespace, argparse.Namespace)):
            for arg, (default, help_str) in args.items():
                arg = arg.replace("-", "_")
                if not hasattr(parser, arg):
                    setattr(parser, arg, default)
        else:
            raise RuntimeError(f"{FAIL}Environment add_env_args got invalid object type when trying " \
                            f"to add environment arguments. Input object should be either an " \
                            f"ArgumentParser or a SimpleNamespace.{ENDC}")
    except:
        raise RuntimeError(f"Cannot locate env with name {env_name}.\n"
                           f"Check if modules names are aligned exactly the same from class to folder names.")

    return parser