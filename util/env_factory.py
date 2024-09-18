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

    if "cassie" in env_name.lower():
        robot_type = "cassie"
    elif "digit" in env_name.lower():
        robot_type = "digit"
    else:
        raise RuntimeError("Please add options here to include a new robot type.")

    try:
        env_module = import_module(f"env.{robot_type}.{env_name.lower()}.{env_name.lower()}")
        env_class = getattr(env_module, env_name)
        if isinstance(env_args, list):
            env_argparse = argparse.ArgumentParser()
            env_argparse = env_module.add_env_args(env_argparse)
            env_args, non_env_args = env_argparse.parse_known_args(env_args)
            if len(non_env_args) > 0:
                print(f"{WARNING}env_factory got non env args {non_env_args}.{ENDC}")
        elif isinstance(env_args, SimpleNamespace) or isinstance(env_args, argparse.Namespace):
            env_args = env_module.add_env_args(env_args)
        else:
            raise RuntimeError(f"{FAIL}env_factory must receive either a list of un-parsed args " \
                               f"or a SimpleNamespace of already parsed args.{ENDC}")
        return partial(env_class, **vars(env_args))
    except:
        raise RuntimeError(f"Cannot locate env with name {env_name}.\n"
                           f"Check if modules names are aligned exactly the same from class to folder names.")

def add_env_parser(env_name, parser):
    if "cassie" in env_name.lower():
        robot_type = "cassie"
    elif "digit" in env_name.lower():
        robot_type = "digit"
    else:
        raise RuntimeError("Please add options here to include a new robot type.")

    try:
        env_module = import_module(f"env.{robot_type}.{env_name.lower()}.{env_name.lower()}")
        parser = env_module.add_env_args(parser)
        return parser
    except:
        raise RuntimeError(f"Cannot locate env with name {env_name}.\n"
                           f"Check if modules names are aligned exactly the same from class to folder names.")