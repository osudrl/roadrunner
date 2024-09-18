import hashlib, os, pickle
import wandb
import datetime

from util.colors import BOLD, ORANGE, ENDC
from collections import OrderedDict

# Logger stores in trained_models by default
def create_logger(all_args, algo_args, env_args, nn_args):
    from torch.utils.tensorboard import SummaryWriter
    """Use hyperparms to set a directory to output diagnostic files."""

    arg_dict = all_args.__dict__
    assert "seed" in arg_dict, \
    "You must provide a 'seed' key in your command line arguments"
    assert "logdir" in arg_dict, \
    "You must provide a 'logdir' key in your command line arguments."
    assert "env_name" in arg_dict, \
    "You must provide a 'env_name' key in your command line arguments."

    # sort the keys so the same hyperparameters will always have the same hash
    arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))

    # remove seed so it doesn't get hashed, store value for filename
    # same for logging directory
    run_name = arg_dict.pop('run_name')
    seed = str(arg_dict.pop("seed"))
    logdir = str(arg_dict.pop('logdir'))
    env_name = str(arg_dict['env_name'])
    arg_dict.__dict__['previous_run_name'] = run_name
    arg_dict.__dict__['previous_seed'] = seed
    arg_dict.__dict__['previous_logdir'] = logdir

    # see if this run has a unique name, if so then that is going to be the name of the folder
    if run_name is not None:
        logdir = os.path.join(logdir, env_name)
        output_dir = os.path.join(logdir, run_name)
        # Check if policy name already exists. If it does, increment filename
        if os.path.exists(logdir) or os.path.exists(output_dir):
            logdir = os.path.join(logdir)+"/"+datetime.datetime.now().strftime("%m-%d-%H-%M")
            output_dir = os.path.join(output_dir)+"/"+datetime.datetime.now().strftime("%m-%d-%H-%M")
    else:
        # see if we are resuming a previous run, if we are mark as continued
        if hasattr(all_args, 'previous') and all_args.previous != "":
            logdir = os.path.join(logdir)+"/"+datetime.datetime.now().strftime("%m-%d-%H-%M")
            output_dir = all_args.previous[0:-1] + '-cont-' + datetime.datetime.now().strftime("%m-%d-%H-%M")
        else:
            # get a unique hash for the hyperparameter settings, truncated at 10 chars
            arg_hash   = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:6] + '-seed' + seed
            logdir     = os.path.join(logdir, env_name)
            output_dir = os.path.join(logdir, arg_hash)

    # create a directory with the hyperparm hash as its name, if it doesn't
    # already exist.
    os.makedirs(output_dir, exist_ok=True)

    # Create a file with all the hyperparam settings in human-readable plaintext,
    # also pickle file for resuming training easily
    info_path = os.path.join(output_dir, "experiment.info")
    pkl_path = os.path.join(output_dir, "experiment.pkl")
    with open(pkl_path, 'wb') as file:
        dict = {'all_args': all_args,
                'algo_args': algo_args,
                'env_args': env_args,
                'nn_args': nn_args}
        pickle.dump(dict, file)
    with open(info_path, 'w') as file:
        for key, val in arg_dict.items():
            file.write("%s: %s" % (key, val))
            file.write('\n')

    # wandb init before tensorboard.
    if all_args.wandb:
        wandb.init(group = all_args.run_name,
                   project=all_args.wandb_project_name,
                   config=all_args,
                   sync_tensorboard=True,
                   dir=all_args.wandb_dir)

    logger = SummaryWriter(output_dir, flush_secs=0.1) # flush_secs=0.1 actually slows down quite a bit, even on parallelized set ups
    print("Logging to " + BOLD + ORANGE + str(output_dir) + ENDC)

    logger.dir = output_dir
    return logger
