from util.colors import OKGREEN, FAIL, ENDC
from types import SimpleNamespace


def test_all_algos():
    test_ppo()
    print(f"{OKGREEN}Passed all algo tests{ENDC}")

def test_ppo():
    from algo.ppo import add_algo_args, run_experiment

    print("Testing PPO training")
    args = SimpleNamespace(
        env_name     = "LocomotionClockEnv",
        robot_name   = "cassie",
        logdir       = "./trained_models/test/",
        wandb        = False,
        run_name     = None,
        nolog        = True,
        seed         = 0,
        traj_len     = 100,
        timesteps    = 200,
        num_steps    = 100,
        workers      = 2,
        mirror       = 1,
        backprop_workers = 1,
        arch         = "lstm"
    )

    add_algo_args(args)
    run_experiment(args, args.env_name)
    print(f"{OKGREEN}Passed PPO test{ENDC}")