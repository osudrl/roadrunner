# Roadrunner Refactor

## Setup Instructions
Conda is required to run the setup script included with this repository.
To avoid licensing issues with Anaconda, it is recommended you install conda on your machine via
[Miniconda](https://docs.anaconda.com/miniconda/) rather than Anaconda.

To create a fresh conda env with all the necessary dependencies, simply run
```
chmod +x setup.sh
bash setup.sh
```
at the root directory of this repository. This script will setup a new conda env, install some additional pip packages, and install mujoco210.

You also need to have ar-software installed in order to run the Digit async simulation. Download it from the Google Drive [here](https://drive.google.com/file/d/1CbesyvdkN1s_V36csKk-cA5IdmaPGGlu/view?usp=sharing), unzip it and move it to your home directory.

You might need to install ffmpeg, with
```
sudo apt install ffmpeg
```

You should then be able to run the tests. Start with sim tests first, then env tests, then finally algo test. Run:
```
python test.py --all
```
Alternatively, you can run each test individually with the following commands:
```
python test.py --sim
python test.py --env
python test.py --algo
python test.py --nn
python test.py --render
python test.py --mirror
python test.py --timing
```
Note that for the sim test there is an intermittent seg fault issue with the libcassie viewer. If you get a segfault during libcassiesim test, you might have to try running it again a few times. We've found that it can sometimes happen if you close the viewer window too early.

## Training Instructions
Run the following to launch training on your local machine.
```
python run_ppo.py
```

You can also launch training from command with the `train.py` script. Algo generic arguments are defined in here while algo specific ones are defined in the corresponding algo file (see [`algo`](algo) for explanation on PPO arguments). `train.py` arguments are:
- `env-name`: The name of the environment to train on.
- `wandb`: Whether to use wandb for logging or not. By default is set to False.
- `wandb-project-name`: If using wandb, what is the project name to log to. By default is "roadrunner_refactor"
- `logdir`: The path of the directory to log and save files to.
- `run-name`: The name of the run/policy. Actor and critic `.pt` files, along with all logging files will be saved to the folder `logdir/env-name/run-name/timestamp/`. If a `run-name` is not provided, a hash string will be auto generated and used instead.
- `seed`: What to set the random seed to.
- `traj-len`: The maximum allowed trajectory length. During training, we will not collect samples beyond this point and will instead use the current critic to estimate the infinite horizon value.
- `timesteps`: How many total (for the *entire* training) timesteps to sample. Rather than train for a certain number of interations, we say that we are train using X amount of total samples. Often we just set this to be arbitrarily large like 5e9 to make the training run "forever" and then just stop the training manually ourselves when we see the learning curve plateau.

## Evaluation Instructions
After training a policy (or you can test with the provided policies in `./pretrained_models`) you can evaluate with the `eval.py` script. For example, run
```
python eval.py interactive --path ./pretrained_models/LocomotionEnv/cassie-LocomotionEnv/10-27-17-03/
```
to visualize and run a Cassie walking policy. Terminal printout will show a legend of keyboard commands along with what the current commands are. See `evaluation_factory` [documentation](util/readme.md#L15) for more details.

## Cassie Asynchronous Sim/Hardware Evaluation Instructions
To run Cassie policies in the asynchronous simulator or on the hardware, use the `cassie_udp.py` script. It simply takes in arguments of the policy path along with whether to do logging or not (logging is on by default). Note that you need to [cassie-mujoco-sim](https://github.com/osudrl/cassie-mujoco-sim) to run the async simulator. Once setup, run `./cassiesim -vxr` in the example directory as a separate process. This will open a visualization window, and you can then run `cassie_udp.py` and it should automatically connect to the cassiesim process.

## Structure Overview

The repo is split into 6 main folders. Each contains it's own readme with further documentation.
- [`algo`](algo): Contains all of the PPO implentation code. All algorithm/training code should go here.
- [`nn`](nn): Contains all of the neural network definitions used for both actors and critics. Implements things like FF networks, LSTM networks, etc.
- [`env`](env): Contains all of the environment definitions. Split into Cassie and Digit envs. Also contains all of the reward functions.
- [`sim`](sim): Contains all of the simulation classes that the environment use and interact with.
- [`testing`](testing): Contains all of the testing functions used for CI and debugging. Performance testing for policies will go here as well.
- [`util`](util): Contains repo wide utility functions. Only utilities that are used across multiple of the above folders, or in scripts at the top level should be here. Otherwise they should go into the corresponding folder's util folder.

## Notes
This repository is provided completely as is and is intended purely as an open sourcing of specific paper codebases and other related code. While things should work, functionality, future support and compatility, requests for changes/features, etc. are not guaranteed. Also note that this is very specific research code and was not made with total generality in mind like Gym or stable-baselines.