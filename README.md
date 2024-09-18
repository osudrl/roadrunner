# Learning Decentralized Multi-Biped Control for Payload Transport
This code is for the paper [Learning Decentralized Multi-Biped Control for Payload Transport](https://arxiv.org/abs/2406.17279) by Bikram Pandit, Ashutosh Gupta, Mohitvishnu S. Gadde, Addison Johnson, Aayam Kumar Shrestha, Helei Duan, Jeremy Dao, and Alan Fern.

## Setup Instructions
Conda is required to run the setup script included with this repository.
To install conda on your machine, visit [this page](https://www.anaconda.com/download#downloads)
for download links.

To create a fresh conda env with all the necessary dependencies, simply run
```
chmod +x setup.sh
source setup.sh
```
at the root directory of this repository. This script will setup a new conda env, install some additional pip packages, and install mujoco210.

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

On vLab, go to `ppo_seeds.sh` and modify repo directory and conda env name, and then run
```
qsub ppo_seeds.sh
```
By default, this script will run the training using an exclusive node.
