# Sim
This folder contains all of the simulation classes and functions. We primarily use [Mujoco](https://mujoco.org/) for our simulation, but also support use of the Agility provided libraries [cassie-mujoco-sim](https://github.com/osudrl/cassie-mujoco-sim) for more "realistic" Cassie simulation and ar-control sim for Digit (though the latter is only for evaluation)

## GenericSim
The goal of this sim setup is to abstract away the simulator type. The environment should be completely blind to what simulator is actually being underneath, so we can swap out simulator type without any issues and without changing any env code. To facilitate all sim classes will inherit from [`GenericSim`](generic_sim.py). `GenericSim` defines all of the basic simulator functions that the env might use from resetting, simulating forward, control, as well as getters and setters. Ideally, all envs will interact with these functions only, and will not use things specfic to a single sim type (like you can not assume that there is a Mujoco sim to reference `sim.data.qpos`).

## MujocoSim
[`MujocoSim`](mujoco_sim.py) handles interaction with the Mujoco simulator. It is robot generic and for the most part can be treated as an (incomplete) abstraction of the actual Mujoco code for our own purposes. The main additions we add on/enforce are for sim2real purposes, like things for dynamics randomization as well as torque delay and velocity based torque limits. This class contains pretty much all of the needed code, and the robot specific `MjCassieSim` and `MjDigitSim` that inherit `MujocoSim` just need to define robot specific variables like motor/joint indicies, PD gains, name lists, and torque parameters.

There is also the [`MujocoViewer`](mujoco_viewer.py) class that handles visualization of the Mujoco sim. This is used internally by `MujocoSim` and you do not need to interact with it, you will use the `MujocoSim` [viewer](mujoco_sim.py#L169) functions instead. During visualization press the F1, F2, and F3 keys to toggle through the help overlays. These will tell you all of the mouse and key interactivity available to you.

[`MujocoRender`](mujoco_render.py) is for camera rendering simulation, like RBG and depth rendering during simulation.

## LibCassieSim
[`LibCassieSim`](cassie_sim/lib_cassiesim.py) implements the C library version of the Cassie sim. It uses the [`cassie-mujoco-sim`](https://github.com/osudrl/cassie-mujoco-sim) C library which uses old C Mujoco version 2.10. This contains the code to actually communicate with the Cassie hardware as well. This C code gets compiled into the `libcassiemujoco.so` library which we write a Python wrapper around (see [`cassiemujoco`](cassie_sim/cassiemujoco/)). When visualizing using this simulator type (plus when using Cassie async sim) see the cassie-mujoco-sim [readme](https://github.com/osudrl/cassie-mujoco-sim) for interactivity instructions. Note that this is largely kept for legacy reasons (along for use with the async sim and hardware) and is largely deprecated and abandoned. Newer features may not be compatible and it is recommended to always use MujocoSim instead for training policies.


