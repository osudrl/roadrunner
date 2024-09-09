# Digit AR Sim
### This module contains the required files to execute `ar-control` in simulation or on robot hardware.

These files provide an interface to Agility's control stack and API via Agility's Python SDK. 
`/llapi` allows user to read basic robot observations and send low level commands. 
However, when using LLAPI mode, most of the API calls will not be available to use,
since the user commands handle the robot entirely. 

### Compile
Install `ctypesgen` for generating the ctypes wrapper
```
pip install ctypesgen
```

Build the shared library to interface with Agility's control stack
```
./compile
```

### Run
First, download ar-control binary from Google Drive. Extract the binary file,
and run the binary by passing the TOML config file,
```
./ar-control /PATH_TO_THIS_DIRECTORY/digit-rl.toml
```
This will bring up the simulator and listen to the PySDK API and LLAPI commands.

Then, to run policy that interfaces with Agility's control stack, user has to go the root of the
repo and run the proper udp python script.
