#!/bin/bash

# Compile the C library
make interface

# Generate the ctypes wrapper for the C library
# Requiring pip install ctypesgen
ctypesgen -linterface lowlevelapi.h udp.h interface.h -o interface_ctypes.py

# Move the compiled library and the ctypes wrapper to the parent directory
mv interface.so ../
mv interface_ctypes.py ../