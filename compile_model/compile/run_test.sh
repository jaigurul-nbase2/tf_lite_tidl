#!/bin/bash

# Set up environment for TIDL delegate
export LD_LIBRARY_PATH="./tidl_tools:$LD_LIBRARY_PATH"
export LD_PRELOAD="./tidl_tools/libvx_tidl_rt.so"
export TIDL_RT_DEBUG_LEVEL="4"
export TIDL_RT_TRACE_LEVEL="4"

# Run the test
python3 test.py "$@"