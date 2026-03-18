# Phase 1: CUDA API Interceptor & IPC Bridge

This directory contains the user-space `LD_PRELOAD` library used to intercept PyTorch CUDA memory allocations (`cudaMalloc`) and deallocations (`cudaFree`). It also establishes an Inter-Process Communication (IPC) bridge to send this telemetry data to a background scheduling daemon via UNIX Domain Sockets.

## Included Files
* `minimal_intercept.cpp`: The core C++ library that intercepts CUDA calls and sends JSON payloads over IPC.
* `fake_scheduler.py`: A mock background daemon that listens on `/tmp/nyx.sock` to verify IPC connectivity.
* `test.py`: A minimal PyTorch script designed to trigger VRAM allocations and force cache clears.
* `Makefile`: Build instructions for the shared object library.

## Build Dependencies
To compile this library, your Linux environment must have:
* **GCC/G++ Compiler** (Tested with standard Linux GNU toolchain)
* **NVIDIA CUDA Toolkit** (Requires `libcudart` and standard CUDA include headers, typically at `/usr/local/cuda`)
* **Python 3 & PyTorch** (For running the verification scripts)

## How to Compile
Run the standard `make` command, or manually compile using `g++`:

```bash
g++ -O3 -fPIC -shared minimal_intercept.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -ldl -o liborion_capture.so
