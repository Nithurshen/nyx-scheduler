#define _GNU_SOURCE
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

// Function pointers to hold the REAL CUDA functions
static cudaError_t (*real_cudaMalloc)(void **, size_t) = NULL;
static cudaError_t (*real_cudaFree)(void *) = NULL;
static cudaError_t (*real_cudaLaunchKernel)(const void *, dim3, dim3, void **, size_t, cudaStream_t) = NULL;

// Helper function to send JSON to the Rust scheduler and wait for a reply
void send_to_scheduler(const char *json_msg) {
  int sock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock < 0) return; 

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, "/tmp/nyx.sock", sizeof(addr.sun_path) - 1);

  if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
    send(sock, json_msg, strlen(json_msg), 0);
    char buffer[256];
    int n = recv(sock, buffer, sizeof(buffer) - 1, 0);
    if (n > 0) {
        buffer[n] = '\0';
    }
  }
  close(sock);
}

// 1. Intercept cudaMalloc
extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
  if (!real_cudaMalloc) {
    real_cudaMalloc = (cudaError_t(*)(void **, size_t))dlsym(RTLD_NEXT, "cudaMalloc");
  }
  char msg[256];
  snprintf(msg, sizeof(msg), "{\"action\": \"malloc\", \"bytes\": %zu}", size);
  send_to_scheduler(msg);
  return real_cudaMalloc(devPtr, size);
}

// 2. Intercept cudaFree
extern "C" cudaError_t cudaFree(void *devPtr) {
  if (!real_cudaFree) {
    real_cudaFree = (cudaError_t(*)(void *))dlsym(RTLD_NEXT, "cudaFree");
  }
  char msg[256];
  snprintf(msg, sizeof(msg), "{\"action\": \"free\", \"ptr\": \"%p\"}", devPtr);
  send_to_scheduler(msg);
  return real_cudaFree(devPtr);
}

// 3. NEW: Intercept cudaLaunchKernel (Compute Time-Slicing)
extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
  if (!real_cudaLaunchKernel) {
    real_cudaLaunchKernel = (cudaError_t(*)(const void *, dim3, dim3, void **, size_t, cudaStream_t))dlsym(RTLD_NEXT, "cudaLaunchKernel");
  }

  // Format the compute message utilizing the grid and block X dimensions
  char msg[256];
  snprintf(msg, sizeof(msg), "{\"action\": \"compute\", \"grid_x\": %u, \"block_x\": %u}", gridDim.x, blockDim.x);

  // Send to the background scheduler and wait for "Go"
  send_to_scheduler(msg);

  // Execute the actual GPU kernel launch
  return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}