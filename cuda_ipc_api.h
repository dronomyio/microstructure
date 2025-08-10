#ifndef CUDA_IPC_API_H
#define CUDA_IPC_API_H

// CUDA IPC API Header - Custom implementation for 4-GPU distributed processing
// This header provides Inter-Process Communication functionality for multi-GPU setups

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA IPC Memory Handle structure
typedef struct {
    char reserved[64];
} cudaIpcMemHandle_t;

// CUDA IPC Event Handle structure  
typedef struct {
    char reserved[64];
} cudaIpcEventHandle_t;

// CUDA IPC Memory Flags
typedef enum {
    cudaIpcMemLazyEnablePeerAccess = 0x1
} cudaIpcMemFlags;

// Function declarations for CUDA IPC operations

/**
 * @brief Get IPC memory handle for a device pointer
 * @param handle Pointer to IPC memory handle
 * @param devPtr Device pointer to get handle for
 * @return cudaError_t Error code
 */
cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr);

/**
 * @brief Open IPC memory handle
 * @param devPtr Pointer to device pointer
 * @param handle IPC memory handle
 * @param flags IPC memory flags
 * @return cudaError_t Error code
 */
cudaError_t cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int flags);

/**
 * @brief Close IPC memory handle
 * @param devPtr Device pointer to close
 * @return cudaError_t Error code
 */
cudaError_t cudaIpcCloseMemHandle(void* devPtr);

/**
 * @brief Get IPC event handle
 * @param handle Pointer to IPC event handle
 * @param event CUDA event
 * @return cudaError_t Error code
 */
cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle, cudaEvent_t event);

/**
 * @brief Open IPC event handle
 * @param event Pointer to CUDA event
 * @param handle IPC event handle
 * @return cudaError_t Error code
 */
cudaError_t cudaIpcOpenEventHandle(cudaEvent_t* event, cudaIpcEventHandle_t handle);

// Multi-GPU synchronization functions

/**
 * @brief Synchronize across multiple GPUs
 * @param num_gpus Number of GPUs to synchronize
 * @param gpu_ids Array of GPU IDs
 * @return cudaError_t Error code
 */
cudaError_t cudaMultiGpuSync(int num_gpus, int* gpu_ids);

/**
 * @brief Enable peer access between GPUs
 * @param gpu_id_1 First GPU ID
 * @param gpu_id_2 Second GPU ID
 * @return cudaError_t Error code
 */
cudaError_t cudaEnablePeerAccess(int gpu_id_1, int gpu_id_2);

/**
 * @brief Disable peer access between GPUs
 * @param gpu_id_1 First GPU ID
 * @param gpu_id_2 Second GPU ID
 * @return cudaError_t Error code
 */
cudaError_t cudaDisablePeerAccess(int gpu_id_1, int gpu_id_2);

/**
 * @brief Check if peer access is possible between GPUs
 * @param can_access Pointer to result
 * @param gpu_id_1 First GPU ID
 * @param gpu_id_2 Second GPU ID
 * @return cudaError_t Error code
 */
cudaError_t cudaDeviceCanAccessPeer(int* can_access, int gpu_id_1, int gpu_id_2);

// 4-GPU distributed processing specific functions

/**
 * @brief Initialize 4-GPU distributed processing
 * @param gpu_ids Array of 4 GPU IDs
 * @return cudaError_t Error code
 */
cudaError_t cuda4GpuInit(int* gpu_ids);

/**
 * @brief Cleanup 4-GPU distributed processing
 * @return cudaError_t Error code
 */
cudaError_t cuda4GpuCleanup();

/**
 * @brief Distribute data across 4 GPUs
 * @param data Host data pointer
 * @param size Data size in bytes
 * @param gpu_ptrs Array of 4 GPU pointers
 * @return cudaError_t Error code
 */
cudaError_t cuda4GpuDistributeData(void* data, size_t size, void** gpu_ptrs);

/**
 * @brief Gather results from 4 GPUs
 * @param gpu_ptrs Array of 4 GPU pointers
 * @param size Data size per GPU in bytes
 * @param result Host result pointer
 * @return cudaError_t Error code
 */
cudaError_t cuda4GpuGatherResults(void** gpu_ptrs, size_t size, void* result);

/**
 * @brief Load balance work across 4 GPUs
 * @param total_work Total amount of work
 * @param work_distribution Array of 4 work amounts per GPU
 * @return cudaError_t Error code
 */
cudaError_t cuda4GpuLoadBalance(size_t total_work, size_t* work_distribution);

#ifdef __cplusplus
}
#endif

#endif // CUDA_IPC_API_H


