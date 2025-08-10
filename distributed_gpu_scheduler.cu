// Distributed GPU Scheduler for 4-GPU Hawkes Processing (FINAL CORRECTED VERSION)
// Handles load balancing and inter-GPU communication

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>

// 4-GPU distributed processing constants
#define MAX_GPUS 4
#define THREADS_PER_BLOCK 256

// GPU scheduler state
struct GPUSchedulerState {
    int num_active_gpus;
    int gpu_ids[MAX_GPUS];
    cudaStream_t streams[MAX_GPUS];
    void* device_memory[MAX_GPUS];
    size_t memory_allocated[MAX_GPUS];
    float gpu_utilization[MAX_GPUS];
    bool peer_access_enabled[MAX_GPUS][MAX_GPUS];
};

static GPUSchedulerState scheduler_state = {0};

// Initialize the distributed GPU scheduler
extern "C" bool initialize_distributed_scheduler() {
    cudaError_t error;
    
    // Get number of available GPUs
    int device_count;
    error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        printf("Error getting device count: %s\n", cudaGetErrorString(error));
        return false;
    }
    
    if (device_count < 4) {
        printf("Warning: Only %d GPUs available, need 4 for optimal performance\n", device_count);
    }
    
    scheduler_state.num_active_gpus = std::min(device_count, MAX_GPUS);
    
    // Initialize each GPU
    for (int i = 0; i < scheduler_state.num_active_gpus; i++) {
        scheduler_state.gpu_ids[i] = i;
        
        error = cudaSetDevice(i);
        if (error != cudaSuccess) {
            printf("Error setting device %d: %s\n", i, cudaGetErrorString(error));
            return false;
        }
        
        // Create stream for this GPU
        error = cudaStreamCreate(&scheduler_state.streams[i]);
        if (error != cudaSuccess) {
            printf("Error creating stream for GPU %d: %s\n", i, cudaGetErrorString(error));
            return false;
        }
        
        // Check GPU properties
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);
        if (error == cudaSuccess) {
            printf("GPU %d: %s, %zu MB memory\n", i, prop.name, prop.totalGlobalMem / (1024*1024));
        }
        
        // Initialize utilization tracking
        scheduler_state.gpu_utilization[i] = 0.0f;
    }
    
    // Enable peer access between GPUs
    for (int i = 0; i < scheduler_state.num_active_gpus; i++) {
        for (int j = 0; j < scheduler_state.num_active_gpus; j++) {
            if (i != j) {
                cudaSetDevice(i);
                int can_access;
                error = cudaDeviceCanAccessPeer(&can_access, i, j);
                if (error == cudaSuccess && can_access) {
                    error = cudaDeviceEnablePeerAccess(j, 0);
                    if (error == cudaSuccess || error == cudaErrorPeerAccessAlreadyEnabled) {
                        scheduler_state.peer_access_enabled[i][j] = true;
                        printf("Enabled peer access: GPU %d -> GPU %d\n", i, j);
                    }
                }
            }
        }
    }
    
    printf("Distributed GPU scheduler initialized with %d GPUs\n", scheduler_state.num_active_gpus);
    return true;
}

// Cleanup the distributed GPU scheduler
extern "C" void cleanup_distributed_scheduler() {
    for (int i = 0; i < scheduler_state.num_active_gpus; i++) {
        cudaSetDevice(i);
        
        // Free device memory
        if (scheduler_state.device_memory[i]) {
            cudaFree(scheduler_state.device_memory[i]);
            scheduler_state.device_memory[i] = nullptr;
        }
        
        // Destroy stream
        if (scheduler_state.streams[i]) {
            cudaStreamDestroy(scheduler_state.streams[i]);
        }
        
        // Disable peer access
        for (int j = 0; j < scheduler_state.num_active_gpus; j++) {
            if (i != j && scheduler_state.peer_access_enabled[i][j]) {
                cudaDeviceDisablePeerAccess(j);
                scheduler_state.peer_access_enabled[i][j] = false;
            }
        }
    }
    
    scheduler_state.num_active_gpus = 0;
    printf("Distributed GPU scheduler cleaned up\n");
}

// Distribute work across GPUs based on current utilization
extern "C" void distribute_work_load(size_t total_work, size_t* work_per_gpu) {
    if (scheduler_state.num_active_gpus == 0) {
        printf("Error: Scheduler not initialized\n");
        return;
    }
    
    // Simple equal distribution for now
    size_t base_work = total_work / scheduler_state.num_active_gpus;
    size_t remainder = total_work % scheduler_state.num_active_gpus;
    
    for (int i = 0; i < scheduler_state.num_active_gpus; i++) {
        work_per_gpu[i] = base_work + (i < remainder ? 1 : 0);
        printf("GPU %d: Assigned %zu work units\n", i, work_per_gpu[i]);
    }
}

// Synchronize all GPUs
extern "C" bool synchronize_all_gpus() {
    cudaError_t error;
    
    for (int i = 0; i < scheduler_state.num_active_gpus; i++) {
        error = cudaSetDevice(i);
        if (error != cudaSuccess) {
            printf("Error setting device %d: %s\n", i, cudaGetErrorString(error));
            return false;
        }
        
        error = cudaStreamSynchronize(scheduler_state.streams[i]);
        if (error != cudaSuccess) {
            printf("Error synchronizing GPU %d: %s\n", i, cudaGetErrorString(error));
            return false;
        }
    }
    
    return true;
}

// Allocate memory across all GPUs
extern "C" bool allocate_distributed_memory(size_t size_per_gpu) {
    cudaError_t error;
    
    for (int i = 0; i < scheduler_state.num_active_gpus; i++) {
        error = cudaSetDevice(i);
        if (error != cudaSuccess) {
            printf("Error setting device %d: %s\n", i, cudaGetErrorString(error));
            return false;
        }
        
        error = cudaMalloc(&scheduler_state.device_memory[i], size_per_gpu);
        if (error != cudaSuccess) {
            printf("Error allocating %zu bytes on GPU %d: %s\n", 
                   size_per_gpu, i, cudaGetErrorString(error));
            return false;
        }
        
        scheduler_state.memory_allocated[i] = size_per_gpu;
        printf("GPU %d: Allocated %zu MB\n", i, size_per_gpu / (1024*1024));
    }
    
    return true;
}

// Get scheduler statistics - FIXED VERSION (no decltype)
extern "C" void get_scheduler_stats(void* stats_ptr) {
    // Define the struct type properly to avoid decltype issues
    typedef struct {
        int num_gpus;
        float utilization[MAX_GPUS];
        size_t memory_allocated[MAX_GPUS];
        bool peer_access[MAX_GPUS][MAX_GPUS];
    } SchedulerStatsStruct;
    
    // Cast to the proper type - NO MORE DECLTYPE ERROR
    SchedulerStatsStruct* stats = (SchedulerStatsStruct*)stats_ptr;
    
    stats->num_gpus = scheduler_state.num_active_gpus;
    
    for (int i = 0; i < MAX_GPUS; i++) {
        stats->utilization[i] = scheduler_state.gpu_utilization[i];
        stats->memory_allocated[i] = scheduler_state.memory_allocated[i];
        
        for (int j = 0; j < MAX_GPUS; j++) {
            stats->peer_access[i][j] = scheduler_state.peer_access_enabled[i][j];
        }
    }
}

// Update GPU utilization metrics
extern "C" void update_gpu_utilization(int gpu_id, float utilization) {
    if (gpu_id >= 0 && gpu_id < scheduler_state.num_active_gpus) {
        scheduler_state.gpu_utilization[gpu_id] = utilization;
    }
}

// Get optimal GPU for new work based on current utilization
extern "C" int get_optimal_gpu() {
    if (scheduler_state.num_active_gpus == 0) {
        return -1;
    }
    
    int best_gpu = 0;
    float min_utilization = scheduler_state.gpu_utilization[0];
    
    for (int i = 1; i < scheduler_state.num_active_gpus; i++) {
        if (scheduler_state.gpu_utilization[i] < min_utilization) {
            min_utilization = scheduler_state.gpu_utilization[i];
            best_gpu = i;
        }
    }
    
    return best_gpu;
}

// Copy data between GPUs using peer access
extern "C" bool copy_between_gpus(int src_gpu, int dst_gpu, void* src_ptr, void* dst_ptr, size_t size) {
    if (!scheduler_state.peer_access_enabled[src_gpu][dst_gpu]) {
        printf("Peer access not enabled between GPU %d and GPU %d\n", src_gpu, dst_gpu);
        return false;
    }
    
    cudaError_t error = cudaSetDevice(src_gpu);
    if (error != cudaSuccess) {
        printf("Error setting source device %d: %s\n", src_gpu, cudaGetErrorString(error));
        return false;
    }
    
    error = cudaMemcpyPeer(dst_ptr, dst_gpu, src_ptr, src_gpu, size);
    if (error != cudaSuccess) {
        printf("Error copying between GPUs %d->%d: %s\n", src_gpu, dst_gpu, cudaGetErrorString(error));
        return false;
    }
    
    return true;
}


