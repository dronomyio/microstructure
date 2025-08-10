#ifndef DISTRIBUTED_GPU_SCHEDULER_H
#define DISTRIBUTED_GPU_SCHEDULER_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Constants
#define MAX_GPUS 4

// Scheduler statistics structure
struct SchedulerStats {
    int num_gpus;
    float utilization[MAX_GPUS];
    size_t memory_allocated[MAX_GPUS];
    bool peer_access[MAX_GPUS][MAX_GPUS];
};

// Initialization and cleanup
bool initialize_distributed_scheduler();
void cleanup_distributed_scheduler();

// Work distribution
void distribute_work_load(size_t total_work, size_t* work_per_gpu);
bool synchronize_all_gpus();

// Memory management
bool allocate_distributed_memory(size_t size_per_gpu);

// Statistics and monitoring
void get_scheduler_stats(void* stats_ptr);
void update_gpu_utilization(int gpu_id, float utilization);
int get_optimal_gpu();

// Inter-GPU communication
bool copy_between_gpus(int src_gpu, int dst_gpu, void* src_ptr, void* dst_ptr, size_t size);

#ifdef __cplusplus
}
#endif

#endif // DISTRIBUTED_GPU_SCHEDULER_H


