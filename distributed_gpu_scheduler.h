/*
 * Distributed GPU Scheduler Header
 * Fault-tolerant load balancing for 4-GPU Hawkes Processes Engine
 */

#ifndef DISTRIBUTED_GPU_SCHEDULER_H
#define DISTRIBUTED_GPU_SCHEDULER_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Constants
#define MAX_GPUS 4

// GPU roles
#define ROLE_LEADER 0
#define ROLE_BACKUP 1
#define ROLE_WORKER 2
#define ROLE_FAILED 3

// Status structure for host queries
typedef struct {
    int current_leader;
    int backup_leader;
    int my_role;
    bool consensus_active;
    float system_throughput;
    int failed_gpu_count;
    
    bool gpu_alive[MAX_GPUS];
    int gpu_roles[MAX_GPUS];
    float gpu_health_scores[MAX_GPUS];
    float gpu_utilizations[MAX_GPUS];
    int queue_lengths[MAX_GPUS];
} DistributedSchedulerStatus;

// Host interface functions
cudaError_t initialize_distributed_scheduler(int num_gpus);

cudaError_t run_distributed_scheduling_cycle(
    int my_gpu_id,
    int* task_assignments,
    int n_tasks,
    float* gpu_metrics
);

cudaError_t get_distributed_scheduler_status(
    int my_gpu_id,
    DistributedSchedulerStatus* status
);

cudaError_t force_leader_election(int my_gpu_id);

#ifdef __cplusplus
}
#endif

#endif // DISTRIBUTED_GPU_SCHEDULER_H

