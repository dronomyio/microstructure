/*
 * Hawkes Processes GPU Kernels with Distributed Scheduler Integration
 * 4-GPU fault-tolerant architecture with redundant coordination
 * 
 * Integration with distributed_gpu_scheduler.cu for enterprise-grade reliability
 */

#include "hawkes_gpu_kernels.h"
#include "distributed_gpu_scheduler.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

// Include the distributed scheduler
#include "distributed_gpu_scheduler.cu"

/*
 * =============================================================================
 * HAWKES PROCESSES WITH DISTRIBUTED SCHEDULING HOOKS
 * =============================================================================
 */

// Enhanced Hawkes engine state with distributed scheduling
struct DistributedHawkesEngine {
    // Original Hawkes state
    HawkesEngineState hawkes_state;
    
    // Distributed scheduling state
    DistributedSchedulerStatus scheduler_status;
    int my_gpu_id;
    bool distributed_mode_enabled;
    
    // Performance tracking
    float processing_times[MAX_GPUS];
    float gpu_utilizations[MAX_GPUS];
    int task_queue_lengths[MAX_GPUS];
    
    // Fault tolerance
    bool emergency_mode;
    int backup_gpu_assignments[1000];  // Backup task assignments
};

// Global distributed Hawkes engine
__device__ DistributedHawkesEngine d_distributed_hawkes;

/*
 * =============================================================================
 * DISTRIBUTED HAWKES INITIALIZATION
 * =============================================================================
 */

__global__ void initialize_distributed_hawkes_kernel(int gpu_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Initialize distributed Hawkes engine
        d_distributed_hawkes.my_gpu_id = gpu_id;
        d_distributed_hawkes.distributed_mode_enabled = true;
        d_distributed_hawkes.emergency_mode = false;
        
        // Initialize performance metrics
        for (int i = 0; i < MAX_GPUS; i++) {
            d_distributed_hawkes.processing_times[i] = 0.0f;
            d_distributed_hawkes.gpu_utilizations[i] = 0.0f;
            d_distributed_hawkes.task_queue_lengths[i] = 0;
        }
        
        // Initialize backup assignments
        for (int i = 0; i < 1000; i++) {
            d_distributed_hawkes.backup_gpu_assignments[i] = gpu_id % MAX_GPUS;
        }
    }
}

/*
 * =============================================================================
 * DISTRIBUTED HAWKES PROCESSING KERNELS
 * =============================================================================
 */

__global__ void distributed_hawkes_intensity_kernel(
    const QuoteData* quotes,
    float* intensities,
    HawkesParameters* params,
    int n_quotes,
    int* task_assignments
) {
    int quote_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int my_gpu = d_distributed_hawkes.my_gpu_id;
    
    if (quote_idx < n_quotes) {
        // Check if this task is assigned to my GPU
        bool process_task = false;
        
        if (d_distributed_hawkes.distributed_mode_enabled) {
            // Use distributed task assignment
            if (task_assignments[quote_idx] == my_gpu) {
                process_task = true;
            }
        } else {
            // Fallback to simple round-robin
            if (quote_idx % MAX_GPUS == my_gpu) {
                process_task = true;
            }
        }
        
        if (process_task) {
            // HOOK: Report task start to distributed scheduler
            if (threadIdx.x == 0) {
                atomicAdd(&d_distributed_hawkes.task_queue_lengths[my_gpu], 1);
            }
            
            // Original Hawkes intensity calculation
            uint64_t current_time = quotes[quote_idx].sip_timestamp;
            float intensity = params->mu;  // Base intensity
            
            // Self-excitation from previous events
            for (int i = 0; i < quote_idx; i++) {
                if (quotes[i].sip_timestamp < current_time) {
                    float time_diff = (current_time - quotes[i].sip_timestamp) / 1e9f;  // Convert to seconds
                    float decay = expf(-params->beta * time_diff);
                    intensity += params->alpha * decay;
                }
            }
            
            intensities[quote_idx] = intensity;
            
            // HOOK: Report task completion to distributed scheduler
            if (threadIdx.x == 0) {
                atomicSub(&d_distributed_hawkes.task_queue_lengths[my_gpu], 1);
            }
        }
    }
}

__global__ void distributed_parameter_estimation_kernel(
    const QuoteData* quotes,
    const float* intensities,
    HawkesParameters* params,
    int n_quotes,
    int* task_assignments
) {
    int quote_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int my_gpu = d_distributed_hawkes.my_gpu_id;
    
    if (quote_idx < n_quotes) {
        // Check task assignment
        bool process_task = false;
        
        if (d_distributed_hawkes.distributed_mode_enabled) {
            if (task_assignments[quote_idx] == my_gpu) {
                process_task = true;
            }
        } else {
            if (quote_idx % MAX_GPUS == my_gpu) {
                process_task = true;
            }
        }
        
        if (process_task) {
            // HOOK: Performance monitoring start
            clock_t start_time = clock();
            
            // Parameter estimation using maximum likelihood
            // (Simplified gradient descent step)
            
            float log_likelihood = 0.0f;
            float gradient_alpha = 0.0f;
            float gradient_beta = 0.0f;
            float gradient_mu = 0.0f;
            
            // Calculate gradients
            for (int i = 0; i < n_quotes; i++) {
                if (intensities[i] > 0) {
                    log_likelihood += logf(intensities[i]);
                    
                    // Simplified gradient calculations
                    gradient_alpha += 1.0f / intensities[i];
                    gradient_beta += 1.0f / intensities[i];
                    gradient_mu += 1.0f / intensities[i];
                }
            }
            
            // Update parameters (simplified)
            float learning_rate = 0.001f;
            atomicAdd(&params->alpha, learning_rate * gradient_alpha / n_quotes);
            atomicAdd(&params->beta, learning_rate * gradient_beta / n_quotes);
            atomicAdd(&params->mu, learning_rate * gradient_mu / n_quotes);
            
            // Ensure parameter constraints
            if (params->alpha < 0) params->alpha = 0.01f;
            if (params->beta < 0) params->beta = 0.01f;
            if (params->mu < 0) params->mu = 0.01f;
            
            // HOOK: Performance monitoring end
            clock_t end_time = clock();
            float processing_time = (float)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0f;
            d_distributed_hawkes.processing_times[my_gpu] = processing_time;
        }
    }
}

__global__ void distributed_clustering_analysis_kernel(
    const QuoteData* quotes,
    const float* intensities,
    float* clustering_coefficients,
    int n_quotes,
    int* task_assignments
) {
    int quote_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int my_gpu = d_distributed_hawkes.my_gpu_id;
    
    if (quote_idx < n_quotes) {
        // Check task assignment
        bool process_task = false;
        
        if (d_distributed_hawkes.distributed_mode_enabled) {
            if (task_assignments[quote_idx] == my_gpu) {
                process_task = true;
            }
        } else {
            if (quote_idx % MAX_GPUS == my_gpu) {
                process_task = true;
            }
        }
        
        if (process_task) {
            // Calculate clustering coefficient for this quote
            float clustering = 0.0f;
            int window_size = 50;  // 50-quote window
            
            int start_idx = max(0, quote_idx - window_size);
            int end_idx = min(n_quotes - 1, quote_idx + window_size);
            
            // Calculate local clustering
            float mean_intensity = 0.0f;
            int count = 0;
            
            for (int i = start_idx; i <= end_idx; i++) {
                mean_intensity += intensities[i];
                count++;
            }
            mean_intensity /= count;
            
            // Clustering coefficient based on intensity variance
            float variance = 0.0f;
            for (int i = start_idx; i <= end_idx; i++) {
                float diff = intensities[i] - mean_intensity;
                variance += diff * diff;
            }
            variance /= count;
            
            clustering = variance / (mean_intensity + 1e-6f);  // Normalized variance
            clustering_coefficients[quote_idx] = clustering;
        }
    }
}

/*
 * =============================================================================
 * DISTRIBUTED COORDINATION AND FAULT TOLERANCE
 * =============================================================================
 */

__global__ void distributed_coordination_kernel(int my_gpu_id) {
    if (threadIdx.x == 0) {
        // Update GPU utilization
        float utilization = 0.0f;
        
        // Calculate utilization based on queue length and processing time
        int queue_length = d_distributed_hawkes.task_queue_lengths[my_gpu_id];
        float processing_time = d_distributed_hawkes.processing_times[my_gpu_id];
        
        utilization = min(1.0f, (queue_length / 100.0f) + (processing_time / 1000.0f));
        d_distributed_hawkes.gpu_utilizations[my_gpu_id] = utilization;
        
        // Check for emergency mode
        if (d_distributed_hawkes.scheduler_status.failed_gpu_count >= 2) {
            d_distributed_hawkes.emergency_mode = true;
            d_distributed_hawkes.distributed_mode_enabled = false;
        } else {
            d_distributed_hawkes.emergency_mode = false;
            d_distributed_hawkes.distributed_mode_enabled = true;
        }
    }
}

__global__ void emergency_fallback_hawkes_kernel(
    const QuoteData* quotes,
    float* intensities,
    HawkesParameters* params,
    int n_quotes
) {
    int quote_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int my_gpu = d_distributed_hawkes.my_gpu_id;
    
    if (quote_idx < n_quotes && d_distributed_hawkes.emergency_mode) {
        // Simple round-robin assignment in emergency mode
        if (quote_idx % MAX_GPUS == my_gpu) {
            // Basic Hawkes intensity calculation (simplified for emergency)
            uint64_t current_time = quotes[quote_idx].sip_timestamp;
            float intensity = params->mu;
            
            // Only consider recent events to reduce computation
            int lookback = min(100, quote_idx);  // Look back only 100 events
            
            for (int i = quote_idx - lookback; i < quote_idx; i++) {
                if (i >= 0 && quotes[i].sip_timestamp < current_time) {
                    float time_diff = (current_time - quotes[i].sip_timestamp) / 1e9f;
                    if (time_diff < 10.0f) {  // Only consider events within 10 seconds
                        float decay = expf(-params->beta * time_diff);
                        intensity += params->alpha * decay;
                    }
                }
            }
            
            intensities[quote_idx] = intensity;
        }
    }
}

/*
 * =============================================================================
 * HOST INTERFACE FUNCTIONS WITH DISTRIBUTED SCHEDULING
 * =============================================================================
 */

extern "C" {

// Initialize distributed Hawkes engine
cudaError_t initialize_distributed_hawkes_engine(int num_gpus) {
    cudaError_t status = cudaSuccess;
    
    // Initialize distributed scheduler first
    status = initialize_distributed_scheduler(num_gpus);
    if (status != cudaSuccess) return status;
    
    // Initialize Hawkes engine on each GPU
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        initialize_distributed_hawkes_kernel<<<1, 1>>>(gpu);
        cudaDeviceSynchronize();
    }
    
    return status;
}

// Execute Hawkes analysis with distributed scheduling
cudaError_t execute_distributed_hawkes_analysis(
    int my_gpu_id,
    const QuoteData* quotes,
    float* intensities,
    HawkesParameters* params,
    float* clustering_coefficients,
    int n_quotes
) {
    cudaSetDevice(my_gpu_id);
    
    // Allocate task assignments
    int* d_task_assignments;
    cudaMalloc(&d_task_assignments, n_quotes * sizeof(int));
    
    // Prepare GPU metrics for distributed scheduler
    float gpu_metrics[MAX_GPUS * 3];  // utilization, processing_time, queue_length
    
    // Get current metrics from device
    DistributedHawkesEngine h_engine;
    cudaMemcpyFromSymbol(&h_engine, d_distributed_hawkes, sizeof(DistributedHawkesEngine));
    
    for (int i = 0; i < MAX_GPUS; i++) {
        gpu_metrics[i * 3 + 0] = h_engine.gpu_utilizations[i];
        gpu_metrics[i * 3 + 1] = h_engine.processing_times[i];
        gpu_metrics[i * 3 + 2] = (float)h_engine.task_queue_lengths[i];
    }
    
    // HOOK: Run distributed scheduling cycle
    cudaError_t scheduler_status = run_distributed_scheduling_cycle(
        my_gpu_id, d_task_assignments, n_quotes, gpu_metrics
    );
    
    if (scheduler_status != cudaSuccess) {
        // Fallback to emergency mode
        emergency_fallback_hawkes_kernel<<<(n_quotes + 255) / 256, 256>>>(
            quotes, intensities, params, n_quotes
        );
    } else {
        // Normal distributed processing
        
        // Step 1: Distributed coordination
        distributed_coordination_kernel<<<1, 1>>>(my_gpu_id);
        
        // Step 2: Hawkes intensity calculation
        distributed_hawkes_intensity_kernel<<<(n_quotes + 255) / 256, 256>>>(
            quotes, intensities, params, n_quotes, d_task_assignments
        );
        
        // Step 3: Parameter estimation
        distributed_parameter_estimation_kernel<<<(n_quotes + 255) / 256, 256>>>(
            quotes, intensities, params, n_quotes, d_task_assignments
        );
        
        // Step 4: Clustering analysis
        distributed_clustering_analysis_kernel<<<(n_quotes + 255) / 256, 256>>>(
            quotes, intensities, clustering_coefficients, n_quotes, d_task_assignments
        );
    }
    
    // Cleanup
    cudaFree(d_task_assignments);
    
    return cudaDeviceSynchronize();
}

// Get distributed Hawkes engine status
cudaError_t get_distributed_hawkes_status(
    int my_gpu_id,
    DistributedHawkesStatus* status
) {
    cudaSetDevice(my_gpu_id);
    
    // Get distributed scheduler status
    DistributedSchedulerStatus scheduler_status;
    get_distributed_scheduler_status(my_gpu_id, &scheduler_status);
    
    // Get Hawkes engine state
    DistributedHawkesEngine h_engine;
    cudaMemcpyFromSymbol(&h_engine, d_distributed_hawkes, sizeof(DistributedHawkesEngine));
    
    // Fill status structure
    status->my_gpu_id = my_gpu_id;
    status->distributed_mode_enabled = h_engine.distributed_mode_enabled;
    status->emergency_mode = h_engine.emergency_mode;
    status->current_leader = scheduler_status.current_leader;
    status->backup_leader = scheduler_status.backup_leader;
    status->my_role = scheduler_status.my_role;
    status->system_throughput = scheduler_status.system_throughput;
    status->failed_gpu_count = scheduler_status.failed_gpu_count;
    
    for (int i = 0; i < MAX_GPUS; i++) {
        status->gpu_alive[i] = scheduler_status.gpu_alive[i];
        status->gpu_utilizations[i] = h_engine.gpu_utilizations[i];
        status->processing_times[i] = h_engine.processing_times[i];
        status->queue_lengths[i] = h_engine.task_queue_lengths[i];
    }
    
    return cudaSuccess;
}

// Force failover to backup GPU
cudaError_t force_hawkes_failover(int my_gpu_id) {
    cudaSetDevice(my_gpu_id);
    
    // Force leader election in distributed scheduler
    force_leader_election(my_gpu_id);
    
    // Update Hawkes engine state
    bool emergency_flag = true;
    cudaMemcpyToSymbol(d_distributed_hawkes, &emergency_flag, sizeof(bool),
                       offsetof(DistributedHawkesEngine, emergency_mode));
    
    return cudaDeviceSynchronize();
}

} // extern "C"

/*
 * =============================================================================
 * INTEGRATION HOOKS FOR ORIGINAL HAWKES KERNELS
 * =============================================================================
 */

// Hook function to be called from hawkes_gpu_kernels_4gpu_loadbalancer.cu
__device__ void distributed_scheduler_hook_pre_processing(int my_gpu_id) {
    // Update scheduler with current GPU status
    distributed_coordination_kernel<<<1, 1>>>(my_gpu_id);
}

__device__ void distributed_scheduler_hook_post_processing(int my_gpu_id, float processing_time) {
    // Update processing time metrics
    d_distributed_hawkes.processing_times[my_gpu_id] = processing_time;
}

__device__ bool distributed_scheduler_hook_should_process_task(int task_id, int my_gpu_id) {
    // Check if this GPU should process this task based on distributed assignment
    if (d_distributed_hawkes.distributed_mode_enabled) {
        // Use distributed task assignment (would need to be passed from host)
        return (task_id % MAX_GPUS == my_gpu_id);  // Simplified for now
    } else {
        // Emergency mode: simple round-robin
        return (task_id % MAX_GPUS == my_gpu_id);
    }
}

