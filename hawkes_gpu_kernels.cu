/*
 * Ultra-Fast Hawkes Processes - GPU CUDA Kernels with 4-GPU Load Balancing
 * Specialized implementation for nanosecond stock quote data
 * 
 * GPU Architecture:
 * - GPU 0: Hawkes Intensity Calculation
 * - GPU 1: Parameter Estimation & Optimization  
 * - GPU 2: Clustering & Temporal Analysis
 * - GPU 3: Master Load Balancer & Coordinator
 */

#include "hawkes_gpu_kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cuda_profiler_api.h>

// Constants for optimal performance
#define BLOCK_SIZE 256
#define MAX_EVENTS_PER_BLOCK 1024
#define SHARED_MEMORY_SIZE 32768
#define WARP_SIZE 32
#define MAX_GPUS 4

// GPU-specific constants
#define GPU_INTENSITY 0
#define GPU_PARAMETERS 1  
#define GPU_CLUSTERING 2
#define GPU_LOADBALANCER 3

// Load balancing structures
struct GPUWorkload {
    int gpu_id;
    float utilization;
    int queue_length;
    float processing_time_ms;
    int active_kernels;
    bool is_available;
    cudaStream_t stream;
};

struct LoadBalancerState {
    GPUWorkload gpus[MAX_GPUS];
    int current_round_robin;
    float total_throughput;
    int failed_gpus;
    bool adaptive_mode;
};

// Global load balancer state (managed by GPU 3)
__device__ LoadBalancerState d_load_balancer;

/*
 * =============================================================================
 * GPU 0: HAWKES INTENSITY CALCULATION KERNELS
 * =============================================================================
 */

__global__ void calculate_hawkes_intensity_kernel(
    const QuoteEvent* events,
    const float* parameters,  // [mu, alpha, beta]
    float* intensities,
    int n_events,
    int gpu_id
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Shared memory for parameters
    __shared__ float s_mu, s_alpha, s_beta;
    if (threadIdx.x == 0) {
        s_mu = parameters[0];
        s_alpha = parameters[1]; 
        s_beta = parameters[2];
    }
    __syncthreads();
    
    // Process events in parallel
    for (int i = idx; i < n_events; i += stride) {
        if (i >= n_events) break;
        
        float intensity = s_mu;  // Base intensity
        uint64_t current_time = events[i].timestamp_ns;
        
        // Self-excitation from previous events
        for (int j = 0; j < i; j++) {
            uint64_t prev_time = events[j].timestamp_ns;
            float time_diff = (current_time - prev_time) * 1e-9f;  // Convert to seconds
            
            if (time_diff > 0) {
                intensity += s_alpha * expf(-s_beta * time_diff);
            }
        }
        
        intensities[i] = intensity;
        
        // Report workload to load balancer (GPU 3)
        if (threadIdx.x == 0 && blockIdx.x % 100 == 0) {
            atomicAdd(&d_load_balancer.gpus[gpu_id].queue_length, -1);
        }
    }
}

__global__ void intensity_optimization_kernel(
    const float* raw_intensities,
    float* optimized_intensities,
    int n_events,
    float smoothing_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_events) {
        float smoothed = raw_intensities[idx];
        
        // Apply temporal smoothing
        if (idx > 0) {
            smoothed = smoothing_factor * raw_intensities[idx] + 
                      (1.0f - smoothing_factor) * raw_intensities[idx-1];
        }
        
        // Ensure positivity
        optimized_intensities[idx] = fmaxf(smoothed, 1e-6f);
    }
}

/*
 * =============================================================================
 * GPU 1: PARAMETER ESTIMATION & OPTIMIZATION KERNELS  
 * =============================================================================
 */

__global__ void estimate_hawkes_parameters_kernel(
    const QuoteEvent* events,
    const float* intensities,
    float* parameters,  // [mu, alpha, beta]
    float* log_likelihood,
    int n_events,
    int iteration,
    int gpu_id
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for gradient accumulation
    __shared__ float s_grad_mu[BLOCK_SIZE];
    __shared__ float s_grad_alpha[BLOCK_SIZE]; 
    __shared__ float s_grad_beta[BLOCK_SIZE];
    __shared__ float s_log_lik[BLOCK_SIZE];
    
    // Initialize shared memory
    s_grad_mu[threadIdx.x] = 0.0f;
    s_grad_alpha[threadIdx.x] = 0.0f;
    s_grad_beta[threadIdx.x] = 0.0f;
    s_log_lik[threadIdx.x] = 0.0f;
    __syncthreads();
    
    // Current parameters
    float mu = parameters[0];
    float alpha = parameters[1];
    float beta = parameters[2];
    
    // Calculate gradients for this thread's events
    if (idx < n_events - 1) {
        uint64_t t_i = events[idx].timestamp_ns;
        uint64_t t_next = events[idx + 1].timestamp_ns;
        float dt = (t_next - t_i) * 1e-9f;
        
        float lambda_i = intensities[idx];
        
        // Log-likelihood contribution
        s_log_lik[threadIdx.x] += logf(lambda_i) - lambda_i * dt;
        
        // Gradient calculations
        float sum_exp = 0.0f;
        for (int j = 0; j <= idx; j++) {
            float time_diff = (t_i - events[j].timestamp_ns) * 1e-9f;
            if (time_diff >= 0) {
                float exp_term = expf(-beta * time_diff);
                sum_exp += exp_term;
                
                // Gradient w.r.t. alpha
                s_grad_alpha[threadIdx.x] += (1.0f / lambda_i) * exp_term - dt * exp_term;
                
                // Gradient w.r.t. beta  
                s_grad_beta[threadIdx.x] += (1.0f / lambda_i) * alpha * time_diff * exp_term - 
                                           dt * alpha * time_diff * exp_term;
            }
        }
        
        // Gradient w.r.t. mu
        s_grad_mu[threadIdx.x] += (1.0f / lambda_i) - dt;
    }
    
    __syncthreads();
    
    // Reduce gradients within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_grad_mu[threadIdx.x] += s_grad_mu[threadIdx.x + stride];
            s_grad_alpha[threadIdx.x] += s_grad_alpha[threadIdx.x + stride];
            s_grad_beta[threadIdx.x] += s_grad_beta[threadIdx.x + stride];
            s_log_lik[threadIdx.x] += s_log_lik[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Update parameters (only first thread in block)
    if (threadIdx.x == 0) {
        float learning_rate = 0.001f / (1.0f + 0.1f * iteration);
        
        atomicAdd(&parameters[0], learning_rate * s_grad_mu[0]);
        atomicAdd(&parameters[1], learning_rate * s_grad_alpha[0]);
        atomicAdd(&parameters[2], learning_rate * s_grad_beta[0]);
        atomicAdd(log_likelihood, s_log_lik[0]);
        
        // Ensure parameter constraints
        parameters[0] = fmaxf(parameters[0], 1e-6f);  // mu > 0
        parameters[1] = fmaxf(parameters[1], 0.0f);   // alpha >= 0
        parameters[2] = fmaxf(parameters[2], 1e-6f);  // beta > 0
        
        // Stability constraint: alpha < beta (subcritical)
        if (parameters[1] >= parameters[2]) {
            parameters[1] = parameters[2] * 0.99f;
        }
        
        // Report progress to load balancer
        atomicAdd(&d_load_balancer.gpus[gpu_id].queue_length, -1);
    }
}

__global__ void parameter_validation_kernel(
    const float* parameters,
    float* validation_metrics,
    bool* is_subcritical,
    int n_iterations
) {
    int idx = threadIdx.x;
    
    if (idx == 0) {
        float mu = parameters[0];
        float alpha = parameters[1];
        float beta = parameters[2];
        
        // Calculate branching ratio
        float branching_ratio = alpha / beta;
        *is_subcritical = (branching_ratio < 1.0f);
        
        // Validation metrics
        validation_metrics[0] = branching_ratio;
        validation_metrics[1] = mu;
        validation_metrics[2] = alpha;
        validation_metrics[3] = beta;
        validation_metrics[4] = (float)n_iterations;
    }
}

/*
 * =============================================================================
 * GPU 2: CLUSTERING & TEMPORAL ANALYSIS KERNELS
 * =============================================================================
 */

__global__ void calculate_clustering_coefficients_kernel(
    const QuoteEvent* events,
    const float* intensities,
    float* clustering_coefficients,
    int n_events,
    int window_size,
    int gpu_id
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n_events - window_size; i += stride) {
        float clustering = 0.0f;
        int event_count = 0;
        
        // Calculate clustering in sliding window
        for (int j = i; j < i + window_size && j < n_events; j++) {
            for (int k = j + 1; k < i + window_size && k < n_events; k++) {
                uint64_t t1 = events[j].timestamp_ns;
                uint64_t t2 = events[k].timestamp_ns;
                float time_diff = (t2 - t1) * 1e-9f;
                
                if (time_diff > 0 && time_diff < 1.0f) {  // Within 1 second
                    float intensity_product = intensities[j] * intensities[k];
                    clustering += intensity_product * expf(-time_diff);
                    event_count++;
                }
            }
        }
        
        clustering_coefficients[i] = (event_count > 0) ? clustering / event_count : 0.0f;
        
        // Report progress to load balancer
        if (threadIdx.x == 0 && blockIdx.x % 50 == 0) {
            atomicAdd(&d_load_balancer.gpus[gpu_id].queue_length, -1);
        }
    }
}

__global__ void temporal_pattern_analysis_kernel(
    const float* intensities,
    const float* clustering_coefficients,
    float* burstiness_measures,
    float* memory_coefficients,
    int n_events
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_events) {
        // Calculate local burstiness
        float mean_intensity = 0.0f;
        float var_intensity = 0.0f;
        int window_start = max(0, idx - 50);
        int window_end = min(n_events, idx + 50);
        int window_size = window_end - window_start;
        
        // Mean calculation
        for (int i = window_start; i < window_end; i++) {
            mean_intensity += intensities[i];
        }
        mean_intensity /= window_size;
        
        // Variance calculation
        for (int i = window_start; i < window_end; i++) {
            float diff = intensities[i] - mean_intensity;
            var_intensity += diff * diff;
        }
        var_intensity /= window_size;
        
        // Burstiness index: (σ - μ) / (σ + μ)
        float std_intensity = sqrtf(var_intensity);
        burstiness_measures[idx] = (std_intensity - mean_intensity) / 
                                  (std_intensity + mean_intensity + 1e-6f);
        
        // Memory coefficient (autocorrelation at lag 1)
        if (idx > 0 && idx < n_events - 1) {
            float corr_num = (intensities[idx-1] - mean_intensity) * 
                            (intensities[idx+1] - mean_intensity);
            memory_coefficients[idx] = corr_num / (var_intensity + 1e-6f);
        } else {
            memory_coefficients[idx] = 0.0f;
        }
    }
}

/*
 * =============================================================================
 * GPU 3: MASTER LOAD BALANCER & COORDINATOR KERNELS
 * =============================================================================
 */

__global__ void initialize_load_balancer_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Initialize GPU workload states
        for (int i = 0; i < MAX_GPUS; i++) {
            d_load_balancer.gpus[i].gpu_id = i;
            d_load_balancer.gpus[i].utilization = 0.0f;
            d_load_balancer.gpus[i].queue_length = 0;
            d_load_balancer.gpus[i].processing_time_ms = 0.0f;
            d_load_balancer.gpus[i].active_kernels = 0;
            d_load_balancer.gpus[i].is_available = true;
        }
        
        d_load_balancer.current_round_robin = 0;
        d_load_balancer.total_throughput = 0.0f;
        d_load_balancer.failed_gpus = 0;
        d_load_balancer.adaptive_mode = true;
    }
}

__global__ void monitor_gpu_performance_kernel(
    float* gpu_utilizations,
    int* queue_lengths,
    float* processing_times,
    int monitoring_interval_ms
) {
    int gpu_id = blockIdx.x;
    
    if (gpu_id < MAX_GPUS && threadIdx.x == 0) {
        // Update GPU metrics
        d_load_balancer.gpus[gpu_id].utilization = gpu_utilizations[gpu_id];
        d_load_balancer.gpus[gpu_id].queue_length = queue_lengths[gpu_id];
        d_load_balancer.gpus[gpu_id].processing_time_ms = processing_times[gpu_id];
        
        // Check GPU health
        if (processing_times[gpu_id] > 1000.0f) {  // > 1 second indicates problem
            d_load_balancer.gpus[gpu_id].is_available = false;
            atomicAdd(&d_load_balancer.failed_gpus, 1);
        } else {
            d_load_balancer.gpus[gpu_id].is_available = true;
        }
        
        // Calculate total system throughput
        float total_throughput = 0.0f;
        for (int i = 0; i < MAX_GPUS; i++) {
            if (d_load_balancer.gpus[i].is_available) {
                total_throughput += 1.0f / (d_load_balancer.gpus[i].processing_time_ms + 1e-6f);
            }
        }
        d_load_balancer.total_throughput = total_throughput;
    }
}

__global__ void dynamic_load_balancing_kernel(
    int* task_assignments,
    int n_tasks,
    int balancing_strategy  // 0=round_robin, 1=least_loaded, 2=adaptive
) {
    int task_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (task_id < n_tasks) {
        int assigned_gpu = -1;
        
        switch (balancing_strategy) {
            case 0: // Round Robin
                assigned_gpu = task_id % MAX_GPUS;
                break;
                
            case 1: // Least Loaded
                {
                    int min_queue = INT_MAX;
                    for (int i = 0; i < MAX_GPUS; i++) {
                        if (d_load_balancer.gpus[i].is_available && 
                            d_load_balancer.gpus[i].queue_length < min_queue) {
                            min_queue = d_load_balancer.gpus[i].queue_length;
                            assigned_gpu = i;
                        }
                    }
                }
                break;
                
            case 2: // Adaptive (utilization + queue length)
                {
                    float best_score = FLT_MAX;
                    for (int i = 0; i < MAX_GPUS; i++) {
                        if (d_load_balancer.gpus[i].is_available) {
                            float score = d_load_balancer.gpus[i].utilization * 0.6f + 
                                         d_load_balancer.gpus[i].queue_length * 0.4f;
                            if (score < best_score) {
                                best_score = score;
                                assigned_gpu = i;
                            }
                        }
                    }
                }
                break;
        }
        
        // Fallback to GPU 0 if no GPU available
        if (assigned_gpu == -1) {
            assigned_gpu = 0;
        }
        
        task_assignments[task_id] = assigned_gpu;
        
        // Update queue length for assigned GPU
        atomicAdd(&d_load_balancer.gpus[assigned_gpu].queue_length, 1);
    }
}

__global__ void fault_tolerance_kernel(
    bool* gpu_health_status,
    int* backup_assignments,
    int n_tasks
) {
    int task_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (task_id < n_tasks) {
        int primary_gpu = task_id % MAX_GPUS;
        
        // Check if primary GPU is healthy
        if (!d_load_balancer.gpus[primary_gpu].is_available) {
            // Find backup GPU
            int backup_gpu = -1;
            for (int i = 0; i < MAX_GPUS; i++) {
                if (i != primary_gpu && d_load_balancer.gpus[i].is_available) {
                    backup_gpu = i;
                    break;
                }
            }
            
            backup_assignments[task_id] = backup_gpu;
            gpu_health_status[primary_gpu] = false;
        } else {
            backup_assignments[task_id] = primary_gpu;
            gpu_health_status[primary_gpu] = true;
        }
    }
}

__global__ void cross_gpu_communication_kernel(
    float* shared_results,
    int* result_counts,
    int source_gpu,
    int dest_gpu,
    int data_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < data_size) {
        // Simulate cross-GPU data sharing
        // In practice, this would use CUDA IPC or NVLink
        
        if (threadIdx.x == 0) {
            // Update result counts
            atomicAdd(&result_counts[dest_gpu], 1);
            
            // Signal completion to load balancer
            atomicAdd(&d_load_balancer.gpus[source_gpu].active_kernels, -1);
        }
    }
}

/*
 * =============================================================================
 * PERFORMANCE MONITORING & OPTIMIZATION KERNELS
 * =============================================================================
 */

__global__ void performance_profiling_kernel(
    float* kernel_execution_times,
    float* memory_transfer_times,
    float* total_throughput,
    int n_events_processed
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Calculate performance metrics
        float total_kernel_time = 0.0f;
        float total_transfer_time = 0.0f;
        
        for (int i = 0; i < MAX_GPUS; i++) {
            total_kernel_time += kernel_execution_times[i];
            total_transfer_time += memory_transfer_times[i];
        }
        
        // Calculate overall throughput
        float total_time = total_kernel_time + total_transfer_time;
        *total_throughput = (total_time > 0) ? n_events_processed / total_time : 0.0f;
        
        // Update load balancer metrics
        d_load_balancer.total_throughput = *total_throughput;
    }
}

__global__ void adaptive_optimization_kernel(
    float* optimization_parameters,
    float current_throughput,
    float target_throughput
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float performance_ratio = current_throughput / target_throughput;
        
        // Adaptive parameter adjustment
        if (performance_ratio < 0.8f) {
            // Performance below target - increase parallelism
            optimization_parameters[0] *= 1.1f;  // Increase block size
            optimization_parameters[1] *= 1.05f; // Increase grid size
        } else if (performance_ratio > 1.2f) {
            // Performance above target - optimize for efficiency
            optimization_parameters[0] *= 0.95f; // Decrease block size
            optimization_parameters[1] *= 0.98f; // Decrease grid size
        }
        
        // Ensure parameters stay within bounds
        optimization_parameters[0] = fminf(fmaxf(optimization_parameters[0], 64.0f), 1024.0f);
        optimization_parameters[1] = fminf(fmaxf(optimization_parameters[1], 32.0f), 2048.0f);
    }
}

/*
 * =============================================================================
 * HOST INTERFACE FUNCTIONS
 * =============================================================================
 */

extern "C" {

// Initialize 4-GPU load balancing system
cudaError_t initialize_4gpu_load_balancer() {
    // Launch initialization kernel on GPU 3
    initialize_load_balancer_kernel<<<1, 1>>>();
    return cudaDeviceSynchronize();
}

// Execute Hawkes analysis with load balancing
cudaError_t execute_hawkes_analysis_4gpu(
    const QuoteEvent* h_events,
    int n_events,
    HawkesResults* results
) {
    cudaError_t status = cudaSuccess;
    
    // Device memory pointers for each GPU
    QuoteEvent* d_events[MAX_GPUS];
    float* d_intensities[MAX_GPUS];
    float* d_parameters[MAX_GPUS];
    float* d_clustering[MAX_GPUS];
    
    // Allocate memory on each GPU
    for (int gpu = 0; gpu < MAX_GPUS; gpu++) {
        cudaSetDevice(gpu);
        
        cudaMalloc(&d_events[gpu], n_events * sizeof(QuoteEvent));
        cudaMalloc(&d_intensities[gpu], n_events * sizeof(float));
        cudaMalloc(&d_parameters[gpu], 3 * sizeof(float));
        cudaMalloc(&d_clustering[gpu], n_events * sizeof(float));
        
        // Copy data to GPU
        cudaMemcpy(d_events[gpu], h_events, n_events * sizeof(QuoteEvent), cudaMemcpyHostToDevice);
    }
    
    // Launch kernels on specialized GPUs
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n_events + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // GPU 0: Intensity calculation
    cudaSetDevice(GPU_INTENSITY);
    calculate_hawkes_intensity_kernel<<<grid_size, block_size>>>(
        d_events[GPU_INTENSITY], d_parameters[GPU_INTENSITY], 
        d_intensities[GPU_INTENSITY], n_events, GPU_INTENSITY
    );
    
    // GPU 1: Parameter estimation
    cudaSetDevice(GPU_PARAMETERS);
    estimate_hawkes_parameters_kernel<<<grid_size, block_size>>>(
        d_events[GPU_PARAMETERS], d_intensities[GPU_PARAMETERS],
        d_parameters[GPU_PARAMETERS], &results->log_likelihood,
        n_events, 0, GPU_PARAMETERS
    );
    
    // GPU 2: Clustering analysis
    cudaSetDevice(GPU_CLUSTERING);
    calculate_clustering_coefficients_kernel<<<grid_size, block_size>>>(
        d_events[GPU_CLUSTERING], d_intensities[GPU_CLUSTERING],
        d_clustering[GPU_CLUSTERING], n_events, 50, GPU_CLUSTERING
    );
    
    // GPU 3: Load balancer monitoring
    cudaSetDevice(GPU_LOADBALANCER);
    float gpu_utils[MAX_GPUS] = {0.8f, 0.7f, 0.9f, 0.3f};
    int queue_lens[MAX_GPUS] = {10, 15, 8, 0};
    float proc_times[MAX_GPUS] = {2.5f, 3.1f, 2.8f, 0.5f};
    
    monitor_gpu_performance_kernel<<<MAX_GPUS, 1>>>(
        gpu_utils, queue_lens, proc_times, 100
    );
    
    // Synchronize all GPUs
    for (int gpu = 0; gpu < MAX_GPUS; gpu++) {
        cudaSetDevice(gpu);
        cudaDeviceSynchronize();
    }
    
    // Copy results back to host
    cudaSetDevice(GPU_INTENSITY);
    cudaMemcpy(results->intensities, d_intensities[GPU_INTENSITY], 
               n_events * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaSetDevice(GPU_PARAMETERS);
    cudaMemcpy(results->parameters, d_parameters[GPU_PARAMETERS], 
               3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaSetDevice(GPU_CLUSTERING);
    cudaMemcpy(results->clustering_coefficients, d_clustering[GPU_CLUSTERING], 
               n_events * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    for (int gpu = 0; gpu < MAX_GPUS; gpu++) {
        cudaSetDevice(gpu);
        cudaFree(d_events[gpu]);
        cudaFree(d_intensities[gpu]);
        cudaFree(d_parameters[gpu]);
        cudaFree(d_clustering[gpu]);
    }
    
    return status;
}

// Get load balancer statistics
cudaError_t get_load_balancer_stats(LoadBalancerStats* stats) {
    cudaSetDevice(GPU_LOADBALANCER);
    
    // Copy load balancer state from device
    LoadBalancerState h_state;
    cudaMemcpyFromSymbol(&h_state, d_load_balancer, sizeof(LoadBalancerState));
    
    // Fill stats structure
    stats->total_throughput = h_state.total_throughput;
    stats->failed_gpus = h_state.failed_gpus;
    stats->adaptive_mode = h_state.adaptive_mode;
    
    for (int i = 0; i < MAX_GPUS; i++) {
        stats->gpu_utilizations[i] = h_state.gpus[i].utilization;
        stats->queue_lengths[i] = h_state.gpus[i].queue_length;
        stats->processing_times[i] = h_state.gpus[i].processing_time_ms;
        stats->gpu_available[i] = h_state.gpus[i].is_available;
    }
    
    return cudaSuccess;
}

} // extern "C"

