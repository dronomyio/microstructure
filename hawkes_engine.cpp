#include <cmath>
#include <algorithm>
#include "hawkes_engine.h"
#include "hawkes_gpu_kernels.h"
#include "hawkes_simd_processors.h"
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>

// Constructor
HawkesEngine::HawkesEngine(const HawkesEngineConfig& config) 
    : config_(config), gpu_initialized_(false), simd_processor_(nullptr) {
    
    // Initialize SIMD processor
    simd_processor_ = std::make_unique<HawkesSIMDProcessor>();
    
    // Initialize GPU if available
    if (config_.use_gpu) {
        initialize_gpu();
    }
    
    // Allocate host memory with alignment for SIMD
    allocate_host_memory();
    
    std::cout << "HawkesEngine initialized with:" << std::endl;
    std::cout << "  GPU: " << (gpu_initialized_ ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  SIMD: " << simd_processor_->get_simd_level_name() << std::endl;
    std::cout << "  Max events: " << config_.max_events << std::endl;
}

// Destructor
HawkesEngine::~HawkesEngine() {
    cleanup_gpu_memory();
    cleanup_host_memory();
}

// Initialize GPU resources
bool HawkesEngine::initialize_gpu() {
    cudaError_t error;
    
    // Check GPU availability
    int device_count;
    error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA-capable GPU found" << std::endl;
        return false;
    }
    
    // Set GPU device
    error = cudaSetDevice(config_.gpu_device_id);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set GPU device " << config_.gpu_device_id << std::endl;
        return false;
    }
    
    // Get GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, config_.gpu_device_id);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    
    // Allocate GPU memory
    size_t events_size = config_.max_events * sizeof(HawkesEvent);
    size_t matrix_size = config_.max_events * config_.max_events * sizeof(float);
    size_t intensities_size = config_.max_events * sizeof(float);
    
    // Events
    error = cudaMalloc(&d_events_, events_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for events" << std::endl;
        return false;
    }
    
    // Decay matrix
    error = cudaMalloc(&d_decay_matrix_, matrix_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for decay matrix" << std::endl;
        return false;
    }
    
    // Intensities
    error = cudaMalloc(&d_intensities_, intensities_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for intensities" << std::endl;
        return false;
    }
    
    // Log-likelihood terms
    error = cudaMalloc(&d_log_likelihood_terms_, intensities_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for log-likelihood terms" << std::endl;
        return false;
    }
    
    // Parameters
    error = cudaMalloc(&d_mu_, sizeof(float));
    error = cudaMalloc(&d_alpha_, sizeof(float));
    error = cudaMalloc(&d_beta_, sizeof(float));
    error = cudaMalloc(&d_gradients_, 3 * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for parameters" << std::endl;
        return false;
    }
    
    // Additional arrays
    error = cudaMalloc(&d_clustering_coeffs_, intensities_size);
    error = cudaMalloc(&d_residuals_, intensities_size);
    error = cudaMalloc(&d_simulated_events_, events_size);
    error = cudaMalloc(&d_n_simulated_, sizeof(int));
    error = cudaMalloc(&d_random_states_, sizeof(unsigned int));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate additional GPU memory" << std::endl;
        return false;
    }
    
    // Initialize random state
    unsigned int initial_seed = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    cudaMemcpy(d_random_states_, &initial_seed, sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    gpu_initialized_ = true;
    return true;
}

// Allocate aligned host memory for SIMD operations
void HawkesEngine::allocate_host_memory() {
    size_t events_size = config_.max_events * sizeof(HawkesEvent);
    size_t floats_size = config_.max_events * sizeof(float);
    
    // Allocate aligned memory (32-byte alignment for AVX-256, 64-byte for AVX-512)
    size_t alignment = 64;
    
    h_events_ = static_cast<HawkesEvent*>(aligned_alloc(alignment, events_size));
    h_intensities_ = static_cast<float*>(aligned_alloc(alignment, floats_size));
    h_decay_values_ = static_cast<float*>(aligned_alloc(alignment, floats_size));
    h_log_likelihood_terms_ = static_cast<float*>(aligned_alloc(alignment, floats_size));
    h_clustering_coeffs_ = static_cast<float*>(aligned_alloc(alignment, floats_size));
    h_residuals_ = static_cast<float*>(aligned_alloc(alignment, floats_size));
    h_time_diffs_ = static_cast<float*>(aligned_alloc(alignment, floats_size));
    
    if (!h_events_ || !h_intensities_ || !h_decay_values_ || 
        !h_log_likelihood_terms_ || !h_clustering_coeffs_ || 
        !h_residuals_ || !h_time_diffs_) {
        throw std::runtime_error("Failed to allocate aligned host memory");
    }
    
    // Initialize memory
    memset(h_events_, 0, events_size);
    memset(h_intensities_, 0, floats_size);
    memset(h_decay_values_, 0, floats_size);
    memset(h_log_likelihood_terms_, 0, floats_size);
    memset(h_clustering_coeffs_, 0, floats_size);
    memset(h_residuals_, 0, floats_size);
    memset(h_time_diffs_, 0, floats_size);
}

// Main analysis function
HawkesResults HawkesEngine::analyze_hawkes_process(
    const std::vector<HawkesEvent>& events,
    const HawkesParameters& initial_params
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (events.empty()) {
        throw std::invalid_argument("Events vector is empty");
    }
    
    if (events.size() > config_.max_events) {
        throw std::invalid_argument("Too many events for current configuration");
    }
    
    size_t n_events = events.size();
    
    // Copy events to aligned host memory
    std::copy(events.begin(), events.end(), h_events_);
    
    HawkesResults results;
    results.n_events = n_events;
    
    if (gpu_initialized_ && config_.use_gpu && n_events >= config_.gpu_threshold) {
        // GPU-accelerated processing
        results = process_with_gpu(n_events, initial_params);
    } else {
        // CPU/SIMD processing
        results = process_with_simd(n_events, initial_params);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    results.processing_time_ms = duration.count() / 1000.0;
    results.throughput_events_per_sec = n_events / (results.processing_time_ms / 1000.0);
    
    return results;
}

// GPU processing pipeline
HawkesResults HawkesEngine::process_with_gpu(size_t n_events, const HawkesParameters& initial_params) {
    HawkesResults results;
    
    // Transfer events to GPU
    size_t events_size = n_events * sizeof(HawkesEvent);
    cudaMemcpy(d_events_, h_events_, events_size, cudaMemcpyHostToDevice);
    
    // Initialize parameters on GPU
    float mu = initial_params.mu;
    float alpha = initial_params.alpha;
    float beta = initial_params.beta;
    
    cudaMemcpy(d_mu_, &mu, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha_, &alpha, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta_, &beta, sizeof(float), cudaMemcpyHostToDevice);
    
    // Step 1: Calculate exponential decay matrix
    launch_exponential_decay_kernel(d_events_, d_decay_matrix_, n_events, beta);
    cudaDeviceSynchronize();
    
    // Step 2: Calculate Hawkes intensities
    launch_hawkes_intensity_kernel(d_events_, d_decay_matrix_, d_intensities_, 
                                   n_events, mu, alpha, -1);  // -1 = all marks
    cudaDeviceSynchronize();
    
    // Step 3: Calculate log-likelihood
    uint64_t T_end = h_events_[n_events-1].timestamp_ns;
    launch_log_likelihood_kernel(d_events_, d_intensities_, d_log_likelihood_terms_,
                                 n_events, mu, alpha, beta, T_end);
    cudaDeviceSynchronize();
    
    // Step 4: Parameter estimation (if enabled)
    if (config_.estimate_parameters) {
        for (int iter = 0; iter < config_.max_iterations; iter++) {
            launch_parameter_estimation_kernel(d_events_, d_mu_, d_alpha_, d_beta_,
                                               d_gradients_, n_events, config_.learning_rate, iter);
            cudaDeviceSynchronize();
            
            // Check convergence (simplified)
            if (iter % 10 == 0) {
                float gradients[3];
                cudaMemcpy(gradients, d_gradients_, 3 * sizeof(float), cudaMemcpyDeviceToHost);
                float grad_norm = sqrtf(gradients[0]*gradients[0] + gradients[1]*gradients[1] + gradients[2]*gradients[2]);
                if (grad_norm < config_.convergence_threshold) {
                    break;
                }
            }
        }
        
        // Get updated parameters
        cudaMemcpy(&mu, d_mu_, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&alpha, d_alpha_, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&beta, d_beta_, sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    // Step 5: Calculate clustering coefficients
    launch_clustering_coefficient_kernel(d_events_, d_clustering_coeffs_, 
                                         n_events, config_.clustering_window_ns);
    cudaDeviceSynchronize();
    
    // Step 6: Calculate residuals for goodness-of-fit
    launch_residuals_kernel(d_events_, d_intensities_, d_residuals_, 
                            n_events, mu, alpha, beta);
    cudaDeviceSynchronize();
    
    // Transfer results back to host
    cudaMemcpy(h_intensities_, d_intensities_, n_events * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_log_likelihood_terms_, d_log_likelihood_terms_, n_events * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clustering_coeffs_, d_clustering_coeffs_, n_events * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_residuals_, d_residuals_, n_events * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Use SIMD for final statistical calculations
    results = finalize_results_with_simd(n_events, mu, alpha, beta);
    results.processing_method = "GPU + SIMD";
    
    return results;
}

// SIMD processing pipeline
HawkesResults HawkesEngine::process_with_simd(size_t n_events, const HawkesParameters& initial_params) {
    HawkesResults results;
    
    // Step 1: Calculate time differences
    for (size_t i = 0; i < n_events; i++) {
        h_time_diffs_[i] = 0.0f;
        for (size_t j = 0; j < i; j++) {
            h_time_diffs_[i] += (h_events_[i].timestamp_ns - h_events_[j].timestamp_ns) * 1e-9f;
        }
    }
    
    // Step 2: Calculate exponential decay using SIMD
    float beta = initial_params.beta;
    simd_processor_->calculate_exponential_decay(h_time_diffs_, h_decay_values_, n_events, beta);
    
    // Step 3: Calculate intensities (simplified CPU version)
    float mu = initial_params.mu;
    float alpha = initial_params.alpha;
    
    for (size_t i = 0; i < n_events; i++) {
        h_intensities_[i] = mu;
        for (size_t j = 0; j < i; j++) {
            float time_diff = (h_events_[i].timestamp_ns - h_events_[j].timestamp_ns) * 1e-9f;
            h_intensities_[i] += alpha * expf(-beta * time_diff);
        }
    }
    
    // Step 4: Calculate log-likelihood terms using SIMD
    simd_processor_->calculate_log_likelihood_terms(h_intensities_, h_log_likelihood_terms_, n_events);
    
    // Step 5: Calculate clustering coefficients using SIMD
    std::vector<uint64_t> timestamps(n_events);
    for (size_t i = 0; i < n_events; i++) {
        timestamps[i] = h_events_[i].timestamp_ns;
    }
    simd_processor_->calculate_clustering_coefficients(timestamps.data(), h_clustering_coeffs_, 
                                                       n_events, config_.clustering_window_ns);
    
    // Step 6: Parameter estimation using gradient descent (if enabled)
    if (config_.estimate_parameters) {
        estimate_parameters_cpu(n_events, mu, alpha, beta);
    }
    
    // Finalize results
    results = finalize_results_with_simd(n_events, mu, alpha, beta);
    results.processing_method = "SIMD";
    
    return results;
}

// Finalize results using SIMD calculations
HawkesResults HawkesEngine::finalize_results_with_simd(size_t n_events, float mu, float alpha, float beta) {
    HawkesResults results;
    
    // Set parameters
    results.parameters.mu = mu;
    results.parameters.alpha = alpha;
    results.parameters.beta = beta;
    results.parameters.branching_ratio = alpha / beta;
    results.parameters.n_events = n_events;
    
    // Calculate log-likelihood
    float log_likelihood = 0.0f;
    for (size_t i = 0; i < n_events; i++) {
        log_likelihood += h_log_likelihood_terms_[i];
    }
    results.parameters.log_likelihood = log_likelihood;
    
    // Calculate AIC and BIC
    int k = 3;  // Number of parameters (mu, alpha, beta)
    results.parameters.aic = 2 * k - 2 * log_likelihood;
    results.parameters.bic = k * logf(n_events) - 2 * log_likelihood;
    
    // Calculate statistics using SIMD
    results.statistics.mean_intensity = 0.0f;
    results.statistics.max_intensity = h_intensities_[0];
    for (size_t i = 0; i < n_events; i++) {
        results.statistics.mean_intensity += h_intensities_[i];
        results.statistics.max_intensity = fmaxf(results.statistics.max_intensity, h_intensities_[i]);
    }
    results.statistics.mean_intensity /= n_events;
    
    // Calculate intensity variance using SIMD
    float variance = 0.0f;
    for (size_t i = 0; i < n_events; i++) {
        float diff = h_intensities_[i] - results.statistics.mean_intensity;
        variance += diff * diff;
    }
    results.statistics.intensity_variance = variance / (n_events - 1);
    
    // Calculate clustering coefficient (average)
    results.statistics.clustering_coefficient = 0.0f;
    for (size_t i = 0; i < n_events; i++) {
        results.statistics.clustering_coefficient += h_clustering_coeffs_[i];
    }
    results.statistics.clustering_coefficient /= n_events;
    
    // Calculate burstiness using SIMD
    std::vector<uint64_t> timestamps(n_events);
    for (size_t i = 0; i < n_events; i++) {
        timestamps[i] = h_events_[i].timestamp_ns;
    }
    results.statistics.burstiness = simd_processor_->calculate_burstiness(timestamps.data(), n_events);
    
    // Calculate memory coefficient using SIMD
    results.statistics.memory_coefficient = simd_processor_->calculate_memory_coefficient(
        h_intensities_, n_events, config_.max_memory_lag);
    
    // Calculate criticality index
    results.statistics.criticality_index = (results.parameters.branching_ratio < 1.0f) ? 
        1.0f - results.parameters.branching_ratio : 0.0f;
    
    // Goodness-of-fit (simplified KS test)
    results.statistics.goodness_of_fit = calculate_goodness_of_fit(n_events);
    
    // Copy intensities and clustering coefficients to results
    results.intensities.assign(h_intensities_, h_intensities_ + n_events);
    results.clustering_coefficients.assign(h_clustering_coeffs_, h_clustering_coeffs_ + n_events);
    results.residuals.assign(h_residuals_, h_residuals_ + n_events);
    
    return results;
}

// CPU-based parameter estimation
void HawkesEngine::estimate_parameters_cpu(size_t n_events, float& mu, float& alpha, float& beta) {
    for (int iter = 0; iter < config_.max_iterations; iter++) {
        // Calculate gradients
        float grad_mu = 0.0f;
        float grad_alpha = 0.0f;
        float grad_beta = 0.0f;
        
        for (size_t i = 0; i < n_events; i++) {
            float intensity = mu;
            float sum_decay = 0.0f;
            float sum_weighted_decay = 0.0f;
            
            for (size_t j = 0; j < i; j++) {
                float time_diff = (h_events_[i].timestamp_ns - h_events_[j].timestamp_ns) * 1e-9f;
                float decay = expf(-beta * time_diff);
                sum_decay += decay;
                sum_weighted_decay += time_diff * decay;
            }
            
            intensity += alpha * sum_decay;
            
            // Gradient contributions
            grad_mu += 1.0f / fmaxf(intensity, 1e-10f);
            grad_alpha += sum_decay / fmaxf(intensity, 1e-10f);
            grad_beta += alpha * sum_weighted_decay / fmaxf(intensity, 1e-10f);
        }
        
        // Subtract integral terms (simplified)
        float T_total = (h_events_[n_events-1].timestamp_ns - h_events_[0].timestamp_ns) * 1e-9f;
        grad_mu -= T_total;
        grad_alpha -= n_events / fmaxf(beta, 1e-10f);
        
        // Update parameters with learning rate decay
        float lr = config_.learning_rate / (1.0f + 0.01f * iter);
        mu = fmaxf(mu + lr * grad_mu, 1e-6f);
        alpha = fmaxf(alpha + lr * grad_alpha, 1e-6f);
        beta = fmaxf(beta + lr * grad_beta, 1e-6f);
        
        // Check convergence
        float grad_norm = sqrtf(grad_mu*grad_mu + grad_alpha*grad_alpha + grad_beta*grad_beta);
        if (grad_norm < config_.convergence_threshold) {
            break;
        }
    }
}

// Simplified goodness-of-fit test
float HawkesEngine::calculate_goodness_of_fit(size_t n_events) {
    // Simplified version - in practice would use proper KS test
    if (n_events < 10) return 1.0f;
    
    // Calculate empirical distribution of residuals
    std::vector<float> sorted_residuals(h_residuals_, h_residuals_ + n_events);
    std::sort(sorted_residuals.begin(), sorted_residuals.end());
    
    // Compare with exponential distribution (simplified)
    float max_diff = 0.0f;
    for (size_t i = 0; i < n_events; i++) {
        float empirical_cdf = (i + 1.0f) / n_events;
        float theoretical_cdf = 1.0f - expf(-sorted_residuals[i]);
        max_diff = fmaxf(max_diff, fabsf(empirical_cdf - theoretical_cdf));
    }
    
    // Convert to p-value (very simplified)
    return expf(-2.0f * n_events * max_diff * max_diff);
}

// Simulate Hawkes process
std::vector<HawkesEvent> HawkesEngine::simulate_hawkes_process(
    const HawkesParameters& params,
    uint64_t start_time_ns,
    uint64_t end_time_ns,
    int max_events
) {
    std::vector<HawkesEvent> simulated_events;
    
    if (gpu_initialized_ && config_.use_gpu) {
        // GPU simulation
        int n_simulated;
        launch_simulation_kernel(d_simulated_events_, d_n_simulated_, 
                                 params.mu, params.alpha, params.beta,
                                 start_time_ns, end_time_ns, d_random_states_, max_events);
        cudaDeviceSynchronize();
        
        cudaMemcpy(&n_simulated, d_n_simulated_, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (n_simulated > 0) {
            std::vector<HawkesEvent> temp_events(n_simulated);
            cudaMemcpy(temp_events.data(), d_simulated_events_, 
                       n_simulated * sizeof(HawkesEvent), cudaMemcpyDeviceToHost);
            simulated_events = std::move(temp_events);
        }
    } else {
        // CPU simulation using Ogata's thinning algorithm
        simulated_events = simulate_hawkes_cpu(params, start_time_ns, end_time_ns, max_events);
    }
    
    return simulated_events;
}

// CPU-based Hawkes simulation
std::vector<HawkesEvent> HawkesEngine::simulate_hawkes_cpu(
    const HawkesParameters& params,
    uint64_t start_time_ns,
    uint64_t end_time_ns,
    int max_events
) {
    std::vector<HawkesEvent> events;
    
    uint64_t current_time = start_time_ns;
    float current_intensity = params.mu;
    
    // Simple random number generator
    static unsigned int seed = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    
    while (current_time < end_time_ns && events.size() < max_events) {
        // Generate inter-arrival time using thinning
        float lambda_max = current_intensity + params.alpha;  // Upper bound
        
        // Generate candidate time
        seed = seed * 1103515245 + 12345;  // Simple LCG
        float u1 = (seed & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        float inter_arrival = -logf(u1) / lambda_max;
        
        uint64_t candidate_time = current_time + (uint64_t)(inter_arrival * 1e9f);
        
        if (candidate_time >= end_time_ns) break;
        
        // Calculate actual intensity at candidate time
        float actual_intensity = params.mu;
        for (const auto& event : events) {
            float time_diff = (candidate_time - event.timestamp_ns) * 1e-9f;
            actual_intensity += params.alpha * expf(-params.beta * time_diff);
        }
        
        // Accept/reject
        seed = seed * 1103515245 + 12345;
        float u2 = (seed & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        
        if (u2 * lambda_max <= actual_intensity) {
            // Accept event
            HawkesEvent event;
            event.timestamp_ns = candidate_time;
            event.mark = 0;  // Default mark
            event.intensity = actual_intensity;
            event.price = 0.0f;  // Not used in basic simulation
            event.size = 0.0f;   // Not used in basic simulation
            event.exchange_id = 0;
            strcpy(event.ticker, "SIM");
            
            events.push_back(event);
            current_intensity = actual_intensity;
        }
        
        current_time = candidate_time;
    }
    
    return events;
}

// Cleanup functions
void HawkesEngine::cleanup_gpu_memory() {
    if (gpu_initialized_) {
        cudaFree(d_events_);
        cudaFree(d_decay_matrix_);
        cudaFree(d_intensities_);
        cudaFree(d_log_likelihood_terms_);
        cudaFree(d_mu_);
        cudaFree(d_alpha_);
        cudaFree(d_beta_);
        cudaFree(d_gradients_);
        cudaFree(d_clustering_coeffs_);
        cudaFree(d_residuals_);
        cudaFree(d_simulated_events_);
        cudaFree(d_n_simulated_);
        cudaFree(d_random_states_);
    }
}

void HawkesEngine::cleanup_host_memory() {
    if (h_events_) free(h_events_);
    if (h_intensities_) free(h_intensities_);
    if (h_decay_values_) free(h_decay_values_);
    if (h_log_likelihood_terms_) free(h_log_likelihood_terms_);
    if (h_clustering_coeffs_) free(h_clustering_coeffs_);
    if (h_residuals_) free(h_residuals_);
    if (h_time_diffs_) free(h_time_diffs_);
}

// Get engine status
HawkesEngineStatus HawkesEngine::get_status() const {
    HawkesEngineStatus status;
    status.gpu_available = gpu_initialized_;
    status.simd_level = simd_processor_->get_simd_level_name();
    status.max_events = config_.max_events;
    status.gpu_device_id = config_.gpu_device_id;
    
    if (gpu_initialized_) {
        cudaMemGetInfo(&status.gpu_memory_free, &status.gpu_memory_total);
    } else {
        status.gpu_memory_free = 0;
        status.gpu_memory_total = 0;
    }
    
    return status;
}

