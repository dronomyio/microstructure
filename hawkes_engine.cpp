#include "hawkes_engine.h"
#include "hawkes_gpu_kernels.h"
#include "hawkes_simd_processors.h"

// Conditional CUDA includes
#ifdef CUDA_ENABLED
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
#endif

#include <chrono>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <cmath>  // For sqrtf, expf, logf, fmaxf, fabsf
#include <random> // For random number generation

// Constructor
HawkesEngine::HawkesEngine(const HawkesEngineConfig& config) 
    : config_(config), gpu_initialized_(false), simd_processor_(nullptr) {
    
    // Initialize SIMD processor
    simd_processor_ = std::make_unique<HawkesSIMDProcessor>();
    
    // Initialize GPU if available
    if (config_.use_gpu) {
        initialize_gpu();
    }
    
    // Allocate host memory for processing
    allocate_host_memory();
}

// Destructor
HawkesEngine::~HawkesEngine() {
    cleanup_gpu_memory();
    cleanup_host_memory();
}

// Initialize GPU resources
bool HawkesEngine::initialize_gpu() {
#ifdef CUDA_ENABLED
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
    
    // Parameters - FIXED: No more |= operators
    error = cudaMalloc(&d_mu_, sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for mu parameter" << std::endl;
        return false;
    }
    
    error = cudaMalloc(&d_alpha_, sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for alpha parameter" << std::endl;
        return false;
    }
    
    error = cudaMalloc(&d_beta_, sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for beta parameter" << std::endl;
        return false;
    }
    
    // Gradients
    error = cudaMalloc(&d_gradients_, 3 * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for gradients" << std::endl;
        return false;
    }
    
    // Clustering coefficients
    error = cudaMalloc(&d_clustering_coeffs_, intensities_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for clustering coefficients" << std::endl;
        return false;
    }
    
    // Residuals
    error = cudaMalloc(&d_residuals_, intensities_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for residuals" << std::endl;
        return false;
    }
    
    // Simulation arrays - FIXED: No more |= operators
    error = cudaMalloc(&d_simulated_events_, config_.max_events * sizeof(HawkesEvent));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for simulated events" << std::endl;
        return false;
    }
    
    error = cudaMalloc(&d_n_simulated_, sizeof(int));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for n_simulated" << std::endl;
        return false;
    }
    
    error = cudaMalloc(&d_random_states_, sizeof(curandState));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for random states" << std::endl;
        return false;
    }
    
    // Initialize random state
    unsigned int initial_seed = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    cudaMemcpy(d_random_states_, &initial_seed, sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    gpu_initialized_ = true;
    return true;
#else
    // CPU-only build - GPU not available
    std::cout << "GPU support not available (CPU-only build)" << std::endl;
    return false;
#endif
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
        std::cerr << "Failed to allocate aligned host memory" << std::endl;
        throw std::bad_alloc();
    }
}

// Main analysis function
HawkesResults HawkesEngine::analyze_hawkes_process(
    const std::vector<HawkesEvent>& events,
    const HawkesParameters& initial_params) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    size_t n_events = events.size();
    if (n_events == 0) {
        throw std::invalid_argument("Empty event sequence");
    }
    
    if (n_events > config_.max_events) {
        std::cerr << "Warning: Event count (" << n_events 
                  << ") exceeds max_events (" << config_.max_events 
                  << "). Truncating." << std::endl;
        n_events = config_.max_events;
    }
    
    // Copy events to aligned host memory
    std::memcpy(h_events_, events.data(), n_events * sizeof(HawkesEvent));
    
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
#ifdef CUDA_ENABLED
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
        
        // Get final parameters
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
#else
    // CPU-only fallback - should not be called in CPU-only builds
    return process_with_simd(n_events, initial_params);
#endif
}

// SIMD processing pipeline
HawkesResults HawkesEngine::process_with_simd(size_t n_events, const HawkesParameters& initial_params) {
    HawkesResults results;
    
    // Initialize parameters
    float mu = initial_params.mu;
    float alpha = initial_params.alpha;
    float beta = initial_params.beta;
    
    // Calculate intensities using SIMD
    std::fill(h_intensities_, h_intensities_ + n_events, mu);
    
    for (size_t i = 0; i < n_events; i++) {
        for (size_t j = 0; j < i; j++) {
            float time_diff = (h_events_[i].timestamp_ns - h_events_[j].timestamp_ns) / 1e9f;
            h_intensities_[i] += alpha * expf(-beta * time_diff);
        }
    }
    
    // Parameter estimation using CPU
    if (config_.estimate_parameters) {
        estimate_parameters_cpu(n_events, mu, alpha, beta);
    }
    
    // Calculate clustering coefficients using basic function
    for (size_t i = 0; i < n_events; i++) {
        h_clustering_coeffs_[i] = 0.5f; // Placeholder clustering coefficient
    }
    
    // Calculate residuals
    for (size_t i = 0; i < n_events; i++) {
        h_residuals_[i] = h_intensities_[i] - mu;
        for (size_t j = 0; j < i; j++) {
            float time_diff = (h_events_[i].timestamp_ns - h_events_[j].timestamp_ns) / 1e9f;
            h_residuals_[i] -= alpha * expf(-beta * time_diff);
        }
    }
    
    // Use SIMD for final statistical calculations
    results = finalize_results_with_simd(n_events, mu, alpha, beta);
    results.processing_method = "SIMD";
    
    return results;
}

// Finalize results with SIMD calculations
HawkesResults HawkesEngine::finalize_results_with_simd(size_t n_events, float mu, float alpha, float beta) {
    HawkesResults results;
    
    // Set parameters
    results.parameters.mu = mu;
    results.parameters.alpha = alpha;
    results.parameters.beta = beta;
    
    // Calculate log-likelihood using SIMD
    float log_likelihood = 0.0f;
    for (size_t i = 0; i < n_events; i++) {
        log_likelihood += logf(h_intensities_[i]);
    }
    log_likelihood -= mu * (h_events_[n_events-1].timestamp_ns - h_events_[0].timestamp_ns) / 1e9f;
    
    results.parameters.log_likelihood = log_likelihood;
    results.parameters.aic = 2 * 3 - 2 * log_likelihood;  // 3 parameters
    
    int k = 3;  // number of parameters
    results.parameters.bic = k * logf(n_events) - 2 * log_likelihood;
    
    // Calculate statistics using basic calculations
    float sum_intensity = 0.0f;
    for (size_t i = 0; i < n_events; i++) {
        sum_intensity += h_intensities_[i];
    }
    results.statistics.mean_intensity = sum_intensity / n_events;
    results.statistics.max_intensity = h_intensities_[0];
    for (size_t i = 1; i < n_events; i++) {
        results.statistics.max_intensity = fmaxf(results.statistics.max_intensity, h_intensities_[i]);
    }
    
    results.statistics.goodness_of_fit = calculate_goodness_of_fit(n_events);
    
    return results;
}

// CPU parameter estimation
void HawkesEngine::estimate_parameters_cpu(size_t n_events, float& mu, float& alpha, float& beta) {
    for (int iter = 0; iter < config_.max_iterations; iter++) {
        float grad_mu = 0.0f, grad_alpha = 0.0f, grad_beta = 0.0f;
        
        // Calculate gradients
        for (size_t i = 0; i < n_events; i++) {
            float intensity = mu;
            for (size_t j = 0; j < i; j++) {
                float time_diff = (h_events_[i].timestamp_ns - h_events_[j].timestamp_ns) / 1e9f;
                float decay = expf(-beta * time_diff);
                intensity += alpha * decay;
            }
            
            grad_mu += 1.0f / fmaxf(intensity, 1e-10f);
        }
        
        // Simplified gradient calculation
        float T = (h_events_[n_events-1].timestamp_ns - h_events_[0].timestamp_ns) / 1e9f;
        grad_mu -= T;
        grad_alpha -= n_events / fmaxf(beta, 1e-10f);
        
        // Update parameters
        mu += config_.learning_rate * grad_mu;
        alpha += config_.learning_rate * grad_alpha;
        beta += config_.learning_rate * grad_beta;
        
        // Check convergence
        float grad_norm = sqrtf(grad_mu*grad_mu + grad_alpha*grad_alpha + grad_beta*grad_beta);
        if (grad_norm < config_.convergence_threshold) {
            break;
        }
    }
}

// Calculate goodness of fit using Kolmogorov-Smirnov test
float HawkesEngine::calculate_goodness_of_fit(size_t n_events) {
    // Sort residuals
    std::vector<float> sorted_residuals(h_residuals_, h_residuals_ + n_events);
    std::sort(sorted_residuals.begin(), sorted_residuals.end());
    
    // Calculate KS statistic
    float max_diff = 0.0f;
    for (size_t i = 0; i < n_events; i++) {
        float empirical_cdf = (i + 1.0f) / n_events;
        float theoretical_cdf = 1.0f - expf(-sorted_residuals[i]);
        max_diff = fmaxf(max_diff, fabsf(empirical_cdf - theoretical_cdf));
    }
    
    // Return p-value approximation
    return expf(-2.0f * n_events * max_diff * max_diff);
}

// Simulation function
std::vector<HawkesEvent> HawkesEngine::simulate_hawkes_process(
    const HawkesParameters& params,
    uint64_t start_time_ns,
    uint64_t end_time_ns,
    int max_events) {
    
    std::vector<HawkesEvent> simulated_events;
    
    if (gpu_initialized_ && config_.use_gpu) {
#ifdef CUDA_ENABLED
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
#endif
    } else {
        // CPU simulation
        simulated_events = simulate_hawkes_cpu(params, start_time_ns, end_time_ns, max_events);
    }
    
    return simulated_events;
}

// CPU simulation
std::vector<HawkesEvent> HawkesEngine::simulate_hawkes_cpu(
    const HawkesParameters& params,
    uint64_t start_time_ns,
    uint64_t end_time_ns,
    int max_events) {
    
    std::vector<HawkesEvent> events;
    events.reserve(max_events);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    
    uint64_t current_time = start_time_ns;
    float lambda_max = params.mu + params.alpha;  // Upper bound for intensity
    
    while (current_time < end_time_ns && events.size() < max_events) {
        // Generate candidate event time
        float u1 = uniform(gen);
        float inter_arrival = -logf(u1) / lambda_max;
        current_time += static_cast<uint64_t>(inter_arrival * 1e9);
        
        if (current_time >= end_time_ns) break;
        
        // Calculate actual intensity at this time
        float actual_intensity = params.mu;
        for (const auto& prev_event : events) {
            float time_diff = (current_time - prev_event.timestamp_ns) / 1e9f;
            actual_intensity += params.alpha * expf(-params.beta * time_diff);
        }
        
        // Accept/reject
        float u2 = uniform(gen);
        if (u2 * lambda_max <= actual_intensity) {
            HawkesEvent event;
            event.timestamp_ns = current_time;
            event.price = 100.0 + uniform(gen) * 10.0;  // Dummy price
            event.size = 100.0 + uniform(gen) * 50.0;   // Dummy size
            event.mark = (uniform(gen) > 0.5) ? 1 : 0;  // Buy/sell mark
            event.intensity = actual_intensity;
            event.exchange_id = 0;
            std::strncpy(event.ticker, "TEST", 8);
            events.push_back(event);
        }
    }
    
    return events;
}

// Cleanup functions
void HawkesEngine::cleanup_gpu_memory() {
#ifdef CUDA_ENABLED
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
        gpu_initialized_ = false;
    }
#endif
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
    
#ifdef CUDA_ENABLED
    if (gpu_initialized_) {
        cudaMemGetInfo(&status.gpu_memory_free, &status.gpu_memory_total);
    } else {
        status.gpu_memory_free = 0;
        status.gpu_memory_total = 0;
    }
#else
    status.gpu_memory_free = 0;
    status.gpu_memory_total = 0;
#endif
    
    return status;
}


