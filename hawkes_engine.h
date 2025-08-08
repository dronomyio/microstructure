#ifndef HAWKES_ENGINE_H
#define HAWKES_ENGINE_H

#include "hawkes_gpu_kernels.h"
#include "hawkes_simd_processors.h"
#include <vector>
#include <memory>
#include <string>

// Forward declarations
class HawkesSIMDProcessor;

// Configuration structure for Hawkes engine
struct HawkesEngineConfig {
    // GPU settings
    bool use_gpu = true;
    int gpu_device_id = 0;
    size_t gpu_threshold = 1000;  // Minimum events to use GPU
    
    // Memory settings
    size_t max_events = 1000000;
    
    // Parameter estimation settings
    bool estimate_parameters = true;
    float learning_rate = 0.01f;
    int max_iterations = 1000;
    float convergence_threshold = 1e-6f;
    
    // Analysis settings
    uint64_t clustering_window_ns = 1000000000ULL;  // 1 second
    int max_memory_lag = 50;
    
    // Performance settings
    bool enable_profiling = false;
    bool verbose_output = false;
};

// Results structure
struct HawkesResults {
    // Basic info
    size_t n_events = 0;
    std::string processing_method;
    double processing_time_ms = 0.0;
    double throughput_events_per_sec = 0.0;
    
    // Estimated parameters
    HawkesParameters parameters;
    
    // Statistical measures
    HawkesStatistics statistics;
    
    // Detailed results
    std::vector<float> intensities;
    std::vector<float> clustering_coefficients;
    std::vector<float> residuals;
    
    // Model validation
    bool is_subcritical = true;
    float model_fit_quality = 0.0f;
    std::string validation_notes;
};

// Engine status
struct HawkesEngineStatus {
    bool gpu_available = false;
    std::string simd_level;
    size_t max_events = 0;
    int gpu_device_id = -1;
    size_t gpu_memory_free = 0;
    size_t gpu_memory_total = 0;
};

/**
 * Main Hawkes Process Engine
 * Orchestrates GPU and SIMD processing for optimal performance
 */
class HawkesEngine {
public:
    // Constructor and destructor
    explicit HawkesEngine(const HawkesEngineConfig& config = HawkesEngineConfig{});
    ~HawkesEngine();
    
    // Disable copy constructor and assignment
    HawkesEngine(const HawkesEngine&) = delete;
    HawkesEngine& operator=(const HawkesEngine&) = delete;
    
    // Main analysis functions
    HawkesResults analyze_hawkes_process(
        const std::vector<HawkesEvent>& events,
        const HawkesParameters& initial_params = {0.1f, 0.5f, 1.0f}
    );
    
    // Simulation functions
    std::vector<HawkesEvent> simulate_hawkes_process(
        const HawkesParameters& params,
        uint64_t start_time_ns,
        uint64_t end_time_ns,
        int max_events = 10000
    );
    
    // Batch processing for large datasets
    std::vector<HawkesResults> analyze_batch(
        const std::vector<std::vector<HawkesEvent>>& event_batches,
        const std::vector<HawkesParameters>& initial_params_list
    );
    
    // Real-time processing
    HawkesResults analyze_streaming_events(
        const std::vector<HawkesEvent>& new_events,
        const HawkesResults& previous_results
    );
    
    // Parameter estimation
    HawkesParameters estimate_parameters(
        const std::vector<HawkesEvent>& events,
        const HawkesParameters& initial_guess = {0.1f, 0.5f, 1.0f}
    );
    
    // Model validation
    float calculate_goodness_of_fit(
        const std::vector<HawkesEvent>& events,
        const HawkesParameters& params
    );
    
    bool validate_model(
        const std::vector<HawkesEvent>& events,
        const HawkesParameters& params,
        std::string& validation_report
    );
    
    // Advanced analysis functions
    std::vector<float> calculate_intensity_timeline(
        const std::vector<HawkesEvent>& events,
        const HawkesParameters& params,
        uint64_t start_time_ns,
        uint64_t end_time_ns,
        uint64_t time_step_ns = 1000000ULL  // 1ms
    );
    
    std::vector<int> detect_regime_changes(
        const std::vector<HawkesEvent>& events,
        int window_size = 100,
        float threshold = 2.0f
    );
    
    std::vector<float> calculate_cross_correlation(
        const std::vector<HawkesEvent>& events1,
        const std::vector<HawkesEvent>& events2,
        const HawkesParameters& params1,
        const HawkesParameters& params2,
        int max_lag = 50
    );
    
    // Multivariate Hawkes processes
    struct MultivariateHawkesParameters {
        std::vector<float> mu;              // Base intensities (n_marks)
        std::vector<std::vector<float>> alpha;  // Interaction matrix (n_marks x n_marks)
        std::vector<std::vector<float>> beta;   // Decay matrix (n_marks x n_marks)
    };
    
    struct MultivariateHawkesResults {
        MultivariateHawkesParameters parameters;
        std::vector<std::vector<float>> intensities;  // n_marks x n_events
        std::vector<std::vector<float>> causality_matrix;  // Granger causality
        float log_likelihood;
        std::string processing_method;
        double processing_time_ms;
    };
    
    MultivariateHawkesResults analyze_multivariate_hawkes(
        const std::vector<HawkesEvent>& events,
        int n_marks,
        const MultivariateHawkesParameters& initial_params
    );
    
    // Financial microstructure specific functions
    struct FinancialHawkesResults {
        HawkesParameters quote_params;
        HawkesParameters trade_params;
        float quote_trade_correlation;
        std::vector<float> price_impact_measures;
        std::vector<float> liquidity_measures;
        std::vector<float> toxicity_measures;
        std::string processing_method;
        double processing_time_ms;
    };
    
    FinancialHawkesResults analyze_financial_hawkes(
        const std::vector<HawkesEvent>& quote_events,
        const std::vector<HawkesEvent>& trade_events
    );
    
    // Utility functions
    HawkesEngineStatus get_status() const;
    void set_config(const HawkesEngineConfig& config);
    HawkesEngineConfig get_config() const { return config_; }
    
    // Performance benchmarking
    struct BenchmarkResults {
        double gpu_processing_time_ms;
        double simd_processing_time_ms;
        double gpu_throughput_events_per_sec;
        double simd_throughput_events_per_sec;
        double speedup_factor;
        std::string recommended_method;
    };
    
    BenchmarkResults benchmark_performance(
        size_t n_events,
        int n_iterations = 10
    );
    
private:
    // Configuration
    HawkesEngineConfig config_;
    
    // GPU resources
    bool gpu_initialized_;
    HawkesEvent* d_events_;
    float* d_decay_matrix_;
    float* d_intensities_;
    float* d_log_likelihood_terms_;
    float* d_mu_;
    float* d_alpha_;
    float* d_beta_;
    float* d_gradients_;
    float* d_clustering_coeffs_;
    float* d_residuals_;
    HawkesEvent* d_simulated_events_;
    int* d_n_simulated_;
    unsigned int* d_random_states_;
    
    // Host memory (aligned for SIMD)
    HawkesEvent* h_events_;
    float* h_intensities_;
    float* h_decay_values_;
    float* h_log_likelihood_terms_;
    float* h_clustering_coeffs_;
    float* h_residuals_;
    float* h_time_diffs_;
    
    // SIMD processor
    std::unique_ptr<HawkesSIMDProcessor> simd_processor_;
    
    // Initialization functions
    bool initialize_gpu();
    void allocate_host_memory();
    
    // Processing pipelines
    HawkesResults process_with_gpu(size_t n_events, const HawkesParameters& initial_params);
    HawkesResults process_with_simd(size_t n_events, const HawkesParameters& initial_params);
    HawkesResults finalize_results_with_simd(size_t n_events, float mu, float alpha, float beta);
    
    // Parameter estimation
    void estimate_parameters_cpu(size_t n_events, float& mu, float& alpha, float& beta);
    
    // Simulation
    std::vector<HawkesEvent> simulate_hawkes_cpu(
        const HawkesParameters& params,
        uint64_t start_time_ns,
        uint64_t end_time_ns,
        int max_events
    );
    
    // Utility functions
    float calculate_goodness_of_fit(size_t n_events);
    
    // Cleanup functions
    void cleanup_gpu_memory();
    void cleanup_host_memory();
};

/**
 * Factory function for creating optimized Hawkes engines
 */
std::unique_ptr<HawkesEngine> create_hawkes_engine(
    const HawkesEngineConfig& config = HawkesEngineConfig{}
);

/**
 * Utility functions for Hawkes process analysis
 */
namespace HawkesUtils {
    
    // Data conversion utilities
    std::vector<HawkesEvent> convert_from_timestamps(
        const std::vector<uint64_t>& timestamps_ns,
        const std::vector<int>& marks = {},
        const std::string& ticker = "DEFAULT"
    );
    
    std::vector<HawkesEvent> convert_from_financial_data(
        const std::vector<uint64_t>& timestamps_ns,
        const std::vector<float>& prices,
        const std::vector<float>& sizes,
        const std::vector<int>& exchange_ids,
        const std::string& ticker
    );
    
    // Statistical utilities
    float calculate_branching_ratio(const HawkesParameters& params);
    bool is_subcritical(const HawkesParameters& params);
    float calculate_expected_intensity(const HawkesParameters& params);
    
    // Validation utilities
    bool validate_events(const std::vector<HawkesEvent>& events, std::string& error_message);
    bool validate_parameters(const HawkesParameters& params, std::string& error_message);
    
    // Visualization data preparation
    struct PlotData {
        std::vector<double> times;
        std::vector<double> values;
        std::string label;
        std::string color;
    };
    
    std::vector<PlotData> prepare_intensity_plot(
        const std::vector<HawkesEvent>& events,
        const std::vector<float>& intensities
    );
    
    std::vector<PlotData> prepare_clustering_plot(
        const std::vector<HawkesEvent>& events,
        const std::vector<float>& clustering_coeffs
    );
    
    PlotData prepare_parameter_evolution_plot(
        const std::vector<HawkesParameters>& parameter_history
    );
}

#endif // HAWKES_ENGINE_H

