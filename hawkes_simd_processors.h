#ifndef HAWKES_SIMD_PROCESSORS_H
#define HAWKES_SIMD_PROCESSORS_H

#include <stdint.h>
#include <cstddef>

// SIMD capability levels
enum SIMDLevel {
    SIMD_SCALAR = 0,
    SIMD_AVX256 = 1,
    SIMD_AVX512 = 2
};

// Forward declarations for SIMD functions
extern "C" {

// AVX-512 optimized functions
void calculate_exponential_decay_avx512(
    const float* time_diffs,
    float* decay_values,
    size_t n_values,
    float beta
);

/**
 * AVX-256 optimized exponential decay calculation
 * Computes exp(-beta * time_diffs) for 8 values simultaneously
 */
void calculate_exponential_decay_avx256(
    const float* time_diffs,
    float* decay_values,
    size_t n_values,
    float beta
); 

void calculate_hawkes_intensity_avx512(
    const float* decay_matrix,
    float* intensities,
    size_t n_events,
    float mu,
    float alpha
);

/**
 * AVX-256 optimized Hawkes intensity calculation
 * λ(t) = μ + Σ α * exp(-β * (t - t_i))
 */
void calculate_hawkes_intensity_avx256(
    const float* decay_matrix,
    float* intensities,
    size_t n_events,
    float mu,
    float alpha
);

void calculate_log_likelihood_terms_avx512(
    const float* intensities,
    float* log_terms,
    size_t n_events
);



float calculate_intensity_correlation_avx512(
    const float* intensities1,
    const float* intensities2,
    size_t n_events
);

void calculate_clustering_coefficients_avx512(
    const uint64_t* timestamps_ns,
    float* clustering_coeffs,
    size_t n_events,
    uint64_t time_window_ns
);

float calculate_burstiness_avx512(
    const uint64_t* timestamps_ns,
    size_t n_events
);

/**
 * AVX-256 optimized burstiness calculation
 * B = (σ - μ) / (σ + μ) where σ, μ are std dev and mean of inter-arrival times
 */
float calculate_burstiness_avx256(
    const uint64_t* timestamps_ns,
    size_t n_events
) ;

float calculate_memory_coefficient_avx512(
    const float* intensities,
    size_t n_events,
    int max_lag
);

/**
 * AVX-256 optimized memory coefficient calculation
 * Measures long-range dependence in event timing
 */
float calculate_memory_coefficient_avx256(
    const float* intensities,
    size_t n_events,
    int max_lag
);

// AVX-256 optimized functions
void calculate_exponential_decay_avx256(
    const float* time_diffs,
    float* decay_values,
    size_t n_values,
    float beta
);

float calculate_intensity_correlation_avx256(
    const float* intensities1,
    const float* intensities2,
    size_t n_events
);

// Cross-correlation and spectral analysis
void calculate_cross_correlation_avx512(
    const float* series1,
    const float* series2,
    float* cross_correlations,
    size_t n_events,
    int max_lag
);

void calculate_power_spectral_density_avx512(
    const float* intensities,
    float* psd,
    size_t n_events,
    float sampling_rate
);

// Advanced statistical measures
void calculate_hurst_exponent_avx512(
    const float* intensities,
    float* hurst_exponent,
    size_t n_events,
    int min_window,
    int max_window
);

void calculate_multifractal_spectrum_avx512(
    const float* intensities,
    float* spectrum,
    size_t n_events,
    const float* q_values,
    int n_q_values
);

// Regime detection and change point analysis
void calculate_regime_probabilities_avx512(
    const float* intensities,
    float* regime_probs,
    size_t n_events,
    int n_regimes,
    const float* regime_thresholds
);

void detect_change_points_avx512(
    const float* intensities,
    int* change_points,
    float* change_scores,
    size_t n_events,
    int window_size,
    float threshold
);

// Financial microstructure specific SIMD functions
void calculate_price_impact_correlation_avx512(
    const float* price_changes,
    const float* intensities,
    float* impact_correlations,
    size_t n_events,
    int max_lag
);

void calculate_volatility_clustering_avx512(
    const float* price_returns,
    const float* intensities,
    float* clustering_measures,
    size_t n_events,
    int window_size
);

void calculate_jump_detection_avx512(
    const float* intensities,
    const float* price_changes,
    int* jump_indicators,
    float* jump_sizes,
    size_t n_events,
    float jump_threshold
);

// Network analysis SIMD functions
void calculate_centrality_measures_avx512(
    const float* adjacency_matrix,
    float* centrality_scores,
    size_t n_nodes,
    int centrality_type  // 0=degree, 1=betweenness, 2=eigenvector
);

void calculate_network_clustering_avx512(
    const float* adjacency_matrix,
    float* clustering_coefficients,
    size_t n_nodes
);

} // extern "C"

/**
 * Main SIMD processor class for Hawkes processes
 * Automatically detects and uses the best available SIMD instruction set
 */
class HawkesSIMDProcessor {
public:
    HawkesSIMDProcessor();
    ~HawkesSIMDProcessor() = default;
    
    // Core Hawkes process calculations
    void calculate_exponential_decay(
        const float* time_diffs,
        float* decay_values,
        size_t n_values,
        float beta
    );
    
    void calculate_hawkes_intensity(
        const float* decay_matrix,
        float* intensities,
        size_t n_events,
        float mu,
        float alpha
    );
    
    void calculate_log_likelihood_terms(
        const float* intensities,
        float* log_terms,
        size_t n_events
    );
    
    // Statistical analysis functions
    float calculate_intensity_correlation(
        const float* intensities1,
        const float* intensities2,
        size_t n_events
    );
    
    void calculate_clustering_coefficients(
        const uint64_t* timestamps_ns,
        float* clustering_coeffs,
        size_t n_events,
        uint64_t time_window_ns
    );
    
    float calculate_burstiness(
        const uint64_t* timestamps_ns,
        size_t n_events
    );
    
    float calculate_memory_coefficient(
        const float* intensities,
        size_t n_events,
        int max_lag
    );
    
    // Advanced statistical measures
    void calculate_cross_correlation(
        const float* series1,
        const float* series2,
        float* cross_correlations,
        size_t n_events,
        int max_lag
    );
    
    void calculate_power_spectral_density(
        const float* intensities,
        float* psd,
        size_t n_events,
        float sampling_rate
    );
    
    float calculate_hurst_exponent(
        const float* intensities,
        size_t n_events,
        int min_window = 10,
        int max_window = 100
    );
    
    void calculate_multifractal_spectrum(
        const float* intensities,
        float* spectrum,
        size_t n_events,
        const float* q_values,
        int n_q_values
    );
    
    // Regime detection and change points
    void calculate_regime_probabilities(
        const float* intensities,
        float* regime_probs,
        size_t n_events,
        int n_regimes,
        const float* regime_thresholds
    );
    
    void detect_change_points(
        const float* intensities,
        int* change_points,
        float* change_scores,
        size_t n_events,
        int window_size = 50,
        float threshold = 2.0f
    );
    
    // Financial microstructure specific functions
    void calculate_price_impact_correlation(
        const float* price_changes,
        const float* intensities,
        float* impact_correlations,
        size_t n_events,
        int max_lag = 20
    );
    
    void calculate_volatility_clustering(
        const float* price_returns,
        const float* intensities,
        float* clustering_measures,
        size_t n_events,
        int window_size = 50
    );
    
    void calculate_jump_detection(
        const float* intensities,
        const float* price_changes,
        int* jump_indicators,
        float* jump_sizes,
        size_t n_events,
        float jump_threshold = 3.0f
    );
    
    // Network analysis functions
    void calculate_centrality_measures(
        const float* adjacency_matrix,
        float* centrality_scores,
        size_t n_nodes,
        int centrality_type = 0  // 0=degree, 1=betweenness, 2=eigenvector
    );
    
    void calculate_network_clustering(
        const float* adjacency_matrix,
        float* clustering_coefficients,
        size_t n_nodes
    );
    
    // Utility functions
    SIMDLevel get_simd_level() const { return simd_level_; }
    const char* get_simd_level_string() const;
    const char* get_simd_level_name() const;
    
    // Performance benchmarking
    struct PerformanceMetrics {
        double processing_time_ms;
        double throughput_events_per_sec;
        size_t memory_usage_bytes;
        const char* simd_level_used;
    };
    
    PerformanceMetrics benchmark_performance(
        size_t n_events,
        int n_iterations = 100
    );
    
private:
    SIMDLevel simd_level_;
    
    // Internal helper functions
    void detect_simd_capabilities();
    bool validate_alignment(const void* ptr, size_t alignment) const;
    void ensure_memory_alignment(float** ptr, size_t size, size_t alignment) const;
};

/**
 * Specialized SIMD processors for different Hawkes process variants
 */

// Multivariate Hawkes SIMD processor
class MultivariateHawkesSIMDProcessor : public HawkesSIMDProcessor {
public:
    void calculate_multivariate_intensity(
        const float* decay_matrices,      // n_marks x n_marks x n_events x n_events
        float* intensities,               // n_marks x n_events
        size_t n_events,
        size_t n_marks,
        const float* mu_vector,           // n_marks
        const float* alpha_matrix         // n_marks x n_marks
    );
    
    void calculate_granger_causality(
        const float* intensities,         // n_marks x n_events
        float* causality_matrix,          // n_marks x n_marks
        size_t n_events,
        size_t n_marks,
        int max_lag = 10
    );
    
    void calculate_cross_excitation_matrix(
        const float* alpha_matrix,        // n_marks x n_marks
        const float* beta_matrix,         // n_marks x n_marks
        float* excitation_matrix,         // n_marks x n_marks
        size_t n_marks
    );
};

// Financial Hawkes SIMD processor
class FinancialHawkesSIMDProcessor : public HawkesSIMDProcessor {
public:
    void calculate_quote_trade_intensity(
        const uint64_t* quote_timestamps,
        const uint64_t* trade_timestamps,
        float* quote_intensities,
        float* trade_intensities,
        size_t n_quotes,
        size_t n_trades,
        const float* parameters  // [mu_q, mu_t, alpha_qq, alpha_qt, alpha_tq, alpha_tt, beta]
    );
    
    void calculate_order_flow_toxicity(
        const float* trade_intensities,
        const float* price_impacts,
        float* toxicity_measures,
        size_t n_events,
        float decay_factor = 0.95f
    );
    
    void calculate_market_impact_kernel(
        const float* trade_sizes,
        const float* intensities,
        float* impact_functions,
        size_t n_events,
        float impact_decay = 0.1f
    );
    
    void calculate_liquidity_dynamics(
        const float* bid_intensities,
        const float* ask_intensities,
        const float* spread_changes,
        float* liquidity_measures,
        size_t n_events
    );
};

// High-frequency Hawkes SIMD processor
class HighFrequencyHawkesSIMDProcessor : public HawkesSIMDProcessor {
public:
    void calculate_microstructure_noise(
        const float* price_changes,
        const float* intensities,
        float* noise_measures,
        size_t n_events,
        int window_size = 20
    );
    
    void calculate_bid_ask_dynamics(
        const float* bid_intensities,
        const float* ask_intensities,
        float* spread_dynamics,
        float* depth_dynamics,
        size_t n_events
    );
    
    void calculate_fragmentation_effects(
        const float* exchange_intensities,  // n_exchanges x n_events
        float* fragmentation_measures,
        size_t n_events,
        size_t n_exchanges
    );
    
    void calculate_latency_arbitrage(
        const float* fast_intensities,
        const float* slow_intensities,
        float* arbitrage_opportunities,
        size_t n_events,
        float latency_threshold_ns = 1000.0f  // 1 microsecond
    );
};

#endif // HAWKES_SIMD_PROCESSORS_H

