#ifndef HAWKES_GPU_KERNELS_H
#define HAWKES_GPU_KERNELS_H

#include <stdint.h>

// Data structure for Hawkes process events
struct alignas(16) HawkesEvent {
    uint64_t timestamp_ns;      // Event timestamp in nanoseconds
    int mark;                   // Event type/mark (0=quote update, 1=trade, etc.)
    float intensity;            // Intensity at event time
    float price;                // Price at event (for financial applications)
    float size;                 // Size/volume at event
    int exchange_id;            // Exchange identifier
    char ticker[8];             // Stock ticker symbol
    float reserved;             // Padding for alignment
};

// Hawkes process parameters structure
struct alignas(16) HawkesParameters {
    float mu;                   // Base intensity (background rate)
    float alpha;                // Jump size (self-excitation strength)
    float beta;                 // Decay rate (memory parameter)
    float branching_ratio;      // α/β (criticality measure)
    float log_likelihood;       // Model log-likelihood
    float aic;                  // Akaike Information Criterion
    float bic;                  // Bayesian Information Criterion
    int n_events;               // Number of events used for estimation
};

// Hawkes process statistics structure
struct alignas(16) HawkesStatistics {
    float mean_intensity;       // Average intensity
    float max_intensity;        // Maximum intensity
    float intensity_variance;   // Intensity variance
    float clustering_coefficient; // Event clustering measure
    float burstiness;           // Burstiness index
    float memory_coefficient;   // Memory/persistence measure
    float criticality_index;    // How close to critical regime
    float goodness_of_fit;      // Model fit quality (KS test p-value)
};

// GPU kernel launch functions
#ifdef __cplusplus
extern "C" {
#endif

// Core Hawkes computation kernels
void launch_exponential_decay_kernel(
    const HawkesEvent* d_events,
    float* d_decay_matrix,
    int n_events,
    float beta
);

void launch_hawkes_intensity_kernel(
    const HawkesEvent* d_events,
    const float* d_decay_matrix,
    float* d_intensities,
    int n_events,
    float mu,
    float alpha,
    int mark_type
);

void launch_log_likelihood_kernel(
    const HawkesEvent* d_events,
    const float* d_intensities,
    float* d_log_likelihood_terms,
    int n_events,
    float mu,
    float alpha,
    float beta,
    float T_end
);

// Parameter estimation kernels
void launch_parameter_estimation_kernel(
    const HawkesEvent* d_events,
    float* d_mu,
    float* d_alpha,
    float* d_beta,
    float* d_gradients,
    int n_events,
    float learning_rate,
    int iteration
);

void launch_branching_ratio_kernel(
    float* d_branching_ratios,
    float* d_criticality_indicators,
    const float* d_alphas,
    const float* d_betas,
    int n_processes
);

// Statistical analysis kernels
void launch_clustering_coefficient_kernel(
    const HawkesEvent* d_events,
    float* d_clustering_coefficients,
    int n_events,
    float time_window_ns
);

void launch_burstiness_kernel(
    const HawkesEvent* d_events,
    float* d_burstiness_measures,
    int n_events,
    float time_window_ns
);

void launch_memory_coefficient_kernel(
    const HawkesEvent* d_events,
    float* d_memory_coefficients,
    int n_events,
    int max_lag
);

// Simulation and validation kernels
void launch_simulation_kernel(
    HawkesEvent* d_simulated_events,
    int* d_n_simulated,
    float mu,
    float alpha,
    float beta,
    uint64_t start_time_ns,
    uint64_t end_time_ns,
    unsigned int* d_random_states,
    int max_events
);

void launch_residuals_kernel(
    const HawkesEvent* d_events,
    const float* d_intensities,
    float* d_residuals,
    int n_events,
    float mu,
    float alpha,
    float beta
);

void launch_goodness_of_fit_kernel(
    const float* d_residuals,
    float* d_ks_statistic,
    float* d_p_value,
    int n_events
);

// Multi-dimensional Hawkes processes
void launch_multivariate_intensity_kernel(
    const HawkesEvent* d_events,
    const float* d_alpha_matrix,  // n_marks x n_marks interaction matrix
    const float* d_beta_matrix,   // n_marks x n_marks decay matrix
    float* d_intensities,
    int n_events,
    int n_marks,
    const float* d_mu_vector
);

void launch_granger_causality_kernel(
    const HawkesEvent* d_events,
    const float* d_alpha_matrix,
    float* d_causality_measures,
    int n_events,
    int n_marks
);

// Financial microstructure specific kernels
void launch_quote_intensity_kernel(
    const HawkesEvent* d_quote_events,
    const HawkesEvent* d_trade_events,
    float* d_quote_intensities,
    int n_quotes,
    int n_trades,
    float mu_quote,
    float alpha_quote_quote,
    float alpha_trade_quote,
    float beta
);

void launch_trade_intensity_kernel(
    const HawkesEvent* d_quote_events,
    const HawkesEvent* d_trade_events,
    float* d_trade_intensities,
    int n_quotes,
    int n_trades,
    float mu_trade,
    float alpha_quote_trade,
    float alpha_trade_trade,
    float beta
);

void launch_price_impact_hawkes_kernel(
    const HawkesEvent* d_events,
    float* d_price_impacts,
    float* d_impact_intensities,
    int n_events,
    float impact_decay,
    float impact_strength
);

// Advanced statistical measures
void launch_hawkes_volatility_kernel(
    const HawkesEvent* d_events,
    const float* d_intensities,
    float* d_volatility_measures,
    int n_events,
    float time_window_ns
);

void launch_regime_detection_kernel(
    const float* d_intensities,
    int* d_regime_indicators,
    float* d_regime_probabilities,
    int n_events,
    int n_regimes
);

void launch_jump_detection_kernel(
    const float* d_intensities,
    int* d_jump_indicators,
    float* d_jump_sizes,
    int n_events,
    float jump_threshold
);

// Utility kernels
void launch_time_aggregation_kernel(
    const HawkesEvent* d_events,
    const float* d_intensities,
    HawkesStatistics* d_aggregated_stats,
    int n_events,
    uint64_t bucket_size_ns,
    int* d_bucket_counts
);

void launch_cross_correlation_kernel(
    const float* d_intensities1,
    const float* d_intensities2,
    float* d_cross_correlations,
    int n_events,
    int max_lag
);

#ifdef __cplusplus
}
#endif

#endif // HAWKES_GPU_KERNELS_H

