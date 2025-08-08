#!/usr/bin/env python3
"""
Ultra-Fast Hawkes Processes Analyzer - Python Wrapper
Provides Python interface to C++/CUDA/SIMD Hawkes process implementation
"""

import ctypes
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# C++ structure definitions matching the header files
class HawkesEvent(ctypes.Structure):
    _fields_ = [
        ("timestamp_ns", ctypes.c_uint64),
        ("mark", ctypes.c_int),
        ("intensity", ctypes.c_float),
        ("price", ctypes.c_float),
        ("size", ctypes.c_float),
        ("exchange_id", ctypes.c_int),
        ("ticker", ctypes.c_char * 8),
        ("reserved", ctypes.c_float)
    ]

class HawkesParameters(ctypes.Structure):
    _fields_ = [
        ("mu", ctypes.c_float),
        ("alpha", ctypes.c_float),
        ("beta", ctypes.c_float),
        ("branching_ratio", ctypes.c_float),
        ("log_likelihood", ctypes.c_float),
        ("aic", ctypes.c_float),
        ("bic", ctypes.c_float),
        ("n_events", ctypes.c_int)
    ]

class HawkesStatistics(ctypes.Structure):
    _fields_ = [
        ("mean_intensity", ctypes.c_float),
        ("max_intensity", ctypes.c_float),
        ("intensity_variance", ctypes.c_float),
        ("clustering_coefficient", ctypes.c_float),
        ("burstiness", ctypes.c_float),
        ("memory_coefficient", ctypes.c_float),
        ("criticality_index", ctypes.c_float),
        ("goodness_of_fit", ctypes.c_float)
    ]

class HawkesEngineConfig(ctypes.Structure):
    _fields_ = [
        ("use_gpu", ctypes.c_bool),
        ("gpu_device_id", ctypes.c_int),
        ("gpu_threshold", ctypes.c_size_t),
        ("max_events", ctypes.c_size_t),
        ("estimate_parameters", ctypes.c_bool),
        ("learning_rate", ctypes.c_float),
        ("max_iterations", ctypes.c_int),
        ("convergence_threshold", ctypes.c_float),
        ("clustering_window_ns", ctypes.c_uint64),
        ("max_memory_lag", ctypes.c_int),
        ("enable_profiling", ctypes.c_bool),
        ("verbose_output", ctypes.c_bool)
    ]

@dataclass
class HawkesAnalysisResults:
    """Python-friendly results structure"""
    n_events: int
    processing_method: str
    processing_time_ms: float
    throughput_events_per_sec: float
    
    # Parameters
    mu: float
    alpha: float
    beta: float
    branching_ratio: float
    log_likelihood: float
    aic: float
    bic: float
    
    # Statistics
    mean_intensity: float
    max_intensity: float
    intensity_variance: float
    clustering_coefficient: float
    burstiness: float
    memory_coefficient: float
    criticality_index: float
    goodness_of_fit: float
    
    # Detailed results
    intensities: np.ndarray
    clustering_coefficients: np.ndarray
    residuals: np.ndarray
    
    # Model validation
    is_subcritical: bool
    model_fit_quality: float
    validation_notes: str

class UltraFastHawkesAnalyzer:
    """
    Ultra-fast Hawkes process analyzer using GPU/SIMD acceleration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Hawkes analyzer
        
        Args:
            config: Configuration dictionary with keys:
                - use_gpu: bool (default True)
                - gpu_device_id: int (default 0)
                - max_events: int (default 1000000)
                - estimate_parameters: bool (default True)
                - learning_rate: float (default 0.01)
                - max_iterations: int (default 1000)
                - clustering_window_ns: int (default 1e9)
        """
        self.lib = None
        self.engine = None
        self.config = self._create_config(config or {})
        
        # Load the compiled library
        self._load_library()
        
        # Initialize the engine
        self._initialize_engine()
        
        print(f"✓ UltraFastHawkesAnalyzer initialized")
        print(f"  GPU: {'Enabled' if self.config.use_gpu else 'Disabled'}")
        print(f"  Max events: {self.config.max_events:,}")
        print(f"  SIMD level: {self._get_simd_level()}")
    
    def _create_config(self, config_dict: Dict) -> HawkesEngineConfig:
        """Create C++ config structure from Python dict"""
        config = HawkesEngineConfig()
        config.use_gpu = config_dict.get('use_gpu', True)
        config.gpu_device_id = config_dict.get('gpu_device_id', 0)
        config.gpu_threshold = config_dict.get('gpu_threshold', 1000)
        config.max_events = config_dict.get('max_events', 1000000)
        config.estimate_parameters = config_dict.get('estimate_parameters', True)
        config.learning_rate = config_dict.get('learning_rate', 0.01)
        config.max_iterations = config_dict.get('max_iterations', 1000)
        config.convergence_threshold = config_dict.get('convergence_threshold', 1e-6)
        config.clustering_window_ns = config_dict.get('clustering_window_ns', int(1e9))
        config.max_memory_lag = config_dict.get('max_memory_lag', 50)
        config.enable_profiling = config_dict.get('enable_profiling', False)
        config.verbose_output = config_dict.get('verbose_output', False)
        return config
    
    def _load_library(self):
        """Load the compiled C++/CUDA library"""
        # Try different possible library names and locations
        possible_paths = [
            "./libhawkes_engine.so",
            "./build/libhawkes_engine.so",
            "./lib/libhawkes_engine.so",
            "/usr/local/lib/libhawkes_engine.so"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    self.lib = ctypes.CDLL(path)
                    print(f"✓ Loaded library: {path}")
                    break
                except Exception as e:
                    print(f"✗ Failed to load {path}: {e}")
                    continue
        
        if self.lib is None:
            raise RuntimeError("Could not load Hawkes engine library. Please compile first.")
        
        # Define function signatures
        self._define_function_signatures()
    
    def _define_function_signatures(self):
        """Define C function signatures for proper calling"""
        # Engine creation/destruction
        self.lib.create_hawkes_engine.argtypes = [ctypes.POINTER(HawkesEngineConfig)]
        self.lib.create_hawkes_engine.restype = ctypes.c_void_p
        
        self.lib.destroy_hawkes_engine.argtypes = [ctypes.c_void_p]
        self.lib.destroy_hawkes_engine.restype = None
        
        # Main analysis function
        self.lib.analyze_hawkes_process.argtypes = [
            ctypes.c_void_p,  # engine
            ctypes.POINTER(HawkesEvent),  # events
            ctypes.c_size_t,  # n_events
            ctypes.POINTER(HawkesParameters),  # initial_params
            ctypes.POINTER(HawkesParameters),  # result_params
            ctypes.POINTER(HawkesStatistics),  # result_stats
            ctypes.POINTER(ctypes.c_float),  # intensities
            ctypes.POINTER(ctypes.c_float),  # clustering_coeffs
            ctypes.POINTER(ctypes.c_float),  # residuals
            ctypes.POINTER(ctypes.c_double),  # processing_time_ms
            ctypes.POINTER(ctypes.c_double)   # throughput
        ]
        self.lib.analyze_hawkes_process.restype = ctypes.c_bool
        
        # Simulation function
        self.lib.simulate_hawkes_process.argtypes = [
            ctypes.c_void_p,  # engine
            ctypes.POINTER(HawkesParameters),  # params
            ctypes.c_uint64,  # start_time_ns
            ctypes.c_uint64,  # end_time_ns
            ctypes.c_int,     # max_events
            ctypes.POINTER(HawkesEvent),  # output_events
            ctypes.POINTER(ctypes.c_int)  # n_simulated
        ]
        self.lib.simulate_hawkes_process.restype = ctypes.c_bool
        
        # Utility functions
        self.lib.get_simd_level_name.argtypes = [ctypes.c_void_p]
        self.lib.get_simd_level_name.restype = ctypes.c_char_p
    
    def _initialize_engine(self):
        """Initialize the C++ engine"""
        self.engine = self.lib.create_hawkes_engine(ctypes.byref(self.config))
        if not self.engine:
            raise RuntimeError("Failed to create Hawkes engine")
    
    def _get_simd_level(self) -> str:
        """Get SIMD capability level"""
        if self.engine and hasattr(self.lib, 'get_simd_level_name'):
            result = self.lib.get_simd_level_name(self.engine)
            return result.decode('utf-8') if result else "Unknown"
        return "Unknown"
    
    def analyze_hawkes_process(
        self,
        events_df: pd.DataFrame,
        initial_params: Optional[Dict] = None,
        ticker: str = "DEFAULT"
    ) -> HawkesAnalysisResults:
        """
        Analyze Hawkes process from pandas DataFrame
        
        Args:
            events_df: DataFrame with columns ['timestamp_ns', 'mark', 'price', 'size', 'exchange_id']
            initial_params: Initial parameter guess {'mu': float, 'alpha': float, 'beta': float}
            ticker: Stock ticker symbol
            
        Returns:
            HawkesAnalysisResults object with all analysis results
        """
        if events_df.empty:
            raise ValueError("Events DataFrame is empty")
        
        # Validate required columns
        required_cols = ['timestamp_ns']
        missing_cols = [col for col in required_cols if col not in events_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert DataFrame to C++ events array
        events_array, n_events = self._convert_dataframe_to_events(events_df, ticker)
        
        # Set initial parameters
        initial_params = initial_params or {'mu': 0.1, 'alpha': 0.5, 'beta': 1.0}
        c_initial_params = HawkesParameters()
        c_initial_params.mu = initial_params['mu']
        c_initial_params.alpha = initial_params['alpha']
        c_initial_params.beta = initial_params['beta']
        
        # Prepare output structures
        result_params = HawkesParameters()
        result_stats = HawkesStatistics()
        
        # Allocate arrays for detailed results
        intensities = (ctypes.c_float * n_events)()
        clustering_coeffs = (ctypes.c_float * n_events)()
        residuals = (ctypes.c_float * n_events)()
        processing_time_ms = ctypes.c_double()
        throughput = ctypes.c_double()
        
        # Call C++ analysis function
        start_time = time.time()
        success = self.lib.analyze_hawkes_process(
            self.engine,
            events_array,
            n_events,
            ctypes.byref(c_initial_params),
            ctypes.byref(result_params),
            ctypes.byref(result_stats),
            intensities,
            clustering_coeffs,
            residuals,
            ctypes.byref(processing_time_ms),
            ctypes.byref(throughput)
        )
        
        if not success:
            raise RuntimeError("Hawkes process analysis failed")
        
        # Convert results to Python-friendly format
        results = HawkesAnalysisResults(
            n_events=n_events,
            processing_method="GPU + SIMD" if self.config.use_gpu else "SIMD",
            processing_time_ms=processing_time_ms.value,
            throughput_events_per_sec=throughput.value,
            
            # Parameters
            mu=result_params.mu,
            alpha=result_params.alpha,
            beta=result_params.beta,
            branching_ratio=result_params.branching_ratio,
            log_likelihood=result_params.log_likelihood,
            aic=result_params.aic,
            bic=result_params.bic,
            
            # Statistics
            mean_intensity=result_stats.mean_intensity,
            max_intensity=result_stats.max_intensity,
            intensity_variance=result_stats.intensity_variance,
            clustering_coefficient=result_stats.clustering_coefficient,
            burstiness=result_stats.burstiness,
            memory_coefficient=result_stats.memory_coefficient,
            criticality_index=result_stats.criticality_index,
            goodness_of_fit=result_stats.goodness_of_fit,
            
            # Detailed results
            intensities=np.array([intensities[i] for i in range(n_events)]),
            clustering_coefficients=np.array([clustering_coeffs[i] for i in range(n_events)]),
            residuals=np.array([residuals[i] for i in range(n_events)]),
            
            # Model validation
            is_subcritical=result_params.branching_ratio < 1.0,
            model_fit_quality=result_stats.goodness_of_fit,
            validation_notes=self._generate_validation_notes(result_params, result_stats)
        )
        
        return results
    
    def simulate_hawkes_process(
        self,
        params: Dict,
        start_time_ns: int,
        end_time_ns: int,
        max_events: int = 10000,
        ticker: str = "SIM"
    ) -> pd.DataFrame:
        """
        Simulate Hawkes process events
        
        Args:
            params: Parameters {'mu': float, 'alpha': float, 'beta': float}
            start_time_ns: Start time in nanoseconds
            end_time_ns: End time in nanoseconds
            max_events: Maximum number of events to simulate
            ticker: Ticker symbol for simulated events
            
        Returns:
            DataFrame with simulated events
        """
        # Set parameters
        c_params = HawkesParameters()
        c_params.mu = params['mu']
        c_params.alpha = params['alpha']
        c_params.beta = params['beta']
        
        # Allocate output arrays
        simulated_events = (HawkesEvent * max_events)()
        n_simulated = ctypes.c_int()
        
        # Call simulation function
        success = self.lib.simulate_hawkes_process(
            self.engine,
            ctypes.byref(c_params),
            start_time_ns,
            end_time_ns,
            max_events,
            simulated_events,
            ctypes.byref(n_simulated)
        )
        
        if not success:
            raise RuntimeError("Hawkes process simulation failed")
        
        # Convert to DataFrame
        events_data = []
        for i in range(n_simulated.value):
            event = simulated_events[i]
            events_data.append({
                'timestamp_ns': event.timestamp_ns,
                'mark': event.mark,
                'intensity': event.intensity,
                'price': event.price,
                'size': event.size,
                'exchange_id': event.exchange_id,
                'ticker': event.ticker.decode('utf-8')
            })
        
        return pd.DataFrame(events_data)
    
    def _convert_dataframe_to_events(self, df: pd.DataFrame, ticker: str) -> Tuple[ctypes.Array, int]:
        """Convert pandas DataFrame to C++ events array"""
        n_events = len(df)
        events_array = (HawkesEvent * n_events)()
        
        for i, row in df.iterrows():
            event = events_array[i]
            event.timestamp_ns = int(row['timestamp_ns'])
            event.mark = int(row.get('mark', 0))
            event.intensity = 0.0  # Will be calculated
            event.price = float(row.get('price', 0.0))
            event.size = float(row.get('size', 0.0))
            event.exchange_id = int(row.get('exchange_id', 0))
            
            # Ensure ticker fits in 8 characters
            ticker_bytes = ticker[:7].encode('utf-8')
            event.ticker = ticker_bytes + b'\x00' * (8 - len(ticker_bytes))
            event.reserved = 0.0
        
        return events_array, n_events
    
    def _generate_validation_notes(self, params: HawkesParameters, stats: HawkesStatistics) -> str:
        """Generate validation notes based on results"""
        notes = []
        
        if params.branching_ratio >= 1.0:
            notes.append("WARNING: Process is supercritical (explosive)")
        elif params.branching_ratio > 0.9:
            notes.append("CAUTION: Process is near-critical")
        else:
            notes.append("Process is subcritical (stable)")
        
        if stats.goodness_of_fit < 0.05:
            notes.append("Poor model fit (p < 0.05)")
        elif stats.goodness_of_fit > 0.1:
            notes.append("Good model fit")
        
        if stats.clustering_coefficient > 0.7:
            notes.append("High temporal clustering detected")
        
        if stats.burstiness > 0.5:
            notes.append("High burstiness in event timing")
        
        return "; ".join(notes)
    
    def create_comprehensive_visualizations(
        self,
        results: HawkesAnalysisResults,
        events_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Create comprehensive visualizations of Hawkes process analysis
        
        Args:
            results: Analysis results
            events_df: Original events DataFrame
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object if plotting is available
        """
        if not HAS_PLOTTING:
            print("Matplotlib not available. Install with: pip install matplotlib seaborn")
            return None
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Hawkes Process Analysis - {results.n_events:,} Events', fontsize=16, fontweight='bold')
        
        # Convert timestamps to relative time in seconds
        times = (events_df['timestamp_ns'].values - events_df['timestamp_ns'].iloc[0]) * 1e-9
        
        # 1. Event timeline
        axes[0, 0].scatter(times, np.ones(len(times)), alpha=0.6, s=1)
        axes[0, 0].set_title('Event Timeline')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Events')
        
        # 2. Intensity evolution
        axes[0, 1].plot(times, results.intensities, linewidth=1, alpha=0.8)
        axes[0, 1].set_title(f'Intensity Evolution (μ={results.mu:.3f})')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Intensity λ(t)')
        
        # 3. Intensity distribution
        axes[0, 2].hist(results.intensities, bins=50, alpha=0.7, density=True)
        axes[0, 2].set_title('Intensity Distribution')
        axes[0, 2].set_xlabel('Intensity')
        axes[0, 2].set_ylabel('Density')
        
        # 4. Parameter summary
        axes[0, 3].text(0.1, 0.8, f'μ (base): {results.mu:.4f}', fontsize=12, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.7, f'α (jump): {results.alpha:.4f}', fontsize=12, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.6, f'β (decay): {results.beta:.4f}', fontsize=12, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.5, f'Branching: {results.branching_ratio:.4f}', fontsize=12, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.4, f'Log-likelihood: {results.log_likelihood:.2f}', fontsize=12, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.3, f'AIC: {results.aic:.2f}', fontsize=12, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.2, f'Processing: {results.processing_time_ms:.2f}ms', fontsize=12, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.1, f'Method: {results.processing_method}', fontsize=12, transform=axes[0, 3].transAxes)
        axes[0, 3].set_title('Analysis Summary')
        axes[0, 3].axis('off')
        
        # 5. Clustering coefficients
        axes[1, 0].plot(times, results.clustering_coefficients, linewidth=1, alpha=0.8)
        axes[1, 0].set_title(f'Clustering Coefficient (avg={results.clustering_coefficient:.3f})')
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Clustering')
        
        # 6. Residuals analysis
        axes[1, 1].plot(times, results.residuals, linewidth=1, alpha=0.8)
        axes[1, 1].set_title('Residuals (Transformed Times)')
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Residuals')
        
        # 7. Residuals QQ plot
        if HAS_SCIPY:
            stats.probplot(results.residuals, dist="expon", plot=axes[1, 2])
            axes[1, 2].set_title('Residuals Q-Q Plot (Exponential)')
        else:
            axes[1, 2].hist(results.residuals, bins=30, alpha=0.7, density=True)
            axes[1, 2].set_title('Residuals Distribution')
        
        # 8. Inter-arrival times
        inter_arrivals = np.diff(times)
        axes[1, 3].hist(inter_arrivals, bins=50, alpha=0.7, density=True)
        axes[1, 3].set_title(f'Inter-arrival Times (Burstiness={results.burstiness:.3f})')
        axes[1, 3].set_xlabel('Time (seconds)')
        axes[1, 3].set_ylabel('Density')
        
        # 9. Autocorrelation of intensities
        if len(results.intensities) > 50:
            lags = range(min(50, len(results.intensities)//2))
            autocorr = [np.corrcoef(results.intensities[:-lag], results.intensities[lag:])[0,1] 
                       if lag > 0 else 1.0 for lag in lags]
            axes[2, 0].plot(lags, autocorr, 'o-', linewidth=1, markersize=3)
            axes[2, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[2, 0].set_title(f'Intensity Autocorrelation (Memory={results.memory_coefficient:.3f})')
            axes[2, 0].set_xlabel('Lag')
            axes[2, 0].set_ylabel('Correlation')
        
        # 10. Model validation
        validation_text = results.validation_notes.replace(';', '\n')
        axes[2, 1].text(0.1, 0.8, 'Model Validation:', fontsize=12, fontweight='bold', transform=axes[2, 1].transAxes)
        axes[2, 1].text(0.1, 0.6, validation_text, fontsize=10, transform=axes[2, 1].transAxes, wrap=True)
        axes[2, 1].text(0.1, 0.3, f'Goodness of fit: {results.goodness_of_fit:.4f}', fontsize=11, transform=axes[2, 1].transAxes)
        axes[2, 1].text(0.1, 0.2, f'Criticality index: {results.criticality_index:.4f}', fontsize=11, transform=axes[2, 1].transAxes)
        axes[2, 1].text(0.1, 0.1, f'Subcritical: {results.is_subcritical}', fontsize=11, transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('Model Validation')
        axes[2, 1].axis('off')
        
        # 11. Performance metrics
        axes[2, 2].text(0.1, 0.8, 'Performance Metrics:', fontsize=12, fontweight='bold', transform=axes[2, 2].transAxes)
        axes[2, 2].text(0.1, 0.6, f'Processing time: {results.processing_time_ms:.2f} ms', fontsize=11, transform=axes[2, 2].transAxes)
        axes[2, 2].text(0.1, 0.5, f'Throughput: {results.throughput_events_per_sec:,.0f} events/sec', fontsize=11, transform=axes[2, 2].transAxes)
        axes[2, 2].text(0.1, 0.4, f'Method: {results.processing_method}', fontsize=11, transform=axes[2, 2].transAxes)
        axes[2, 2].text(0.1, 0.3, f'SIMD level: {self._get_simd_level()}', fontsize=11, transform=axes[2, 2].transAxes)
        axes[2, 2].text(0.1, 0.2, f'Events processed: {results.n_events:,}', fontsize=11, transform=axes[2, 2].transAxes)
        axes[2, 2].set_title('Performance Analysis')
        axes[2, 2].axis('off')
        
        # 12. Intensity vs clustering scatter
        axes[2, 3].scatter(results.intensities, results.clustering_coefficients, alpha=0.6, s=1)
        axes[2, 3].set_title('Intensity vs Clustering')
        axes[2, 3].set_xlabel('Intensity')
        axes[2, 3].set_ylabel('Clustering Coefficient')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        return fig
    
    def generate_analysis_report(self, results: HawkesAnalysisResults, ticker: str = "DEFAULT") -> str:
        """
        Generate comprehensive analysis report
        
        Args:
            results: Analysis results
            ticker: Stock ticker symbol
            
        Returns:
            Formatted analysis report string
        """
        report = f"""
# Hawkes Process Analysis Report - {ticker}

## Executive Summary
- **Events Analyzed**: {results.n_events:,}
- **Processing Method**: {results.processing_method}
- **Processing Time**: {results.processing_time_ms:.2f} ms
- **Throughput**: {results.throughput_events_per_sec:,.0f} events/second
- **Model Quality**: {'Excellent' if results.goodness_of_fit > 0.1 else 'Good' if results.goodness_of_fit > 0.05 else 'Poor'}

## Model Parameters
- **Base Intensity (μ)**: {results.mu:.6f}
  - Background event rate when no self-excitation
- **Jump Size (α)**: {results.alpha:.6f}
  - Strength of self-excitation from each event
- **Decay Rate (β)**: {results.beta:.6f}
  - Speed of intensity decay after events
- **Branching Ratio (α/β)**: {results.branching_ratio:.6f}
  - {'🔴 SUPERCRITICAL (Explosive)' if results.branching_ratio >= 1.0 else '🟡 Near-Critical' if results.branching_ratio > 0.9 else '🟢 Subcritical (Stable)'}

## Statistical Measures
- **Mean Intensity**: {results.mean_intensity:.6f}
- **Max Intensity**: {results.max_intensity:.6f}
- **Intensity Variance**: {results.intensity_variance:.6f}
- **Clustering Coefficient**: {results.clustering_coefficient:.6f}
- **Burstiness Index**: {results.burstiness:.6f}
- **Memory Coefficient**: {results.memory_coefficient:.6f}
- **Criticality Index**: {results.criticality_index:.6f}

## Model Fit Quality
- **Log-Likelihood**: {results.log_likelihood:.2f}
- **AIC**: {results.aic:.2f}
- **BIC**: {results.bic:.2f}
- **Goodness of Fit (p-value)**: {results.goodness_of_fit:.6f}

## Trading Insights
"""
        
        # Add trading insights based on parameters
        if results.branching_ratio > 0.8:
            report += "- ⚠️  **High Self-Excitation**: Events strongly trigger more events\n"
            report += "- 📈 **Momentum Strategy**: Consider momentum-based trading\n"
        else:
            report += "- 📊 **Moderate Self-Excitation**: Balanced event clustering\n"
            report += "- ⚖️  **Mean Reversion**: Consider mean-reversion strategies\n"
        
        if results.clustering_coefficient > 0.5:
            report += "- 🔄 **High Clustering**: Events occur in bursts\n"
            report += "- ⏰ **Timing Strategy**: Focus on cluster detection\n"
        
        if results.burstiness > 0.3:
            report += "- 💥 **Bursty Behavior**: Irregular event timing\n"
            report += "- 🎯 **Volatility Trading**: High volatility periods detected\n"
        
        if results.memory_coefficient > 0.3:
            report += "- 🧠 **Long Memory**: Past events influence future significantly\n"
            report += "- 📊 **Trend Following**: Consider trend-following strategies\n"
        
        report += f"""
## Validation Notes
{results.validation_notes}

## Performance Analysis
- **Processing Speed**: {results.throughput_events_per_sec:,.0f} events/second
- **Scalability**: {'Excellent' if results.throughput_events_per_sec > 100000 else 'Good' if results.throughput_events_per_sec > 10000 else 'Moderate'}
- **Method Used**: {results.processing_method}
- **SIMD Level**: {self._get_simd_level()}

## Recommendations
"""
        
        if results.is_subcritical:
            report += "- ✅ **Stable Process**: Safe for modeling and prediction\n"
        else:
            report += "- ⚠️  **Unstable Process**: Use with caution in risk models\n"
        
        if results.goodness_of_fit > 0.05:
            report += "- ✅ **Good Model Fit**: Hawkes model appropriate\n"
        else:
            report += "- ⚠️  **Poor Model Fit**: Consider alternative models\n"
        
        report += f"""
---
*Report generated by UltraFastHawkesAnalyzer*
*Analysis completed in {results.processing_time_ms:.2f} ms*
"""
        
        return report
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if self.engine and self.lib:
            try:
                self.lib.destroy_hawkes_engine(self.engine)
            except:
                pass

# Utility functions for data preparation
def prepare_quote_data_for_hawkes(
    quotes_df: pd.DataFrame,
    event_types: List[str] = ['quote_update', 'spread_change', 'size_change']
) -> pd.DataFrame:
    """
    Prepare quote data for Hawkes analysis by detecting different event types
    
    Args:
        quotes_df: DataFrame with quote data
        event_types: Types of events to detect
        
    Returns:
        DataFrame formatted for Hawkes analysis
    """
    events = []
    
    for i, row in quotes_df.iterrows():
        if i == 0:
            continue
            
        prev_row = quotes_df.iloc[i-1]
        
        # Detect quote updates
        if 'quote_update' in event_types:
            if (row['bid_price'] != prev_row['bid_price'] or 
                row['ask_price'] != prev_row['ask_price']):
                events.append({
                    'timestamp_ns': row['sip_timestamp'],
                    'mark': 0,  # Quote update
                    'price': (row['bid_price'] + row['ask_price']) / 2,
                    'size': row['bid_size'] + row['ask_size'],
                    'exchange_id': row.get('exchange_id', 0)
                })
        
        # Detect spread changes
        if 'spread_change' in event_types:
            current_spread = row['ask_price'] - row['bid_price']
            prev_spread = prev_row['ask_price'] - prev_row['bid_price']
            if abs(current_spread - prev_spread) > 0.001:  # Threshold
                events.append({
                    'timestamp_ns': row['sip_timestamp'],
                    'mark': 1,  # Spread change
                    'price': current_spread,
                    'size': row['bid_size'] + row['ask_size'],
                    'exchange_id': row.get('exchange_id', 0)
                })
        
        # Detect size changes
        if 'size_change' in event_types:
            if (row['bid_size'] != prev_row['bid_size'] or 
                row['ask_size'] != prev_row['ask_size']):
                events.append({
                    'timestamp_ns': row['sip_timestamp'],
                    'mark': 2,  # Size change
                    'price': (row['bid_price'] + row['ask_price']) / 2,
                    'size': abs(row['bid_size'] - prev_row['bid_size']) + abs(row['ask_size'] - prev_row['ask_size']),
                    'exchange_id': row.get('exchange_id', 0)
                })
    
    return pd.DataFrame(events).sort_values('timestamp_ns').reset_index(drop=True)

def convert_trade_data_for_hawkes(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert trade data to Hawkes event format
    
    Args:
        trades_df: DataFrame with trade data
        
    Returns:
        DataFrame formatted for Hawkes analysis
    """
    events = []
    
    for _, row in trades_df.iterrows():
        events.append({
            'timestamp_ns': row['timestamp_ns'],
            'mark': 3,  # Trade event
            'price': row['price'],
            'size': row['size'],
            'exchange_id': row.get('exchange_id', 0)
        })
    
    return pd.DataFrame(events).sort_values('timestamp_ns').reset_index(drop=True)

# Example usage and testing
def main():
    """Example usage of the UltraFastHawkesAnalyzer"""
    
    # Configuration
    config = {
        'use_gpu': True,
        'max_events': 100000,
        'estimate_parameters': True,
        'learning_rate': 0.01,
        'max_iterations': 500
    }
    
    # Initialize analyzer
    analyzer = UltraFastHawkesAnalyzer(config)
    
    # Generate sample data
    print("Generating sample Hawkes process data...")
    sample_params = {'mu': 0.1, 'alpha': 0.3, 'beta': 1.0}
    start_time = int(time.time() * 1e9)  # Current time in nanoseconds
    end_time = start_time + int(3600 * 1e9)  # 1 hour later
    
    # Simulate events
    simulated_df = analyzer.simulate_hawkes_process(
        sample_params, start_time, end_time, max_events=5000
    )
    
    print(f"✓ Simulated {len(simulated_df)} events")
    
    # Analyze the simulated events
    print("Analyzing Hawkes process...")
    results = analyzer.analyze_hawkes_process(
        simulated_df, 
        initial_params={'mu': 0.05, 'alpha': 0.2, 'beta': 0.8}
    )
    
    print(f"✓ Analysis completed in {results.processing_time_ms:.2f} ms")
    print(f"✓ Throughput: {results.throughput_events_per_sec:,.0f} events/second")
    
    # Generate report
    report = analyzer.generate_analysis_report(results, "SAMPLE")
    print(report)
    
    # Create visualizations
    if HAS_PLOTTING:
        fig = analyzer.create_comprehensive_visualizations(
            results, simulated_df, save_path="hawkes_analysis.png"
        )
        plt.show()
    
    print("✓ Hawkes process analysis completed successfully!")

if __name__ == "__main__":
    main()

