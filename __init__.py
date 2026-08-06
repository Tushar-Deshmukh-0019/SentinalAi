"""
Performance Benchmarking Suite

Comprehensive performance testing and benchmarking system for SentinelAI with:
- Individual detector benchmarks
- Multi-camera load testing
- Memory profiling
- Latency analysis
- Automated report generation
- Performance regression detection

Usage:
    from tests.performance.benchmark_runner import PerformanceBenchmark
    
    benchmark = PerformanceBenchmark()
    result = benchmark.benchmark_single_detector(detector, 'Person', num_frames=100)
    benchmark.save_results()
    
    from tests.performance.benchmark_reporter import BenchmarkReporter
    reporter = BenchmarkReporter()
    reporter.generate_text_report()
    reporter.generate_html_report()
"""

from .benchmark_runner import PerformanceBenchmark, BenchmarkResult, run_comprehensive_benchmark
from .benchmark_reporter import BenchmarkReporter, PerformanceMetrics

__all__ = [
    'PerformanceBenchmark',
    'BenchmarkResult',
    'BenchmarkReporter',
    'PerformanceMetrics',
    'run_comprehensive_benchmark'
]
