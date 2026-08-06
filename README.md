# 📊 Performance Benchmarking Suite

Complete performance testing and benchmarking system for SentinelAI detectors.

## Overview

This suite provides:
- Individual detector performance measurement
- Multi-detector orchestration testing
- Multi-camera load testing (up to 20+ cameras)
- Memory profiling and leak detection
- Latency distribution analysis
- Automated report generation
- Performance regression detection

## Quick Start

### Run All Benchmarks
```bash
python tests/performance/benchmark_runner.py
```

### Generate Reports
```bash
python tests/performance/benchmark_reporter.py
```

### View HTML Report
Open `tests/performance/BENCHMARK_REPORT.html` in browser

## Benchmark Types

### 1. Single Detector Benchmark
Tests each detector independently with 100 frames.

```python
from tests.performance.benchmark_runner import PerformanceBenchmark

benchmark = PerformanceBenchmark()
result = benchmark.benchmark_single_detector(detector, 'Person', num_frames=100)
```

**Measures**:
- FPS (frames per second)
- Latency (average, min, max)
- Memory (used, peak)
- CPU usage

**Performance Requirements**:
- Person: 60+ FPS, <100ms latency
- Vehicle: 30+ FPS, <150ms latency
- Animal: 40+ FPS, <120ms latency
- Object: 35+ FPS, <140ms latency

### 2. Multi-Detector Sequential
Processes frames through all detectors sequentially.

```python
detectors = {
    'Person': PersonDetector(config),
    'Vehicle': VehicleDetector(config),
    'Animal': AnimalDetector(config),
    'Object': ObjectDetector(config)
}

results = benchmark.benchmark_multi_detector(detectors, num_frames=50)
```

**Measures**:
- Per-detector performance
- Orchestrator latency
- Bottleneck identification

### 3. Multi-Camera Load Test
Simulates multiple concurrent camera streams.

```python
for num_cameras in [5, 10, 20]:
    result = benchmark.benchmark_load_test(detectors, num_cameras=num_cameras)
```

**Configurations**:
- 5 cameras: 250 frames total
- 10 cameras: 500 frames total
- 20 cameras: 1,000 frames total

**Measures**:
- Aggregate FPS (all cameras)
- Per-camera FPS
- Latency percentiles (P95, P99)
- Memory growth

### 4. Memory Profiling
Tracks memory usage over 100 frames.

```python
result = benchmark.benchmark_memory_usage(detector, 'Person')
```

**Measures**:
- Average memory
- Peak memory
- Minimum memory
- Memory fluctuation (leak detection)

### 5. Latency Distribution
Analyzes latency distribution over 1,000 frames.

```python
result = benchmark.benchmark_latency_distribution(detector, 'Person', num_frames=1000)
```

**Percentiles**:
- P50 (median)
- P75
- P90
- P95
- P99
- P99.9

## Report Generation

### Text Report
```bash
python tests/performance/benchmark_reporter.py
```

Generates: `BENCHMARK_REPORT.txt`
- Detector-by-detector summary
- Requirements validation
- Performance metrics

### HTML Report
Same command generates: `BENCHMARK_REPORT.html`
- Visual performance summary
- Interactive tables
- Color-coded requirements (green/red)
- Sortable results

## Performance Requirements

### Individual Detectors

| Detector | Minimum FPS | Maximum Latency |
|----------|-------------|-----------------|
| Person | 60+ | 100ms |
| Vehicle | 30+ | 150ms |
| Animal | 40+ | 120ms |
| Object | 35+ | 140ms |

### Multi-Camera System

| Configuration | Minimum Throughput | P95 Latency |
|---------------|-------------------|------------|
| 5 cameras | 40+ FPS aggregate | <120ms |
| 10 cameras | 40+ FPS aggregate | <130ms |
| 20 cameras | 40+ FPS aggregate | <150ms |

## Understanding Latency Percentiles

### What Percentiles Mean

**P50 (Median)**: 50% of requests are faster than this
- Typical case
- Most representative of "normal" performance

**P95**: 95% of requests are faster than this
- Operator experience (rarely see worse)
- SLA target

**P99**: 99% of requests are faster than this
- Almost all users experience this or better
- Production SLA target

**P99.9**: 99.9% of requests are faster than this
- Rare events
- Worst-case performance

### Why Percentiles Matter

Average (mean) latency can be misleading:
```
Request latencies: 10ms, 10ms, 10ms, 10ms, 500ms
Average: 106ms (misleading - 4/5 are fast!)
P95: 10ms (accurate - operator's actual experience)
```

## Regression Detection

Detect performance degradation vs baseline:

```python
from tests.performance.benchmark_reporter import BenchmarkReporter

reporter = BenchmarkReporter()
regression = reporter.check_performance_regression('baseline.json')

if regression['status'] == 'PASS':
    print("✓ No regression")
else:
    for reg in regression['regressions']:
        print(f"✗ {reg['detector']}: {reg['change_percent']:.1f}% slower")
```

### Regression Thresholds
- >5% FPS degradation: FAIL
- <5% improvement: track only

## Configuration

### Custom Benchmark
```python
from tests.performance.benchmark_runner import PerformanceBenchmark
from ai.config import ConfigManager

# Load custom config
config = ConfigManager.load_config()

# Create benchmark
benchmark = PerformanceBenchmark(config)

# Run custom test
result = benchmark.benchmark_single_detector(detector, 'Person', num_frames=200)
```

### Image Resolution Control
Modify image generation in `benchmark_runner.py`:
```python
resolutions = [(480, 640), (720, 1280), (1080, 1920)]
```

### Frame Count Adjustment
More frames = more accurate but longer test:
```python
# Quick test (5 min)
benchmark_single_detector(detector, name, num_frames=50)

# Standard test (10 min)
benchmark_single_detector(detector, name, num_frames=100)

# Comprehensive test (20 min)
benchmark_single_detector(detector, name, num_frames=200)
```

## Interpreting Results

### Good Results ✓
- All detectors > minimum FPS
- P95 latency within budget
- Memory stable (no leaks)
- Aggregate FPS decreases smoothly with cameras

### Warning Signs ⚠️
- Any detector < minimum FPS
- P99 latency > 1.5x budget
- Memory grows unbounded
- Non-linear FPS decrease with cameras

### Failure ✗
- Consistent failures on requirements
- Memory leaks (growing peak)
- Unpredictable latency spikes
- Aggregate FPS collapse at >10 cameras

## Optimization Guide

### If FPS Low
1. Check `detection.model_size` in config (try smaller)
2. Reduce input resolution (if acceptable)
3. Check CPU/GPU utilization
4. Look for memory pressure (swap usage)

### If Latency High
1. Check P99 latency (not just average)
2. Look for outliers (system events)
3. Monitor CPU/GPU during test
4. Check for memory pressure

### If Memory Growing
1. Run memory profile (100 frames)
2. Check for memory leaks in detector
3. Verify batch size setting
4. Monitor over longer period (1000+ frames)

### If Load Test Fails at 20 Cameras
1. Verify CPU has enough cores
2. Check GPU memory (if using)
3. Look at per-camera latency
4. Consider system limits (I/O, network)

## CI/CD Integration

### GitHub Actions
```yaml
name: Performance Tests

on: [push]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run benchmarks
        run: python tests/performance/benchmark_runner.py
      - name: Generate reports
        run: python tests/performance/benchmark_reporter.py
      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-reports
          path: tests/performance/BENCHMARK_REPORT.*
```

### GitLab CI
```yaml
performance_test:
  script:
    - python tests/performance/benchmark_runner.py
    - python tests/performance/benchmark_reporter.py
  artifacts:
    paths:
      - tests/performance/BENCHMARK_REPORT.html
    reports:
      dotenv: metrics.env
  allow_failure: false
```

## JSON Results Format

Results saved to `benchmark_results.json`:

```json
[
  {
    "detector_name": "Person",
    "test_name": "single_detector",
    "frames_processed": 100,
    "total_time": 1.61,
    "avg_latency": 0.016,
    "fps": 62.3,
    "min_latency": 0.014,
    "max_latency": 0.021,
    "memory_used": 12.3,
    "peak_memory": 389.2,
    "cpu_percent": 85.4,
    "gpu_memory_used": 0.0,
    "timestamp": "2026-08-06T10:30:45.123456"
  }
]
```

## File Structure

```
tests/performance/
├── benchmark_runner.py       # Main benchmark orchestrator
├── benchmark_reporter.py     # Report generation
├── __init__.py              # Module exports
├── README.md                # This file
├── benchmark_results.json   # Generated results
├── BENCHMARK_REPORT.txt     # Generated text report
└── BENCHMARK_REPORT.html    # Generated HTML report
```

## Troubleshooting

### "Module not found: ai.detection"
Ensure you're running from project root:
```bash
cd /path/to/SentinelAI
python tests/performance/benchmark_runner.py
```

### "CUDA out of memory"
Reduce batch size or use CPU:
```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU
python tests/performance/benchmark_runner.py
```

### "Results file not found"
Run benchmark_runner first:
```bash
python tests/performance/benchmark_runner.py
python tests/performance/benchmark_reporter.py
```

### Inconsistent Results
- Close other applications
- Ensure system not under load
- Run multiple times and average
- Check thermal throttling (CPU temp)

## Performance Tuning Tips

### For Maximum FPS
1. Use smaller model size (nano/small)
2. Reduce input resolution
3. Use GPU if available
4. Batch multiple frames

### For Predictable Latency
1. Use sequential processing mode
2. Pre-warm GPU (run 10 frames first)
3. Monitor for CPU thermal throttling
4. Check system load

### For Production Deployment
1. Establish baseline with representative hardware
2. Run regression test before each release
3. Monitor P99 latency (not average)
4. Plan for 30% performance headroom

## Support

For issues:
1. Check `BENCHMARK_REPORT.txt` for details
2. View `BENCHMARK_REPORT.html` for visualization
3. Review `benchmark_results.json` for raw data
4. Check system resources (CPU, GPU, memory)

## Summary

The Performance Benchmarking Suite provides:
- ✅ Comprehensive detector performance measurement
- ✅ Multi-camera load testing
- ✅ Memory profiling and leak detection
- ✅ Latency analysis with percentiles
- ✅ Automated report generation
- ✅ Regression detection
- ✅ Production-grade performance data

**Use Case**: Verify system meets performance requirements before production deployment.
