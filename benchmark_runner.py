"""
Performance Benchmark Runner

Comprehensive performance benchmarking system for SentinelAI with:
- Individual detector benchmarks
- Multi-camera load testing
- Real-time FPS tracking
- Memory profiling
- Latency analysis
- Throughput measurement
- Resource utilization tracking
"""

import time
import psutil
import threading
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import cv2
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.detection.person.detector import PersonDetector
from ai.detection.vehicle.detector import VehicleDetector
from ai.detection.animal.detector import AnimalDetector
from ai.detection.object.detector import ObjectDetector
from ai.config import ConfigManager
from ai.logging import setup_logger

logger = setup_logger('performance.benchmark')


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    detector_name: str
    test_name: str
    frames_processed: int
    total_time: float
    avg_latency: float
    fps: float
    min_latency: float
    max_latency: float
    memory_used: float
    peak_memory: float
    cpu_percent: float
    gpu_memory_used: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class PerformanceBenchmark:
    """Main benchmark orchestrator"""
    
    def __init__(self, config=None):
        """Initialize benchmark system"""
        self.config = config or ConfigManager.load_config()
        self.results: List[BenchmarkResult] = []
        self.sample_images = self._generate_sample_images()
        
        logger.info("Performance Benchmark System Initialized")
    
    def _generate_sample_images(self, count=100):
        """Generate sample images for benchmarking"""
        logger.info(f"Generating {count} sample images")
        images = []
        
        # Various resolutions for realistic testing
        resolutions = [(480, 640), (720, 1280), (1080, 1920)]
        
        for i in range(count):
            # Random resolution
            h, w = resolutions[i % len(resolutions)]
            image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            images.append(image)
        
        logger.info(f"Generated {len(images)} sample images")
        return images
    
    def benchmark_single_detector(self, detector, detector_name: str, 
                                 num_frames: int = 100) -> BenchmarkResult:
        """Benchmark single detector"""
        logger.info(f"\nBenchmarking {detector_name}...")
        
        latencies = []
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        for i in range(num_frames):
            image = self.sample_images[i % len(self.sample_images)]
            
            frame_start = time.time()
            result = detector.detect(image)
            frame_elapsed = time.time() - frame_start
            
            latencies.append(frame_elapsed)
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Processed {i + 1}/{num_frames} frames")
        
        total_time = time.time() - start_time
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        peak_memory = max(psutil.Process().memory_info().rss / 1024 / 1024 for _ in range(10))
        
        # Calculate metrics
        fps = num_frames / total_time
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        result = BenchmarkResult(
            detector_name=detector_name,
            test_name="single_detector",
            frames_processed=num_frames,
            total_time=total_time,
            avg_latency=avg_latency,
            fps=fps,
            min_latency=min_latency,
            max_latency=max_latency,
            memory_used=memory_used,
            peak_memory=peak_memory,
            cpu_percent=cpu_percent
        )
        
        self.results.append(result)
        
        logger.info(f"\n{detector_name} Single Detector Results:")
        logger.info(f"  Frames: {num_frames}")
        logger.info(f"  Total Time: {total_time:.2f}s")
        logger.info(f"  FPS: {fps:.2f}")
        logger.info(f"  Avg Latency: {avg_latency*1000:.2f}ms")
        logger.info(f"  Min Latency: {min_latency*1000:.2f}ms")
        logger.info(f"  Max Latency: {max_latency*1000:.2f}ms")
        logger.info(f"  Memory Used: {memory_used:.2f}MB")
        logger.info(f"  CPU: {cpu_percent:.1f}%")
        
        return result
    
    def benchmark_multi_detector(self, detectors: Dict, 
                                num_frames: int = 50) -> List[BenchmarkResult]:
        """Benchmark multiple detectors in sequence"""
        logger.info(f"\nBenchmarking {len(detectors)} detectors sequentially...")
        
        results = []
        
        for name, detector in detectors.items():
            result = self.benchmark_single_detector(detector, name, num_frames)
            results.append(result)
        
        # Calculate orchestrator overhead
        logger.info("\nCalculating orchestrator overhead...")
        orchestrator_latencies = []
        
        for i in range(num_frames):
            image = self.sample_images[i % len(self.sample_images)]
            
            start = time.time()
            for detector in detectors.values():
                detector.detect(image)
            elapsed = time.time() - start
            
            orchestrator_latencies.append(elapsed)
        
        logger.info(f"  Orchestrator Latency (all detectors): {np.mean(orchestrator_latencies)*1000:.2f}ms")
        
        return results
    
    def benchmark_load_test(self, detectors: Dict, 
                           num_cameras: int = 5,
                           frames_per_camera: int = 50) -> Dict:
        """Benchmark multi-camera load"""
        logger.info(f"\nLoad Test: {num_cameras} cameras, {frames_per_camera} frames each")
        
        total_frames = num_cameras * frames_per_camera
        latencies = []
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        
        for camera in range(num_cameras):
            for frame_id in range(frames_per_camera):
                image = self.sample_images[(camera * frames_per_camera + frame_id) % len(self.sample_images)]
                
                frame_start = time.time()
                for detector in detectors.values():
                    detector.detect(image)
                frame_elapsed = time.time() - frame_start
                
                latencies.append(frame_elapsed)
                
                if (frame_id + 1) % 10 == 0:
                    current_fps = (camera * frames_per_camera + frame_id + 1) / (time.time() - start_time)
                    logger.info(f"  Camera {camera+1}/{num_cameras}, Frame {frame_id+1}/{frames_per_camera}, "
                              f"Current FPS: {current_fps:.2f}")
        
        total_time = time.time() - start_time
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        
        fps = total_frames / total_time
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        logger.info(f"\nLoad Test Results ({num_cameras} cameras):")
        logger.info(f"  Total Frames: {total_frames}")
        logger.info(f"  Total Time: {total_time:.2f}s")
        logger.info(f"  Aggregate FPS: {fps:.2f}")
        logger.info(f"  Per-Camera FPS: {fps/num_cameras:.2f}")
        logger.info(f"  Avg Latency: {avg_latency*1000:.2f}ms")
        logger.info(f"  P95 Latency: {p95_latency*1000:.2f}ms")
        logger.info(f"  P99 Latency: {p99_latency*1000:.2f}ms")
        logger.info(f"  Memory Used: {memory_used:.2f}MB")
        
        return {
            'num_cameras': num_cameras,
            'total_frames': total_frames,
            'total_time': total_time,
            'fps': fps,
            'per_camera_fps': fps / num_cameras,
            'avg_latency': avg_latency,
            'p95_latency': p95_latency,
            'p99_latency': p99_latency,
            'memory_used': memory_used
        }
    
    def benchmark_memory_usage(self, detector, detector_name: str) -> Dict:
        """Benchmark memory usage over time"""
        logger.info(f"\nMemory Profiling {detector_name}...")
        
        memory_timeline = []
        
        for i in range(100):
            image = self.sample_images[i % len(self.sample_images)]
            detector.detect(image)
            
            memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_timeline.append(memory)
        
        avg_memory = np.mean(memory_timeline)
        peak_memory = np.max(memory_timeline)
        min_memory = np.min(memory_timeline)
        
        logger.info(f"\n{detector_name} Memory Profile:")
        logger.info(f"  Avg Memory: {avg_memory:.2f}MB")
        logger.info(f"  Peak Memory: {peak_memory:.2f}MB")
        logger.info(f"  Min Memory: {min_memory:.2f}MB")
        logger.info(f"  Fluctuation: {peak_memory - min_memory:.2f}MB")
        
        return {
            'detector': detector_name,
            'avg_memory': avg_memory,
            'peak_memory': peak_memory,
            'min_memory': min_memory,
            'fluctuation': peak_memory - min_memory
        }
    
    def benchmark_latency_distribution(self, detector, detector_name: str,
                                      num_frames: int = 1000) -> Dict:
        """Analyze latency distribution"""
        logger.info(f"\nLatency Distribution Analysis: {detector_name} ({num_frames} frames)")
        
        latencies = []
        
        for i in range(num_frames):
            image = self.sample_images[i % len(self.sample_images)]
            
            start = time.time()
            detector.detect(image)
            latencies.append((time.time() - start) * 1000)  # Convert to ms
            
            if (i + 1) % 200 == 0:
                logger.info(f"  Processed {i + 1}/{num_frames} frames")
        
        latencies = np.array(latencies)
        
        percentiles = {
            'p50': np.percentile(latencies, 50),
            'p75': np.percentile(latencies, 75),
            'p90': np.percentile(latencies, 90),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'p99_9': np.percentile(latencies, 99.9),
        }
        
        logger.info(f"\n{detector_name} Latency Distribution ({num_frames} frames):")
        logger.info(f"  Mean: {np.mean(latencies):.2f}ms")
        logger.info(f"  Median: {np.median(latencies):.2f}ms")
        logger.info(f"  StdDev: {np.std(latencies):.2f}ms")
        for perc, value in percentiles.items():
            logger.info(f"  {perc}: {value:.2f}ms")
        
        return {
            'detector': detector_name,
            'num_frames': num_frames,
            'mean': np.mean(latencies),
            'median': np.median(latencies),
            'std_dev': np.std(latencies),
            **percentiles
        }
    
    def save_results(self, output_file: str = "benchmark_results.json"):
        """Save benchmark results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_data = [asdict(r) for r in self.results]
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to {output_path}")
    
    def print_summary(self):
        """Print summary of all benchmarks"""
        logger.info("\n" + "="*70)
        logger.info("PERFORMANCE BENCHMARK SUMMARY")
        logger.info("="*70)
        
        if not self.results:
            logger.info("No results to display")
            return
        
        # Group by detector
        by_detector = {}
        for result in self.results:
            if result.detector_name not in by_detector:
                by_detector[result.detector_name] = []
            by_detector[result.detector_name].append(result)
        
        for detector, results in sorted(by_detector.items()):
            logger.info(f"\n{detector}:")
            for result in results:
                logger.info(f"  {result.test_name}:")
                logger.info(f"    FPS: {result.fps:.2f}")
                logger.info(f"    Avg Latency: {result.avg_latency*1000:.2f}ms")
                logger.info(f"    Memory: {result.memory_used:.2f}MB")


def run_comprehensive_benchmark():
    """Run complete benchmark suite"""
    logger.info("\n" + "="*70)
    logger.info("SENTINELAI PERFORMANCE BENCHMARK SUITE")
    logger.info("="*70)
    
    config = ConfigManager.load_config()
    benchmark = PerformanceBenchmark(config)
    
    # Initialize detectors
    logger.info("\nInitializing detectors...")
    detectors = {
        'Person': PersonDetector(config),
        'Vehicle': VehicleDetector(config),
        'Animal': AnimalDetector(config),
        'Object': ObjectDetector(config)
    }
    
    # Run benchmarks
    logger.info("\n1. SINGLE DETECTOR BENCHMARKS")
    logger.info("-" * 70)
    for name, detector in detectors.items():
        benchmark.benchmark_single_detector(detector, name, num_frames=100)
    
    logger.info("\n2. MULTI-DETECTOR SEQUENTIAL BENCHMARK")
    logger.info("-" * 70)
    benchmark.benchmark_multi_detector(detectors, num_frames=50)
    
    logger.info("\n3. LOAD TEST - MULTI-CAMERA")
    logger.info("-" * 70)
    load_results = []
    for num_cameras in [5, 10, 20]:
        result = benchmark.benchmark_load_test(detectors, num_cameras=num_cameras, 
                                              frames_per_camera=30)
        load_results.append(result)
    
    logger.info("\n4. MEMORY PROFILING")
    logger.info("-" * 70)
    for name, detector in detectors.items():
        benchmark.benchmark_memory_usage(detector, name)
    
    logger.info("\n5. LATENCY DISTRIBUTION ANALYSIS")
    logger.info("-" * 70)
    for name, detector in detectors.items():
        benchmark.benchmark_latency_distribution(detector, name, num_frames=500)
    
    # Save results
    benchmark.save_results('tests/performance/benchmark_results.json')
    benchmark.print_summary()
    
    logger.info("\n" + "="*70)
    logger.info("BENCHMARK SUITE COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    run_comprehensive_benchmark()
