"""
Performance Benchmark Reporter

Generates comprehensive performance reports with:
- Benchmark result analysis
- Visualization-ready data
- Comparison across detectors
- Performance regression detection
- HTML report generation
"""

import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import statistics
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai.logging import setup_logger

logger = setup_logger('performance.reporter')


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics"""
    detector_name: str
    total_tests: int
    avg_fps: float
    min_fps: float
    max_fps: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_memory_mb: float
    peak_memory_mb: float
    cpu_percent: float


class BenchmarkReporter:
    """Generate performance reports"""
    
    def __init__(self, results_file: str = "tests/performance/benchmark_results.json"):
        """Initialize reporter"""
        self.results_file = Path(results_file)
        self.results = self._load_results()
        logger.info(f"Loaded {len(self.results)} benchmark results")
    
    def _load_results(self) -> List[Dict]:
        """Load benchmark results from file"""
        if not self.results_file.exists():
            logger.warning(f"Results file not found: {self.results_file}")
            return []
        
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def get_detector_metrics(self, detector_name: str) -> PerformanceMetrics:
        """Get aggregated metrics for detector"""
        detector_results = [r for r in self.results if r.get('detector_name') == detector_name]
        
        if not detector_results:
            logger.warning(f"No results for detector: {detector_name}")
            return None
        
        fps_values = [r['fps'] for r in detector_results]
        latency_values = [r['avg_latency'] * 1000 for r in detector_results]  # Convert to ms
        memory_values = [r['memory_used'] for r in detector_results]
        
        return PerformanceMetrics(
            detector_name=detector_name,
            total_tests=len(detector_results),
            avg_fps=statistics.mean(fps_values),
            min_fps=min(fps_values),
            max_fps=max(fps_values),
            avg_latency_ms=statistics.mean(latency_values),
            p95_latency_ms=statistics.quantiles(latency_values, n=20)[18] if len(latency_values) > 1 else latency_values[0],
            p99_latency_ms=statistics.quantiles(latency_values, n=100)[98] if len(latency_values) > 1 else latency_values[0],
            avg_memory_mb=statistics.mean(memory_values),
            peak_memory_mb=max([r['peak_memory'] for r in detector_results]),
            cpu_percent=statistics.mean([r['cpu_percent'] for r in detector_results])
        )
    
    def generate_text_report(self, output_file: str = "tests/performance/BENCHMARK_REPORT.txt"):
        """Generate text report"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SENTINELAI PERFORMANCE BENCHMARK REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Results File: {self.results_file}\n")
            f.write(f"Total Benchmarks: {len(self.results)}\n\n")
            
            # Get unique detectors
            detectors = sorted(set(r.get('detector_name') for r in self.results if r.get('detector_name')))
            
            for detector in detectors:
                metrics = self.get_detector_metrics(detector)
                if metrics:
                    f.write(f"\n{detector.upper()} DETECTOR\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Tests Run:           {metrics.total_tests}\n")
                    f.write(f"FPS (Avg/Min/Max):   {metrics.avg_fps:.2f} / {metrics.min_fps:.2f} / {metrics.max_fps:.2f}\n")
                    f.write(f"Latency (Avg):       {metrics.avg_latency_ms:.2f}ms\n")
                    f.write(f"Latency (P95):       {metrics.p95_latency_ms:.2f}ms\n")
                    f.write(f"Latency (P99):       {metrics.p99_latency_ms:.2f}ms\n")
                    f.write(f"Memory (Avg):        {metrics.avg_memory_mb:.2f}MB\n")
                    f.write(f"Memory (Peak):       {metrics.peak_memory_mb:.2f}MB\n")
                    f.write(f"CPU Usage:           {metrics.cpu_percent:.1f}%\n")
            
            # Performance requirements
            f.write(f"\n{'PERFORMANCE REQUIREMENTS VALIDATION':-^80}\n")
            f.write("-" * 80 + "\n")
            
            requirements = {
                'Person Detection': {'min_fps': 60, 'max_latency_ms': 100},
                'Vehicle Detection': {'min_fps': 30, 'max_latency_ms': 150},
                'Animal Detection': {'min_fps': 40, 'max_latency_ms': 120},
                'Object Detection': {'min_fps': 35, 'max_latency_ms': 140}
            }
            
            for detector, req in requirements.items():
                metrics = self.get_detector_metrics(detector)
                if metrics:
                    fps_ok = "✓" if metrics.avg_fps >= req['min_fps'] else "✗"
                    latency_ok = "✓" if metrics.avg_latency_ms <= req['max_latency_ms'] else "✗"
                    
                    f.write(f"\n{detector}:\n")
                    f.write(f"  FPS ({req['min_fps']}+ required): {metrics.avg_fps:.2f} {fps_ok}\n")
                    f.write(f"  Latency ({req['max_latency_ms']}ms max): {metrics.avg_latency_ms:.2f}ms {latency_ok}\n")
        
        logger.info(f"Text report saved to {output_path}")
        return output_path
    
    def generate_html_report(self, output_file: str = "tests/performance/BENCHMARK_REPORT.html"):
        """Generate HTML report"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        detectors = sorted(set(r.get('detector_name') for r in self.results if r.get('detector_name')))
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SentinelAI Performance Benchmark Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #007bff;
            color: white;
            padding: 12px;
            text-align: left;
            border: 1px solid #0056b3;
        }}
        td {{
            padding: 12px;
            border: 1px solid #ddd;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f0f0f0;
        }}
        .pass {{
            color: green;
            font-weight: bold;
        }}
        .fail {{
            color: red;
            font-weight: bold;
        }}
        .metric-box {{
            background-color: #f9f9f9;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 10px 0;
        }}
        .requirement {{
            margin: 15px 0;
            padding: 10px;
            border: 1px solid #ddd;
            background-color: white;
        }}
    </style>
</head>
<body>
    <h1>🛡️ SentinelAI Performance Benchmark Report</h1>
    <p>Generated: {datetime.now().isoformat()}</p>
    <p>Total Benchmarks: {len(self.results)}</p>
    
    <h2>Detector Performance Summary</h2>
    <table>
        <tr>
            <th>Detector</th>
            <th>Tests</th>
            <th>Avg FPS</th>
            <th>Min/Max FPS</th>
            <th>Avg Latency (ms)</th>
            <th>P95 Latency (ms)</th>
            <th>Memory (MB)</th>
            <th>CPU %</th>
        </tr>
"""
        
        for detector in detectors:
            metrics = self.get_detector_metrics(detector)
            if metrics:
                html += f"""        <tr>
            <td><strong>{detector}</strong></td>
            <td>{metrics.total_tests}</td>
            <td>{metrics.avg_fps:.2f}</td>
            <td>{metrics.min_fps:.2f} / {metrics.max_fps:.2f}</td>
            <td>{metrics.avg_latency_ms:.2f}</td>
            <td>{metrics.p95_latency_ms:.2f}</td>
            <td>{metrics.avg_memory_mb:.2f}</td>
            <td>{metrics.cpu_percent:.1f}%</td>
        </tr>
"""
        
        html += """    </table>
    
    <h2>Performance Requirements Validation</h2>
"""
        
        requirements = {
            'Person': {'min_fps': 60, 'max_latency_ms': 100},
            'Vehicle': {'min_fps': 30, 'max_latency_ms': 150},
            'Animal': {'min_fps': 40, 'max_latency_ms': 120},
            'Object': {'min_fps': 35, 'max_latency_ms': 140}
        }
        
        for detector, req in requirements.items():
            metrics = self.get_detector_metrics(detector)
            if metrics:
                fps_ok = metrics.avg_fps >= req['min_fps']
                latency_ok = metrics.avg_latency_ms <= req['max_latency_ms']
                
                fps_class = "pass" if fps_ok else "fail"
                latency_class = "pass" if latency_ok else "fail"
                
                html += f"""    <div class="requirement">
        <h3>{detector} Detector</h3>
        <div class="metric-box">
            <strong>FPS Requirement:</strong> {req['min_fps']}+ FPS<br>
            Actual: <span class="{fps_class}">{metrics.avg_fps:.2f} FPS</span>
        </div>
        <div class="metric-box">
            <strong>Latency Requirement:</strong> {req['max_latency_ms']}ms max<br>
            Actual: <span class="{latency_class}">{metrics.avg_latency_ms:.2f}ms</span>
        </div>
    </div>
"""
        
        html += """    <h2>Benchmark Details</h2>
    <table>
        <tr>
            <th>Detector</th>
            <th>Test</th>
            <th>Frames</th>
            <th>FPS</th>
            <th>Avg Latency (ms)</th>
            <th>Memory (MB)</th>
        </tr>
"""
        
        for result in sorted(self.results, key=lambda x: (x.get('detector_name', ''), x.get('test_name', ''))):
            html += f"""        <tr>
            <td>{result.get('detector_name', 'N/A')}</td>
            <td>{result.get('test_name', 'N/A')}</td>
            <td>{result.get('frames_processed', 'N/A')}</td>
            <td>{result.get('fps', 0):.2f}</td>
            <td>{result.get('avg_latency', 0)*1000:.2f}</td>
            <td>{result.get('memory_used', 0):.2f}</td>
        </tr>
"""
        
        html += """    </table>
    
    <footer style="margin-top: 40px; color: #666; text-align: center;">
        <p>SentinelAI Defense Surveillance Intelligence System</p>
        <p>Performance testing suite - Day 12</p>
    </footer>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"HTML report saved to {output_path}")
        return output_path
    
    def check_performance_regression(self, baseline_file: str) -> Dict:
        """Check for performance regression vs baseline"""
        baseline_path = Path(baseline_file)
        
        if not baseline_path.exists():
            logger.warning(f"Baseline file not found: {baseline_path}")
            return {}
        
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)
        
        regressions = []
        improvements = []
        
        # Compare FPS
        for detector in set(r.get('detector_name') for r in baseline_results):
            baseline_fps = statistics.mean([
                r['fps'] for r in baseline_results 
                if r.get('detector_name') == detector
            ])
            
            current_fps = statistics.mean([
                r['fps'] for r in self.results 
                if r.get('detector_name') == detector
            ])
            
            change_percent = ((current_fps - baseline_fps) / baseline_fps) * 100
            
            if change_percent < -5:  # More than 5% regression
                regressions.append({
                    'detector': detector,
                    'metric': 'FPS',
                    'baseline': baseline_fps,
                    'current': current_fps,
                    'change_percent': change_percent
                })
            elif change_percent > 5:
                improvements.append({
                    'detector': detector,
                    'metric': 'FPS',
                    'baseline': baseline_fps,
                    'current': current_fps,
                    'change_percent': change_percent
                })
        
        return {
            'regressions': regressions,
            'improvements': improvements,
            'status': 'PASS' if not regressions else 'FAIL'
        }


def main():
    """Generate reports"""
    reporter = BenchmarkReporter()
    
    # Generate text report
    reporter.generate_text_report()
    
    # Generate HTML report
    reporter.generate_html_report()
    
    logger.info("\nReports generated successfully")


if __name__ == '__main__':
    main()
