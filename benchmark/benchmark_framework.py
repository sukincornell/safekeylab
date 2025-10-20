#!/usr/bin/env python3
"""
Aegis Privacy Shield - Enterprise Benchmark Framework
Comprehensive latency and accuracy benchmarking for PII sanitization
"""

import time
import json
import statistics
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pathlib import Path


@dataclass
class LatencyMetrics:
    """Latency performance metrics"""
    p50: float
    p95: float
    p99: float
    mean: float
    std_dev: float
    min_latency: float
    max_latency: float
    throughput_rps: float
    total_requests: int
    failed_requests: int
    timestamp: str = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class AccuracyMetrics:
    """PII detection accuracy metrics"""
    entity_type: str
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    false_negative_risk: str = "Unknown"

    def calculate_metrics(self):
        """Calculate precision, recall, and F1 score"""
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)

        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)

        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)

        # Risk assessment based on false negative rate
        fn_rate = self.false_negatives / max(1, self.false_negatives + self.true_positives)
        if fn_rate == 0:
            self.false_negative_risk = "None"
        elif fn_rate < 0.01:
            self.false_negative_risk = "Low"
        elif fn_rate < 0.05:
            self.false_negative_risk = "Medium"
        elif fn_rate < 0.1:
            self.false_negative_risk = "High"
        else:
            self.false_negative_risk = "CRITICAL"


class LatencyBenchmark:
    """Latency benchmarking for API endpoints"""

    def __init__(self, api_endpoint: str, api_key: str = None):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'X-API-Key': api_key or 'test-key'
        }

    def measure_single_request(self, payload: str) -> float:
        """Measure latency for a single request"""
        import requests

        start = time.perf_counter()
        try:
            response = requests.post(
                self.api_endpoint,
                json={'text': payload},
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            latency = (time.perf_counter() - start) * 1000  # ms
            return latency
        except Exception as e:
            print(f"Request failed: {e}")
            return -1

    async def measure_async_request(self, session: aiohttp.ClientSession, payload: str) -> float:
        """Async request measurement for high throughput testing"""
        start = time.perf_counter()
        try:
            async with session.post(
                self.api_endpoint,
                json={'text': payload},
                headers=self.headers
            ) as response:
                await response.text()
                latency = (time.perf_counter() - start) * 1000
                return latency
        except:
            return -1

    def run_latency_test(self, payloads: List[str], concurrent: int = 10) -> LatencyMetrics:
        """Run comprehensive latency benchmark"""
        latencies = []
        failed = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = [
                executor.submit(self.measure_single_request, payload)
                for payload in payloads
            ]

            for future in as_completed(futures):
                latency = future.result()
                if latency < 0:
                    failed += 1
                else:
                    latencies.append(latency)

        duration = time.time() - start_time

        if not latencies:
            return LatencyMetrics(
                p50=0, p95=0, p99=0, mean=0, std_dev=0,
                min_latency=0, max_latency=0, throughput_rps=0,
                total_requests=len(payloads), failed_requests=failed
            )

        return LatencyMetrics(
            p50=np.percentile(latencies, 50),
            p95=np.percentile(latencies, 95),
            p99=np.percentile(latencies, 99),
            mean=statistics.mean(latencies),
            std_dev=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            min_latency=min(latencies),
            max_latency=max(latencies),
            throughput_rps=len(latencies) / duration,
            total_requests=len(payloads),
            failed_requests=failed
        )

    async def run_async_load_test(self, payloads: List[str], rps_targets: List[int]) -> Dict:
        """Progressive load testing with increasing RPS"""
        results = {}

        for target_rps in rps_targets:
            print(f"Testing at {target_rps} RPS...")
            latencies = []

            async with aiohttp.ClientSession() as session:
                tasks = []
                for i in range(target_rps):
                    payload = payloads[i % len(payloads)]
                    tasks.append(self.measure_async_request(session, payload))

                # Execute within 1 second window
                start = time.time()
                responses = await asyncio.gather(*tasks)
                duration = time.time() - start

                latencies = [r for r in responses if r > 0]

                if latencies:
                    results[f"{target_rps}_rps"] = {
                        'p50': np.percentile(latencies, 50),
                        'p95': np.percentile(latencies, 95),
                        'p99': np.percentile(latencies, 99),
                        'achieved_rps': len(latencies) / duration,
                        'success_rate': len(latencies) / len(responses)
                    }

        return results


class AccuracyBenchmark:
    """Accuracy benchmarking for PII detection"""

    def __init__(self, sanitizer_func):
        self.sanitizer_func = sanitizer_func
        self.entity_types = ['email', 'phone', 'ssn', 'name', 'credit_card', 'address', 'medical_record']

    def evaluate_detection(self, test_cases: List[Dict]) -> List[AccuracyMetrics]:
        """Evaluate detection accuracy across test cases"""
        metrics_by_type = {entity: {
            'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
        } for entity in self.entity_types}

        for case in test_cases:
            text = case['text']
            expected = set(case['entities'])

            # Run detection
            detected = set(self.sanitizer_func(text))

            for entity_type in self.entity_types:
                expected_type = {e for e in expected if e['type'] == entity_type}
                detected_type = {e for e in detected if e['type'] == entity_type}

                # Calculate confusion matrix
                tp = len(expected_type & detected_type)
                fp = len(detected_type - expected_type)
                fn = len(expected_type - detected_type)

                metrics_by_type[entity_type]['tp'] += tp
                metrics_by_type[entity_type]['fp'] += fp
                metrics_by_type[entity_type]['fn'] += fn

        # Create metrics objects
        results = []
        for entity_type, counts in metrics_by_type.items():
            metric = AccuracyMetrics(
                entity_type=entity_type,
                true_positives=counts['tp'],
                false_positives=counts['fp'],
                false_negatives=counts['fn'],
                true_negatives=counts['tn']
            )
            metric.calculate_metrics()
            results.append(metric)

        return results

    def calculate_utility_score(self, original: str, redacted: str) -> Dict:
        """Calculate how much useful content remains after redaction"""
        original_words = original.split()
        redacted_words = redacted.split()

        # Count non-redacted words
        preserved = sum(1 for w in redacted_words if not w.startswith('[') or not w.endswith(']'))

        return {
            'words_preserved_pct': (preserved / max(1, len(original_words))) * 100,
            'char_preserved_pct': (len(redacted) / max(1, len(original))) * 100,
            'readability_score': self._calculate_readability(redacted)
        }

    def _calculate_readability(self, text: str) -> float:
        """Simple readability heuristic"""
        redacted_tokens = text.count('[')
        total_tokens = len(text.split())
        if total_tokens == 0:
            return 0
        return max(0, 1 - (redacted_tokens / total_tokens))


class BenchmarkRunner:
    """Main benchmark orchestrator"""

    def __init__(self, api_endpoint: str = "http://localhost:8000/v1/protect"):
        self.latency_bench = LatencyBenchmark(api_endpoint)
        self.accuracy_bench = None  # Initialized with sanitizer function
        self.results_dir = Path("benchmark/reports")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate_test_payloads(self) -> Dict[str, List[str]]:
        """Generate various test payloads for benchmarking"""
        payloads = {
            'small': [
                "Call me at 555-1234",
                "Email: test@example.com",
                "SSN: 123-45-6789"
            ] * 100,
            'medium': [
                "Hello, my name is John Smith and I live at 123 Main St, Springfield, IL 62701. " * 5,
                "Patient record: DOB 01/15/1980, MRN: 12345678, diagnosed with condition XYZ. " * 5
            ] * 50,
            'large': [
                "Lorem ipsum " * 500 + " Contact: 555-867-5309 email@test.com SSN: 987-65-4321"
            ] * 20,
            'adversarial': [
                "j o h n (at) g mail dot com",
                "five five five - one two three four",
                "Mi número es +52 55 1001 1293",
                "מספר הטלפון שלי הוא 054-1234567"
            ] * 25
        }
        return payloads

    def run_full_benchmark(self) -> Dict:
        """Execute complete benchmark suite"""
        print("🚀 Starting Aegis Privacy Shield Benchmark Suite")
        results = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'latency': {},
            'accuracy': {},
            'load_test': {}
        }

        # 1. Latency benchmarks
        print("\n📊 Running latency benchmarks...")
        payloads = self.generate_test_payloads()

        for payload_type, payload_list in payloads.items():
            print(f"  Testing {payload_type} payloads...")
            metrics = self.latency_bench.run_latency_test(payload_list, concurrent=10)
            results['latency'][payload_type] = asdict(metrics)

        # 2. Load testing
        print("\n🔥 Running progressive load tests...")
        asyncio.run(self._run_load_test(results, payloads['small']))

        # 3. Save results
        self.save_results(results)
        self.generate_report(results)

        return results

    async def _run_load_test(self, results: Dict, payloads: List[str]):
        """Async load testing wrapper"""
        rps_targets = [100, 500, 1000, 5000]
        load_results = await self.latency_bench.run_async_load_test(payloads, rps_targets)
        results['load_test'] = load_results

    def save_results(self, results: Dict):
        """Save benchmark results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"benchmark_{timestamp}.json"

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to {filepath}")

    def generate_report(self, results: Dict):
        """Generate markdown report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"benchmark_report_{timestamp}.md"

        report = ["# Aegis Privacy Shield - Benchmark Report\n"]
        report.append(f"Generated: {results['timestamp']}\n")

        # Latency section
        report.append("\n## Latency Performance\n")
        report.append("| Payload Type | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (RPS) |\n")
        report.append("|-------------|----------|----------|----------|------------------|\n")

        for payload_type, metrics in results['latency'].items():
            report.append(f"| {payload_type.capitalize()} | "
                         f"{metrics['p50']:.2f} | "
                         f"{metrics['p95']:.2f} | "
                         f"{metrics['p99']:.2f} | "
                         f"{metrics['throughput_rps']:.1f} |\n")

        # Load test section
        if results.get('load_test'):
            report.append("\n## Load Test Results\n")
            report.append("| Target RPS | Achieved RPS | P50 (ms) | P95 (ms) | Success Rate |\n")
            report.append("|------------|--------------|----------|----------|-------------|\n")

            for rps_level, metrics in results['load_test'].items():
                target = rps_level.split('_')[0]
                report.append(f"| {target} | "
                             f"{metrics['achieved_rps']:.1f} | "
                             f"{metrics['p50']:.2f} | "
                             f"{metrics['p95']:.2f} | "
                             f"{metrics['success_rate']:.1%} |\n")

        # Write report
        with open(filepath, 'w') as f:
            f.writelines(report)

        print(f"📄 Report generated at {filepath}")


if __name__ == "__main__":
    runner = BenchmarkRunner()
    results = runner.run_full_benchmark()

    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    for payload_type, metrics in results['latency'].items():
        print(f"\n{payload_type.upper()} Payloads:")
        print(f"  P50: {metrics['p50']:.2f}ms")
        print(f"  P95: {metrics['p95']:.2f}ms")
        print(f"  P99: {metrics['p99']:.2f}ms")
        print(f"  Throughput: {metrics['throughput_rps']:.1f} RPS")