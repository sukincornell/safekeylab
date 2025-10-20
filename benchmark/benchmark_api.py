#!/usr/bin/env python3
"""
Aegis Privacy Shield - API Benchmark Suite
Tests the production API server for latency and accuracy
"""

import json
import time
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class AegisAPIBenchmark:
    """Benchmark suite for Aegis API"""

    def __init__(self, api_url: str = "http://localhost:8000/v1/protect"):
        self.api_url = api_url
        self.headers = {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-benchmark-key'
        }
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'api_endpoint': api_url,
            'latency': {},
            'accuracy': {},
            'load_test': {}
        }

    def test_api(self, text: str) -> Dict:
        """Send request to API and measure response"""
        start = time.perf_counter()
        try:
            response = requests.post(
                self.api_url,
                json={'text': text},
                headers=self.headers,
                timeout=10
            )
            latency = (time.perf_counter() - start) * 1000  # ms

            if response.status_code == 200:
                return {
                    'success': True,
                    'latency': latency,
                    'data': response.json()
                }
            else:
                return {'success': False, 'latency': latency, 'error': response.text}
        except Exception as e:
            return {'success': False, 'latency': -1, 'error': str(e)}

    def run_latency_tests(self) -> Dict:
        """Run latency benchmark tests"""
        print("\n📊 LATENCY BENCHMARKS")
        print("="*50)

        test_payloads = {
            'small': [
                "Call me at 555-1234",
                "Email: test@example.com",
                "SSN: 123-45-6789"
            ] * 20,
            'medium': [
                "Hello, my name is John Smith and I live at 123 Main St, Springfield, IL 62701. " * 3,
                "Patient record: DOB 01/15/1980, MRN: 12345678, diagnosed with condition XYZ."
            ] * 15,
            'large': [
                "Lorem ipsum " * 200 + " Contact: 555-867-5309 email@test.com SSN: 987-65-4321"
            ] * 10
        }

        results = {}

        for size, payloads in test_payloads.items():
            print(f"\nTesting {size} payloads...")
            latencies = []
            failures = 0

            for payload in payloads:
                result = self.test_api(payload)
                if result['success']:
                    latencies.append(result['latency'])
                else:
                    failures += 1

            if latencies:
                results[size] = {
                    'p50': np.percentile(latencies, 50),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99),
                    'mean': np.mean(latencies),
                    'std': np.std(latencies),
                    'min': min(latencies),
                    'max': max(latencies),
                    'total_requests': len(payloads),
                    'successful': len(latencies),
                    'failed': failures
                }

                print(f"  ✅ P50: {results[size]['p50']:.2f}ms")
                print(f"  ✅ P95: {results[size]['p95']:.2f}ms")
                print(f"  ✅ P99: {results[size]['p99']:.2f}ms")
                print(f"  ✅ Success Rate: {len(latencies)/len(payloads)*100:.1f}%")

        self.results['latency'] = results
        return results

    def run_accuracy_tests(self) -> Dict:
        """Run accuracy tests using test dataset"""
        print("\n🎯 ACCURACY BENCHMARKS")
        print("="*50)

        # Load test dataset
        dataset_path = Path("benchmark/datasets/test_dataset.json")
        if not dataset_path.exists():
            print("❌ Test dataset not found")
            return {}

        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        total_tp, total_fp, total_fn = 0, 0, 0
        category_results = {}

        for test_case in dataset['test_cases'][:10]:  # Test subset for demo
            text = test_case['text']
            expected = test_case['entities']
            category = test_case['category']

            # Test API
            result = self.test_api(text)
            if not result['success']:
                continue

            detected = result['data'].get('entities_found', [])

            # Simple comparison (by type only for demo)
            expected_types = {e['type'] for e in expected}
            detected_types = {e['type'] for e in detected}

            tp = len(expected_types & detected_types)
            fp = len(detected_types - expected_types)
            fn = len(expected_types - detected_types)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            if category not in category_results:
                category_results[category] = {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0}

            category_results[category]['tp'] += tp
            category_results[category]['fp'] += fp
            category_results[category]['fn'] += fn
            category_results[category]['count'] += 1

        # Calculate metrics
        if total_tp + total_fp > 0:
            precision = total_tp / (total_tp + total_fp)
        else:
            precision = 0

        if total_tp + total_fn > 0:
            recall = total_tp / (total_tp + total_fn)
        else:
            recall = 0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        results = {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': total_tp,
                'false_positives': total_fp,
                'false_negatives': total_fn
            },
            'by_category': category_results
        }

        print(f"\n📈 Overall Results:")
        print(f"  ✅ Precision: {precision:.3f}")
        print(f"  ✅ Recall: {recall:.3f}")
        print(f"  ✅ F1 Score: {f1:.3f}")

        self.results['accuracy'] = results
        return results

    def run_load_test(self) -> Dict:
        """Progressive load testing"""
        print("\n🔥 LOAD TESTING")
        print("="*50)

        test_text = "Contact John Smith at john@example.com or 555-123-4567"
        rps_targets = [10, 50, 100]
        results = {}

        for target_rps in rps_targets:
            print(f"\nTesting {target_rps} RPS...")
            latencies = []
            start_time = time.time()

            # Run for 5 seconds
            while time.time() - start_time < 5:
                batch_start = time.time()

                # Try to maintain target RPS
                for _ in range(target_rps // 10):  # Batch requests
                    result = self.test_api(test_text)
                    if result['success']:
                        latencies.append(result['latency'])

                # Sleep to maintain rate
                elapsed = time.time() - batch_start
                if elapsed < 0.1:
                    time.sleep(0.1 - elapsed)

            duration = time.time() - start_time
            achieved_rps = len(latencies) / duration

            if latencies:
                results[f"{target_rps}_rps"] = {
                    'target': target_rps,
                    'achieved': achieved_rps,
                    'p50': np.percentile(latencies, 50),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99),
                    'success_rate': len(latencies) / max(1, target_rps * 5)
                }

                print(f"  ✅ Achieved: {achieved_rps:.1f} RPS")
                print(f"  ✅ P95 Latency: {results[f'{target_rps}_rps']['p95']:.2f}ms")

        self.results['load_test'] = results
        return results

    def generate_report(self):
        """Generate benchmark report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_path = Path(f"benchmark/reports/api_benchmark_{timestamp}.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Generate Markdown report
        md_path = Path(f"benchmark/reports/api_benchmark_{timestamp}.md")

        report = [
            "# Aegis Privacy Shield - API Benchmark Report\n\n",
            f"**Generated:** {self.results['timestamp']}\n",
            f"**Endpoint:** {self.results['api_endpoint']}\n\n"
        ]

        # Latency section
        if self.results.get('latency'):
            report.append("## 📊 Latency Performance\n\n")
            report.append("| Payload | P50 (ms) | P95 (ms) | P99 (ms) | Success Rate |\n")
            report.append("|---------|----------|----------|----------|-------------|\n")

            for size, metrics in self.results['latency'].items():
                success_rate = metrics['successful'] / metrics['total_requests'] * 100
                report.append(f"| {size.capitalize()} | {metrics['p50']:.2f} | "
                            f"{metrics['p95']:.2f} | {metrics['p99']:.2f} | "
                            f"{success_rate:.1f}% |\n")

        # Accuracy section
        if self.results.get('accuracy'):
            overall = self.results['accuracy']['overall']
            report.append("\n## 🎯 Detection Accuracy\n\n")
            report.append(f"- **Precision:** {overall['precision']:.3f}\n")
            report.append(f"- **Recall:** {overall['recall']:.3f}\n")
            report.append(f"- **F1 Score:** {overall['f1_score']:.3f}\n")

        # Load test section
        if self.results.get('load_test'):
            report.append("\n## 🔥 Load Test Results\n\n")
            report.append("| Target RPS | Achieved RPS | P50 (ms) | P95 (ms) |\n")
            report.append("|------------|--------------|----------|----------|\n")

            for key, metrics in self.results['load_test'].items():
                report.append(f"| {metrics['target']} | {metrics['achieved']:.1f} | "
                            f"{metrics['p50']:.2f} | {metrics['p95']:.2f} |\n")

        # Performance grade
        report.append("\n## 📋 Performance Summary\n\n")

        if self.results.get('latency'):
            avg_p95 = np.mean([m['p95'] for m in self.results['latency'].values()])
            if avg_p95 < 10:
                grade = "Excellent ✅"
            elif avg_p95 < 50:
                grade = "Good ✅"
            elif avg_p95 < 100:
                grade = "Acceptable ⚠️"
            else:
                grade = "Needs Improvement ❌"

            report.append(f"- **Latency Grade:** {grade} (Avg P95: {avg_p95:.2f}ms)\n")

        if self.results.get('accuracy'):
            f1 = self.results['accuracy']['overall']['f1_score']
            if f1 > 0.95:
                grade = "Excellent ✅"
            elif f1 > 0.90:
                grade = "Good ✅"
            elif f1 > 0.85:
                grade = "Acceptable ⚠️"
            else:
                grade = "Needs Improvement ❌"

            report.append(f"- **Accuracy Grade:** {grade} (F1: {f1:.3f})\n")

        # Write report
        with open(md_path, 'w') as f:
            f.writelines(report)

        print(f"\n📄 Report saved to {md_path}")
        print(f"📊 JSON results saved to {json_path}")

        return str(md_path)

    def run_full_benchmark(self):
        """Execute complete benchmark suite"""
        print("\n" + "="*60)
        print("🚀 AEGIS PRIVACY SHIELD - API BENCHMARK SUITE")
        print("="*60)

        # Check if API is running
        print("\n🔍 Checking API availability...")
        test_result = self.test_api("test")

        if not test_result['success']:
            print(f"❌ API not available: {test_result.get('error', 'Unknown error')}")
            print("\n💡 Please ensure the API server is running:")
            print("   cd /Users/sukinyang/aegis")
            print("   source venv/bin/activate")
            print("   python app.py")
            return None

        print("✅ API is available")

        # Run benchmarks
        self.run_latency_tests()
        self.run_accuracy_tests()
        self.run_load_test()

        # Generate report
        report_path = self.generate_report()

        print("\n" + "="*60)
        print("✅ BENCHMARK COMPLETE")
        print("="*60)

        return self.results


if __name__ == "__main__":
    benchmark = AegisAPIBenchmark()
    results = benchmark.run_full_benchmark()

    if results:
        print("\n📊 FINAL SUMMARY:")
        if results.get('latency'):
            avg_p95 = np.mean([m['p95'] for m in results['latency'].values()])
            print(f"  Average P95 Latency: {avg_p95:.2f}ms")

        if results.get('accuracy'):
            print(f"  F1 Score: {results['accuracy']['overall']['f1_score']:.3f}")