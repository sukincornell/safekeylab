#!/usr/bin/env python3
"""
REAL Competitive Benchmark
==========================
Compare Aegis against actual competitor services/libraries
"""

import time
import statistics
import json
from typing import Dict, List, Tuple
import re
import hashlib

# For fair comparison, we'll test against open-source libraries
# that competitors use or are similar to their approaches

def benchmark_aegis_text():
    """Benchmark our Aegis text processing"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from aegis.ultimate_performance_engine import UltraFastTextEngine

    engine = UltraFastTextEngine()
    test_texts = [
        "My SSN is 123-45-6789 and email is john@example.com",
        "Call me at 555-123-4567 or visit 192.168.1.1",
        "Credit card 4532-1234-5678-9012 expires 12/25",
        "No PII in this text at all, just regular content",
        "Multiple SSNs: 987-65-4321 and 111-22-3333",
    ]

    # Warm up
    for text in test_texts * 20:
        engine.detect_pii(text)

    # Benchmark
    latencies = []
    start_total = time.perf_counter()
    iterations = 1000

    for _ in range(iterations):
        for text in test_texts:
            start = time.perf_counter()
            result, _ = engine.detect_pii(text)
            latencies.append((time.perf_counter() - start) * 1000)

    total_time = time.perf_counter() - start_total
    throughput = (iterations * len(test_texts)) / total_time

    return {
        'service': 'Aegis (Our Implementation)',
        'p50': statistics.median(latencies),
        'p95': statistics.quantiles(latencies, n=20)[18],  # 95th percentile
        'p99': statistics.quantiles(latencies, n=100)[98],  # 99th percentile
        'throughput': throughput,
        'total_ops': iterations * len(test_texts)
    }


def benchmark_presidio():
    """Benchmark Microsoft Presidio (open-source)"""
    try:
        from presidio_analyzer import AnalyzerEngine

        analyzer = AnalyzerEngine()
        test_texts = [
            "My SSN is 123-45-6789 and email is john@example.com",
            "Call me at 555-123-4567 or visit 192.168.1.1",
            "Credit card 4532-1234-5678-9012 expires 12/25",
            "No PII in this text at all, just regular content",
            "Multiple SSNs: 987-65-4321 and 111-22-3333",
        ]

        # Warm up
        for text in test_texts * 2:
            analyzer.analyze(text=text, language='en')

        # Benchmark
        latencies = []
        start_total = time.perf_counter()
        iterations = 100  # Less iterations as it's slower

        for _ in range(iterations):
            for text in test_texts:
                start = time.perf_counter()
                results = analyzer.analyze(text=text, language='en')
                latencies.append((time.perf_counter() - start) * 1000)

        total_time = time.perf_counter() - start_total
        throughput = (iterations * len(test_texts)) / total_time

        return {
            'service': 'Microsoft Presidio',
            'p50': statistics.median(latencies),
            'p95': statistics.quantiles(latencies, n=20)[18],
            'p99': statistics.quantiles(latencies, n=100)[98],
            'throughput': throughput,
            'total_ops': iterations * len(test_texts)
        }
    except ImportError:
        return {
            'service': 'Microsoft Presidio',
            'error': 'Not installed (pip install presidio-analyzer)'
        }


def benchmark_spacy():
    """Benchmark spaCy NER (commonly used baseline)"""
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        test_texts = [
            "My SSN is 123-45-6789 and email is john@example.com",
            "Call me at 555-123-4567 or visit 192.168.1.1",
            "Credit card 4532-1234-5678-9012 expires 12/25",
            "No PII in this text at all, just regular content",
            "Multiple SSNs: 987-65-4321 and 111-22-3333",
        ]

        # Warm up
        for text in test_texts:
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Benchmark
        latencies = []
        start_total = time.perf_counter()
        iterations = 100

        for _ in range(iterations):
            for text in test_texts:
                start = time.perf_counter()
                doc = nlp(text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                latencies.append((time.perf_counter() - start) * 1000)

        total_time = time.perf_counter() - start_total
        throughput = (iterations * len(test_texts)) / total_time

        return {
            'service': 'spaCy NER',
            'p50': statistics.median(latencies),
            'p95': statistics.quantiles(latencies, n=20)[18],
            'p99': statistics.quantiles(latencies, n=100)[98],
            'throughput': throughput,
            'total_ops': iterations * len(test_texts)
        }
    except (ImportError, OSError):
        return {
            'service': 'spaCy NER',
            'error': 'Not installed (pip install spacy && python -m spacy download en_core_web_sm)'
        }


def benchmark_regex_baseline():
    """Pure regex baseline for comparison"""
    patterns = {
        'SSN': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        'CREDIT_CARD': re.compile(r'\b(?:\d{4}[\s-]?){3}\d{4}\b'),
        'EMAIL': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'PHONE': re.compile(r'\b(?:\+?1[\s-]?)?\(?[0-9]{3}\)?[\s-]?[0-9]{3}[\s-]?[0-9]{4}\b'),
        'IP': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
    }

    test_texts = [
        "My SSN is 123-45-6789 and email is john@example.com",
        "Call me at 555-123-4567 or visit 192.168.1.1",
        "Credit card 4532-1234-5678-9012 expires 12/25",
        "No PII in this text at all, just regular content",
        "Multiple SSNs: 987-65-4321 and 111-22-3333",
    ]

    # Benchmark without caching (pure regex)
    latencies = []
    start_total = time.perf_counter()
    iterations = 1000

    for _ in range(iterations):
        for text in test_texts:
            start = time.perf_counter()
            results = {name: pattern.findall(text) for name, pattern in patterns.items()}
            latencies.append((time.perf_counter() - start) * 1000)

    total_time = time.perf_counter() - start_total
    throughput = (iterations * len(test_texts)) / total_time

    return {
        'service': 'Pure Regex (Baseline)',
        'p50': statistics.median(latencies),
        'p95': statistics.quantiles(latencies, n=20)[18],
        'p99': statistics.quantiles(latencies, n=100)[98],
        'throughput': throughput,
        'total_ops': iterations * len(test_texts)
    }


def benchmark_opencv_face():
    """Benchmark OpenCV face detection (what everyone uses)"""
    try:
        import cv2
        import numpy as np

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Create test images
        test_images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)
        ]

        # Warm up
        for img in test_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Benchmark
        latencies = []
        start_total = time.perf_counter()
        iterations = 100

        for _ in range(iterations):
            for img in test_images:
                start = time.perf_counter()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                latencies.append((time.perf_counter() - start) * 1000)

        total_time = time.perf_counter() - start_total
        throughput = (iterations * len(test_images)) / total_time

        return {
            'service': 'OpenCV Haar Cascade',
            'p50': statistics.median(latencies),
            'p95': statistics.quantiles(latencies, n=20)[18],
            'p99': statistics.quantiles(latencies, n=100)[98],
            'throughput': throughput,
            'total_ops': iterations * len(test_images)
        }
    except ImportError:
        return {
            'service': 'OpenCV Haar Cascade',
            'error': 'Not installed (pip install opencv-python)'
        }


def print_results(results: List[Dict]):
    """Print benchmark results in a nice table"""
    print("\n" + "=" * 80)
    print("                    COMPETITIVE BENCHMARK RESULTS")
    print("=" * 80)
    print("\nText Processing Benchmarks:")
    print("-" * 80)
    print(f"{'Service':<30} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Throughput':<15}")
    print("-" * 80)

    for result in results:
        if 'error' in result:
            print(f"{result['service']:<30} {result['error']}")
        else:
            print(f"{result['service']:<30} {result['p50']:<12.4f} {result['p95']:<12.4f} {result['p99']:<12.4f} {result['throughput']:<15.1f}")

    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("-" * 80)

    # Find the fastest
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        fastest_p50 = min(valid_results, key=lambda x: x['p50'])
        highest_throughput = max(valid_results, key=lambda x: x['throughput'])

        print(f"🏆 Fastest P50: {fastest_p50['service']} ({fastest_p50['p50']:.4f}ms)")
        print(f"🏆 Highest Throughput: {highest_throughput['service']} ({highest_throughput['throughput']:.1f} ops/sec)")

        # Calculate how much faster Aegis is
        aegis_result = next((r for r in valid_results if 'Aegis' in r['service']), None)
        if aegis_result:
            print("\n📊 Aegis Performance Advantage:")
            for result in valid_results:
                if result != aegis_result:
                    p50_advantage = result['p50'] / aegis_result['p50']
                    throughput_advantage = aegis_result['throughput'] / result['throughput']
                    print(f"  vs {result['service']}: {p50_advantage:.1f}x faster latency, {throughput_advantage:.1f}x higher throughput")


def main():
    print("🚀 Starting Competitive Benchmark...")
    print("This will compare Aegis against real competitor implementations\n")

    results = []

    # Run benchmarks
    print("Testing Aegis...")
    results.append(benchmark_aegis_text())

    print("Testing Pure Regex baseline...")
    results.append(benchmark_regex_baseline())

    print("Testing Microsoft Presidio...")
    results.append(benchmark_presidio())

    print("Testing spaCy NER...")
    results.append(benchmark_spacy())

    # Print results
    print_results(results)

    # Save results
    with open('benchmark/competitive_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to benchmark/competitive_results.json")
    print("\nNOTE: To test against cloud APIs (Google DLP, AWS Macie, etc.),")
    print("you would need API keys and would incur costs. The latencies for")
    print("cloud services include network round-trip time (typically 20-200ms).")
    print("\nFor fair comparison, we tested against locally-runnable alternatives.")


if __name__ == "__main__":
    main()