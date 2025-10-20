#!/usr/bin/env python3
"""
Aegis Privacy Shield - Complete Benchmark Runner
Executes both latency and accuracy benchmarks and generates comprehensive reports
"""

import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List

# Import our benchmark modules
sys.path.append(str(Path(__file__).parent.parent))
from aegis import AegisShield as PrivacyShield


class AegisBenchmarkSuite:
    """Complete benchmark suite for Aegis Privacy Shield"""

    def __init__(self):
        self.shield = PrivacyShield()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'system': 'Aegis Privacy Shield',
            'latency': {},
            'accuracy': {},
            'utility': {}
        }

    def run_latency_benchmarks(self) -> Dict:
        """Execute latency performance tests"""
        print("\n📊 LATENCY BENCHMARKS")
        print("="*50)

        test_payloads = {
            'small': ["Call me at 555-1234"] * 100,
            'medium': ["Hello, my name is John Smith and I live at 123 Main St, Springfield, IL 62701. " * 5] * 50,
            'large': ["Lorem ipsum " * 500 + " Contact: 555-867-5309 email@test.com SSN: 987-65-4321"] * 20
        }

        latency_results = {}

        for size, payloads in test_payloads.items():
            print(f"\nTesting {size} payloads ({len(payloads[0])} chars)...")
            latencies = []
            start_batch = time.time()

            for payload in payloads:
                start = time.perf_counter()
                try:
                    _ = self.shield.protect(payload)
                    latency = (time.perf_counter() - start) * 1000  # ms
                    latencies.append(latency)
                except Exception as e:
                    print(f"Error: {e}")

            batch_time = time.time() - start_batch

            if latencies:
                latency_results[size] = {
                    'p50': np.percentile(latencies, 50),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99),
                    'mean': np.mean(latencies),
                    'std': np.std(latencies),
                    'min': min(latencies),
                    'max': max(latencies),
                    'throughput_rps': len(latencies) / batch_time,
                    'total_requests': len(payloads),
                    'successful': len(latencies)
                }

                print(f"  ✅ P50: {latency_results[size]['p50']:.2f}ms")
                print(f"  ✅ P95: {latency_results[size]['p95']:.2f}ms")
                print(f"  ✅ P99: {latency_results[size]['p99']:.2f}ms")
                print(f"  ✅ Throughput: {latency_results[size]['throughput_rps']:.1f} RPS")

        self.results['latency'] = latency_results
        return latency_results

    def run_accuracy_benchmarks(self) -> Dict:
        """Execute accuracy tests using test dataset"""
        print("\n🎯 ACCURACY BENCHMARKS")
        print("="*50)

        # Load test dataset
        dataset_path = Path("benchmark/datasets/test_dataset.json")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        accuracy_results = {
            'by_category': {},
            'by_entity_type': {},
            'overall': {}
        }

        # Track metrics by category
        category_metrics = {}
        entity_metrics = {}

        total_tp, total_fp, total_fn = 0, 0, 0

        for test_case in dataset['test_cases']:
            text = test_case['text']
            expected = test_case['entities']
            category = test_case['category']

            # Run detection
            result = self.shield.protect(text)
            detected = result.get('entities_found', [])

            # Compare results
            expected_set = {(e['type'], e['value']) for e in expected}
            detected_set = {(e['type'], e['value']) for e in detected}

            tp = len(expected_set & detected_set)
            fp = len(detected_set - expected_set)
            fn = len(expected_set - detected_set)

            # Update totals
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Update category metrics
            if category not in category_metrics:
                category_metrics[category] = {'tp': 0, 'fp': 0, 'fn': 0}
            category_metrics[category]['tp'] += tp
            category_metrics[category]['fp'] += fp
            category_metrics[category]['fn'] += fn

            # Update entity type metrics
            for entity in expected:
                etype = entity['type']
                if etype not in entity_metrics:
                    entity_metrics[etype] = {'tp': 0, 'fp': 0, 'fn': 0}

                if (etype, entity['value']) in detected_set:
                    entity_metrics[etype]['tp'] += 1
                else:
                    entity_metrics[etype]['fn'] += 1

        # Calculate overall metrics
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

        accuracy_results['overall'] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        }

        print(f"\n📈 Overall Results:")
        print(f"  ✅ Precision: {precision:.3f}")
        print(f"  ✅ Recall: {recall:.3f}")
        print(f"  ✅ F1 Score: {f1:.3f}")

        # Calculate metrics by category
        print(f"\n📊 Results by Category:")
        for category, metrics in category_metrics.items():
            tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
            if tp + fp > 0:
                prec = tp / (tp + fp)
            else:
                prec = 0
            if tp + fn > 0:
                rec = tp / (tp + fn)
            else:
                rec = 0

            accuracy_results['by_category'][category] = {
                'precision': prec,
                'recall': rec,
                'samples': sum(1 for tc in dataset['test_cases'] if tc['category'] == category)
            }
            print(f"  {category}: P={prec:.3f}, R={rec:.3f}")

        # Calculate metrics by entity type
        print(f"\n🏷️ Results by Entity Type:")
        for etype, metrics in entity_metrics.items():
            tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
            if tp + fp > 0:
                prec = tp / (tp + fp)
            else:
                prec = 0
            if tp + fn > 0:
                rec = tp / (tp + fn)
            else:
                rec = 0

            # Risk assessment
            if fn == 0:
                risk = "None"
            elif fn / max(1, tp + fn) < 0.02:
                risk = "Low"
            elif fn / max(1, tp + fn) < 0.05:
                risk = "Medium"
            elif fn / max(1, tp + fn) < 0.10:
                risk = "High"
            else:
                risk = "CRITICAL"

            accuracy_results['by_entity_type'][etype] = {
                'precision': prec,
                'recall': rec,
                'false_negative_risk': risk
            }

            if tp + fn > 0:  # Only show entity types that were in test set
                print(f"  {etype}: P={prec:.3f}, R={rec:.3f}, FN Risk={risk}")

        self.results['accuracy'] = accuracy_results
        return accuracy_results

    def run_utility_benchmarks(self) -> Dict:
        """Measure text utility after redaction"""
        print("\n📝 UTILITY BENCHMARKS")
        print("="*50)

        test_texts = [
            "The quick brown fox jumps over the lazy dog. Contact: admin@test.com",
            "John Smith visited Dr. Mary Johnson at 123 Main St on January 15, 2024.",
            "Please send payment of $1,500 to account 1234567890 routing 021000021"
        ]

        utility_results = []

        for text in test_texts:
            result = self.shield.protect(text)
            redacted = result.get('sanitized_text', text)

            original_words = text.split()
            redacted_words = redacted.split()

            # Count preserved content
            preserved = sum(1 for w in redacted_words if '[' not in w)

            utility = {
                'original_length': len(text),
                'redacted_length': len(redacted),
                'words_preserved_pct': (preserved / max(1, len(original_words))) * 100,
                'char_preserved_pct': (len(redacted.replace('[REDACTED]', '')) / max(1, len(text))) * 100
            }
            utility_results.append(utility)

        avg_word_preservation = np.mean([u['words_preserved_pct'] for u in utility_results])
        avg_char_preservation = np.mean([u['char_preserved_pct'] for u in utility_results])

        print(f"  ✅ Average word preservation: {avg_word_preservation:.1f}%")
        print(f"  ✅ Average character preservation: {avg_char_preservation:.1f}%")

        self.results['utility'] = {
            'avg_word_preservation': avg_word_preservation,
            'avg_char_preservation': avg_char_preservation,
            'samples': utility_results
        }

        return self.results['utility']

    def generate_report(self):
        """Generate comprehensive benchmark report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_path = Path(f"benchmark/reports/benchmark_results_{timestamp}.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Generate Markdown report
        md_path = Path(f"benchmark/reports/benchmark_report_{timestamp}.md")

        report = [
            "# Aegis Privacy Shield - Benchmark Report\n\n",
            f"**Generated:** {self.results['timestamp']}\n",
            f"**Version:** {self.results['version']}\n\n",
            "## Executive Summary\n\n",
            "Comprehensive performance and accuracy benchmarks for PII detection and sanitization.\n\n"
        ]

        # Latency section
        report.append("## 📊 Latency Performance\n\n")
        report.append("| Payload Size | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (RPS) |\n")
        report.append("|-------------|----------|----------|----------|------------------|\n")

        for size, metrics in self.results['latency'].items():
            report.append(f"| {size.capitalize()} | {metrics['p50']:.2f} | {metrics['p95']:.2f} | "
                         f"{metrics['p99']:.2f} | {metrics['throughput_rps']:.1f} |\n")

        # Accuracy section
        report.append("\n## 🎯 Detection Accuracy\n\n")
        report.append("### Overall Metrics\n\n")
        overall = self.results['accuracy']['overall']
        report.append(f"- **Precision:** {overall['precision']:.3f}\n")
        report.append(f"- **Recall:** {overall['recall']:.3f}\n")
        report.append(f"- **F1 Score:** {overall['f1_score']:.3f}\n\n")

        # Entity type accuracy
        report.append("### Accuracy by Entity Type\n\n")
        report.append("| Entity Type | Precision | Recall | FN Risk |\n")
        report.append("|------------|-----------|--------|----------|\n")

        for etype, metrics in self.results['accuracy']['by_entity_type'].items():
            report.append(f"| {etype} | {metrics['precision']:.3f} | "
                         f"{metrics['recall']:.3f} | {metrics['false_negative_risk']} |\n")

        # Utility section
        report.append("\n## 📝 Text Utility After Redaction\n\n")
        utility = self.results['utility']
        report.append(f"- **Average Word Preservation:** {utility['avg_word_preservation']:.1f}%\n")
        report.append(f"- **Average Character Preservation:** {utility['avg_char_preservation']:.1f}%\n\n")

        # Conclusions
        report.append("## Conclusions\n\n")

        # Determine performance grade
        p95_latency = np.mean([m['p95'] for m in self.results['latency'].values()])
        if p95_latency < 5:
            latency_grade = "Excellent"
        elif p95_latency < 10:
            latency_grade = "Good"
        elif p95_latency < 20:
            latency_grade = "Acceptable"
        else:
            latency_grade = "Needs Improvement"

        f1_score = overall['f1_score']
        if f1_score > 0.95:
            accuracy_grade = "Excellent"
        elif f1_score > 0.90:
            accuracy_grade = "Good"
        elif f1_score > 0.85:
            accuracy_grade = "Acceptable"
        else:
            accuracy_grade = "Needs Improvement"

        report.append(f"- **Latency Performance:** {latency_grade} (P95 avg: {p95_latency:.2f}ms)\n")
        report.append(f"- **Detection Accuracy:** {accuracy_grade} (F1: {f1_score:.3f})\n")
        report.append(f"- **Ready for Production:** {'Yes ✅' if latency_grade in ['Excellent', 'Good'] and accuracy_grade in ['Excellent', 'Good'] else 'No ❌'}\n")

        # Write report
        with open(md_path, 'w') as f:
            f.writelines(report)

        print(f"\n📄 Report saved to {md_path}")
        print(f"📊 JSON results saved to {json_path}")

        return str(md_path)

    def run_full_suite(self):
        """Execute complete benchmark suite"""
        print("\n" + "="*60)
        print("🚀 AEGIS PRIVACY SHIELD - ENTERPRISE BENCHMARK SUITE")
        print("="*60)

        # Run all benchmarks
        self.run_latency_benchmarks()
        self.run_accuracy_benchmarks()
        self.run_utility_benchmarks()

        # Generate report
        report_path = self.generate_report()

        print("\n" + "="*60)
        print("✅ BENCHMARK COMPLETE")
        print("="*60)

        # Print summary
        print("\n📊 SUMMARY:")
        print(f"  Latency P95 (avg): {np.mean([m['p95'] for m in self.results['latency'].values()]):.2f}ms")
        print(f"  Accuracy F1: {self.results['accuracy']['overall']['f1_score']:.3f}")
        print(f"  Text Utility: {self.results['utility']['avg_word_preservation']:.1f}%")

        return self.results


if __name__ == "__main__":
    suite = AegisBenchmarkSuite()
    results = suite.run_full_suite()