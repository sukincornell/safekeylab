#!/usr/bin/env python3
"""
Accuracy benchmarking module for PII detection
Tests precision, recall, F1 scores across multiple entity types
"""

import json
import re
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import pandas as pd
from pathlib import Path


@dataclass
class TestCase:
    """Single test case for accuracy evaluation"""
    text: str
    entities: List[Dict]  # [{'type': 'email', 'value': 'test@example.com', 'start': 10, 'end': 25}]
    category: str  # 'basic', 'obfuscated', 'multilingual', 'adversarial'
    difficulty: str  # 'easy', 'medium', 'hard'


class PIIDetector:
    """Reference PII detector for benchmarking"""

    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?\d{1,4}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'medical_record': r'\b(MRN|Patient ID|Medical Record)[\s:#]*\d{6,10}\b',
            'api_key': r'\b(sk|pk|api)[-_](live|test|prod)[-_][A-Za-z0-9]{20,}\b',
            'aws_key': r'\b(AKIA|ASIA)[A-Z0-9]{16}\b',
            'date_of_birth': r'\b(DOB|Date of Birth|Born)[\s:]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        }

    def detect(self, text: str) -> List[Dict]:
        """Detect PII entities in text"""
        entities = []

        for entity_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'type': entity_type,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })

        # Name detection (simplified)
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        for match in re.finditer(name_pattern, text):
            # Filter out common non-names
            value = match.group()
            if value not in ['United States', 'New York', 'Los Angeles', 'San Francisco']:
                entities.append({
                    'type': 'name',
                    'value': value,
                    'start': match.start(),
                    'end': match.end()
                })

        return entities


class AccuracyEvaluator:
    """Evaluate detection accuracy against ground truth"""

    def __init__(self):
        self.detector = PIIDetector()

    def evaluate_single(self, test_case: TestCase) -> Dict:
        """Evaluate a single test case"""
        detected = self.detector.detect(test_case.text)

        # Convert to sets for comparison
        detected_set = {(e['type'], e['value']) for e in detected}
        expected_set = {(e['type'], e['value']) for e in test_case.entities}

        tp = detected_set & expected_set
        fp = detected_set - expected_set
        fn = expected_set - detected_set

        return {
            'true_positives': len(tp),
            'false_positives': len(fp),
            'false_negatives': len(fn),
            'detected': list(detected_set),
            'expected': list(expected_set),
            'missed': list(fn),
            'extra': list(fp)
        }

    def evaluate_batch(self, test_cases: List[TestCase]) -> pd.DataFrame:
        """Evaluate multiple test cases and return metrics"""
        results = []

        for tc in test_cases:
            eval_result = self.evaluate_single(tc)
            eval_result['category'] = tc.category
            eval_result['difficulty'] = tc.difficulty
            results.append(eval_result)

        df = pd.DataFrame(results)

        # Calculate aggregate metrics
        metrics = {
            'total_tests': len(test_cases),
            'total_tp': df['true_positives'].sum(),
            'total_fp': df['false_positives'].sum(),
            'total_fn': df['false_negatives'].sum()
        }

        # Calculate precision, recall, F1
        if metrics['total_tp'] + metrics['total_fp'] > 0:
            metrics['precision'] = metrics['total_tp'] / (metrics['total_tp'] + metrics['total_fp'])
        else:
            metrics['precision'] = 0

        if metrics['total_tp'] + metrics['total_fn'] > 0:
            metrics['recall'] = metrics['total_tp'] / (metrics['total_tp'] + metrics['total_fn'])
        else:
            metrics['recall'] = 0

        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0

        return df, metrics


class TestDataGenerator:
    """Generate comprehensive test datasets"""

    @staticmethod
    def generate_basic_tests() -> List[TestCase]:
        """Basic PII detection test cases"""
        return [
            TestCase(
                text="Please contact John Smith at john.smith@example.com or 555-123-4567",
                entities=[
                    {'type': 'name', 'value': 'John Smith'},
                    {'type': 'email', 'value': 'john.smith@example.com'},
                    {'type': 'phone', 'value': '555-123-4567'}
                ],
                category='basic',
                difficulty='easy'
            ),
            TestCase(
                text="SSN: 123-45-6789, Credit Card: 4111-1111-1111-1111",
                entities=[
                    {'type': 'ssn', 'value': '123-45-6789'},
                    {'type': 'credit_card', 'value': '4111-1111-1111-1111'}
                ],
                category='basic',
                difficulty='easy'
            ),
            TestCase(
                text="Patient Mary Johnson, DOB: 01/15/1980, MRN: 12345678",
                entities=[
                    {'type': 'name', 'value': 'Mary Johnson'},
                    {'type': 'date_of_birth', 'value': 'DOB: 01/15/1980'},
                    {'type': 'medical_record', 'value': 'MRN: 12345678'}
                ],
                category='basic',
                difficulty='easy'
            )
        ]

    @staticmethod
    def generate_obfuscated_tests() -> List[TestCase]:
        """Obfuscated PII patterns"""
        return [
            TestCase(
                text="Contact: j o h n (at) g mail dot com",
                entities=[],  # Most detectors should miss this
                category='obfuscated',
                difficulty='hard'
            ),
            TestCase(
                text="Phone: five five five - one two three four",
                entities=[],
                category='obfuscated',
                difficulty='hard'
            ),
            TestCase(
                text="Email is john[AT]company[DOT]org",
                entities=[],
                category='obfuscated',
                difficulty='hard'
            )
        ]

    @staticmethod
    def generate_multilingual_tests() -> List[TestCase]:
        """Multilingual PII patterns"""
        return [
            TestCase(
                text="Mi número es +52 55 1234 5678",
                entities=[
                    {'type': 'phone', 'value': '+52 55 1234 5678'}
                ],
                category='multilingual',
                difficulty='medium'
            ),
            TestCase(
                text="我的电子邮件是 zhang.wei@example.cn",
                entities=[
                    {'type': 'email', 'value': 'zhang.wei@example.cn'}
                ],
                category='multilingual',
                difficulty='medium'
            ),
            TestCase(
                text="Téléphone: +33 1 23 45 67 89",
                entities=[
                    {'type': 'phone', 'value': '+33 1 23 45 67 89'}
                ],
                category='multilingual',
                difficulty='medium'
            )
        ]

    @staticmethod
    def generate_adversarial_tests() -> List[TestCase]:
        """Adversarial test cases to stress-test detection"""
        return [
            TestCase(
                text="Bank of America customer service: 1-800-432-1000",
                entities=[
                    {'type': 'phone', 'value': '1-800-432-1000'}
                ],  # "Bank of America" should NOT be detected as a name
                category='adversarial',
                difficulty='hard'
            ),
            TestCase(
                text="IP: 192.168.1.1, API Key: sk-live-abcdef1234567890abcdef1234567890",
                entities=[
                    {'type': 'ip_address', 'value': '192.168.1.1'},
                    {'type': 'api_key', 'value': 'sk-live-abcdef1234567890abcdef1234567890'}
                ],
                category='adversarial',
                difficulty='medium'
            ),
            TestCase(
                text="AWS Access Key: AKIAIOSFODNN7EXAMPLE",
                entities=[
                    {'type': 'aws_key', 'value': 'AKIAIOSFODNN7EXAMPLE'}
                ],
                category='adversarial',
                difficulty='medium'
            )
        ]

    @staticmethod
    def generate_edge_cases() -> List[TestCase]:
        """Edge cases and boundary conditions"""
        return [
            TestCase(
                text="",  # Empty text
                entities=[],
                category='edge',
                difficulty='easy'
            ),
            TestCase(
                text="No PII here, just regular text about technology and science.",
                entities=[],
                category='edge',
                difficulty='easy'
            ),
            TestCase(
                text="a" * 10000 + " email@test.com " + "b" * 10000,  # Long text
                entities=[
                    {'type': 'email', 'value': 'email@test.com'}
                ],
                category='edge',
                difficulty='medium'
            )
        ]


def run_accuracy_benchmark():
    """Main function to run accuracy benchmarks"""
    print("🎯 Running Accuracy Benchmark Suite")
    print("-" * 50)

    # Generate test data
    generator = TestDataGenerator()
    all_tests = []
    all_tests.extend(generator.generate_basic_tests())
    all_tests.extend(generator.generate_obfuscated_tests())
    all_tests.extend(generator.generate_multilingual_tests())
    all_tests.extend(generator.generate_adversarial_tests())
    all_tests.extend(generator.generate_edge_cases())

    print(f"📊 Total test cases: {len(all_tests)}")

    # Run evaluation
    evaluator = AccuracyEvaluator()
    df, metrics = evaluator.evaluate_batch(all_tests)

    # Print results
    print("\n📈 Overall Metrics:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1']:.3f}")

    # Breakdown by category
    print("\n📊 Metrics by Category:")
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        tp = cat_df['true_positives'].sum()
        fp = cat_df['false_positives'].sum()
        fn = cat_df['false_negatives'].sum()

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        print(f"\n  {category.capitalize()}:")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall: {recall:.3f}")

    # Save results
    results_path = Path("benchmark/reports/accuracy_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'by_category': df.groupby('category').agg({
                'true_positives': 'sum',
                'false_positives': 'sum',
                'false_negatives': 'sum'
            }).to_dict()
        }, f, indent=2)

    print(f"\n💾 Results saved to {results_path}")

    return df, metrics


if __name__ == "__main__":
    df, metrics = run_accuracy_benchmark()