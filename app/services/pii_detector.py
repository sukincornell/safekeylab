"""
Production PII Detection Service with ML Models
"""

import re
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from functools import lru_cache

# Using Presidio for production-ready PII detection
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider

@dataclass
class Entity:
    text: str
    type: str
    start: int
    end: int
    confidence: float
    replacement: Optional[str] = None

class PIIDetector:
    """Production-ready PII detection using Presidio and custom models"""

    def __init__(self):
        self.analyzer = None
        self.custom_patterns = {}
        self._initialized = False

    async def initialize(self):
        """Initialize the PII detection engine"""
        if self._initialized:
            return

        # Create NLP engine (using spaCy)
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
        })

        # Initialize analyzer with the NLP engine
        self.analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=["en"]
        )

        # Add custom recognizers
        self._add_custom_recognizers()
        self._initialized = True

    def _add_custom_recognizers(self):
        """Add custom pattern recognizers"""

        # API Key patterns
        api_key_patterns = [
            Pattern(name="api_key", regex=r"sk[_-][A-Za-z0-9]{20,}", score=0.9),
            Pattern(name="api_key", regex=r"pk[_-][A-Za-z0-9]{20,}", score=0.9),
            Pattern(name="api_key", regex=r"api[_-]?key[_-][A-Za-z0-9]{20,}", score=0.9),
        ]
        api_recognizer = PatternRecognizer(
            supported_entity="API_KEY",
            patterns=api_key_patterns
        )
        self.analyzer.registry.add_recognizer(api_recognizer)

        # Enhanced SSN pattern
        ssn_patterns = [
            Pattern(name="ssn", regex=r"\b\d{3}-\d{2}-\d{4}\b", score=0.85),
            Pattern(name="ssn", regex=r"\b\d{9}\b", score=0.6),
        ]
        ssn_recognizer = PatternRecognizer(
            supported_entity="US_SSN",
            patterns=ssn_patterns
        )
        self.analyzer.registry.add_recognizer(ssn_recognizer)

        # Bank account patterns
        bank_patterns = [
            Pattern(name="iban", regex=r"[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}", score=0.9),
            Pattern(name="routing", regex=r"\b\d{9}\b", score=0.5),
        ]
        bank_recognizer = PatternRecognizer(
            supported_entity="BANK_ACCOUNT",
            patterns=bank_patterns
        )
        self.analyzer.registry.add_recognizer(bank_recognizer)

    async def detect(
        self,
        data: Union[str, Dict, List],
        format: str = "text",
        confidence_threshold: float = 0.85,
        custom_patterns: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Detect PII entities in the data

        Args:
            data: Input data to analyze
            format: Format of the data (text, json, csv, structured)
            confidence_threshold: Minimum confidence score for detection
            custom_patterns: Additional patterns to detect

        Returns:
            List of detected entities
        """
        if not self._initialized:
            await self.initialize()

        # Convert data to text if needed
        text = self._prepare_text(data, format)

        # Apply custom patterns if provided
        if custom_patterns:
            self._apply_custom_patterns(custom_patterns)

        # Analyze text with Presidio
        results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=None  # Detect all entity types
        )

        # Convert to our format and filter by confidence
        entities = []
        for result in results:
            if result.score >= confidence_threshold:
                entities.append({
                    "text": text[result.start:result.end],
                    "type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "confidence": result.score
                })

        # Add custom detection for edge cases
        entities.extend(self._detect_custom_entities(text, confidence_threshold))

        return entities

    def _prepare_text(self, data: Union[str, Dict, List], format: str) -> str:
        """Convert various data formats to text"""
        if format == "text":
            return str(data)
        elif format == "json":
            import json
            return json.dumps(data) if not isinstance(data, str) else data
        elif format == "csv":
            if isinstance(data, list):
                return "\n".join([",".join(map(str, row)) for row in data])
            return str(data)
        else:
            return str(data)

    def _apply_custom_patterns(self, patterns: List[Dict]):
        """Apply user-defined custom patterns"""
        for pattern in patterns:
            if "regex" in pattern and "type" in pattern:
                custom_recognizer = PatternRecognizer(
                    supported_entity=pattern["type"],
                    patterns=[Pattern(
                        name=pattern.get("name", "custom"),
                        regex=pattern["regex"],
                        score=pattern.get("confidence", 0.8)
                    )]
                )
                self.analyzer.registry.add_recognizer(custom_recognizer)

    def _detect_custom_entities(self, text: str, threshold: float) -> List[Dict]:
        """Detect entities not covered by Presidio"""
        entities = []

        # Detect passwords (basic pattern)
        password_pattern = r'(?i)(password|passwd|pwd|pass)[\s:=]+[\S]+'
        for match in re.finditer(password_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "PASSWORD",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.9
            })

        # Detect JWT tokens
        jwt_pattern = r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+'
        for match in re.finditer(jwt_pattern, text):
            entities.append({
                "text": match.group()[:20] + "...",  # Truncate for security
                "type": "JWT_TOKEN",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.95
            })

        # Detect AWS keys
        aws_pattern = r'AKIA[0-9A-Z]{16}'
        for match in re.finditer(aws_pattern, text):
            entities.append({
                "text": match.group()[:8] + "...",
                "type": "AWS_KEY",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.95
            })

        return [e for e in entities if e["confidence"] >= threshold]

    def get_patterns(self) -> Dict:
        """Get all detection patterns"""
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "api_key": r'(sk|pk|api[_-]?key)[_-][A-Za-z0-9]{20,}',
            "jwt": r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+',
            "aws_key": r'AKIA[0-9A-Z]{16}'
        }
        return patterns

    async def cleanup(self):
        """Cleanup resources"""
        self._initialized = False
        self.analyzer = None

# Fast regex-based detector for lightweight operations
class FastPIIDetector:
    """Lightweight PII detector using only regex patterns"""

    PATTERNS = {
        "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "PHONE": re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'),
        "SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "CREDIT_CARD": re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
        "IP_ADDRESS": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        "API_KEY": re.compile(r'(sk|pk|api[_-]?key)[_-][A-Za-z0-9]{20,}'),
        "JWT_TOKEN": re.compile(r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+'),
        "AWS_KEY": re.compile(r'AKIA[0-9A-Z]{16}')
    }

    @classmethod
    def detect(cls, text: str) -> List[Dict]:
        """Fast detection using regex patterns only"""
        entities = []
        for entity_type, pattern in cls.PATTERNS.items():
            for match in pattern.finditer(text):
                entities.append({
                    "text": match.group(),
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9
                })
        return entities