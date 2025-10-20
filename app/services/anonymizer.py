"""
Data Anonymization Service with Multiple Privacy Techniques
"""

import hashlib
import random
import string
from typing import Dict, List, Any, Union, Optional
from datetime import datetime, timedelta
import json
import numpy as np
from faker import Faker

fake = Faker()

class DataAnonymizer:
    """Production data anonymization with multiple privacy methods"""

    def __init__(self):
        self.tokenization_map = {}
        self.salt = "aegis-shield-2024"
        self._initialized = False

    async def initialize(self):
        """Initialize anonymization service"""
        self._initialized = True

    async def process(
        self,
        data: Union[str, Dict, List],
        entities: List[Dict],
        method: str = "auto",
        format: str = "text"
    ) -> Union[str, Dict, List]:
        """
        Apply privacy protection to data

        Args:
            data: Original data
            entities: Detected PII entities
            method: Privacy method to apply
            format: Data format

        Returns:
            Anonymized data
        """
        if method == "auto":
            method = self._select_best_method(entities, format)

        if method == "redaction":
            return self._redact(data, entities, format)
        elif method == "masking":
            return self._mask(data, entities, format)
        elif method == "tokenization":
            return self._tokenize(data, entities, format)
        elif method == "differential_privacy":
            return self._apply_differential_privacy(data, entities, format)
        elif method == "k_anonymity":
            return self._apply_k_anonymity(data, entities, format)
        elif method == "synthetic":
            return self._generate_synthetic(data, entities, format)
        else:
            return self._redact(data, entities, format)

    def _select_best_method(self, entities: List[Dict], format: str) -> str:
        """Select the best anonymization method based on data characteristics"""
        if not entities:
            return "none"

        # Count entity types
        entity_types = [e["type"] for e in entities]

        # For financial data, use tokenization
        if any(t in ["CREDIT_CARD", "BANK_ACCOUNT", "SSN"] for t in entity_types):
            return "tokenization"

        # For structured data, use k-anonymity
        if format in ["csv", "structured", "json"]:
            return "k_anonymity"

        # Default to redaction for text
        return "redaction"

    def _redact(self, data: Any, entities: List[Dict], format: str) -> Any:
        """Replace PII with redacted placeholders"""
        if format == "text":
            text = str(data)
            # Sort entities by position (reverse) to maintain positions
            sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)

            for entity in sorted_entities:
                replacement = f"[{entity['type']}_REDACTED]"
                text = text[:entity["start"]] + replacement + text[entity["end"]:]

            return text

        elif format == "json":
            return self._redact_json(data, entities)

        return data

    def _redact_json(self, data: Dict, entities: List[Dict]) -> Dict:
        """Redact PII from JSON data"""
        if isinstance(data, str):
            data = json.loads(data)

        def redact_value(value, entities):
            if isinstance(value, str):
                for entity in entities:
                    if entity["text"] in value:
                        value = value.replace(entity["text"], f"[{entity['type']}_REDACTED]")
            return value

        def redact_recursive(obj):
            if isinstance(obj, dict):
                return {k: redact_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [redact_recursive(item) for item in obj]
            else:
                return redact_value(obj, entities)

        return redact_recursive(data)

    def _mask(self, data: Any, entities: List[Dict], format: str) -> Any:
        """Partially mask PII (show first/last few characters)"""
        if format == "text":
            text = str(data)
            sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)

            for entity in sorted_entities:
                masked = self._mask_value(entity["text"], entity["type"])
                text = text[:entity["start"]] + masked + text[entity["end"]:]

            return text

        return data

    def _mask_value(self, value: str, entity_type: str) -> str:
        """Mask a value based on its type"""
        if entity_type == "EMAIL":
            parts = value.split("@")
            if len(parts) == 2:
                masked_local = parts[0][:2] + "***"
                return f"{masked_local}@{parts[1]}"

        elif entity_type == "PHONE":
            if len(value) >= 10:
                return value[:3] + "****" + value[-2:]

        elif entity_type == "CREDIT_CARD":
            digits = "".join(c for c in value if c.isdigit())
            if len(digits) == 16:
                return "**** **** **** " + digits[-4:]

        elif entity_type == "SSN":
            return "***-**-" + value[-4:] if len(value) >= 4 else "***-**-****"

        # Default masking
        if len(value) > 4:
            return value[:2] + "*" * (len(value) - 4) + value[-2:]
        return "*" * len(value)

    def _tokenize(self, data: Any, entities: List[Dict], format: str) -> Any:
        """Replace PII with reversible tokens"""
        if format == "text":
            text = str(data)
            sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)

            for entity in sorted_entities:
                token = self._generate_token(entity["text"], entity["type"])
                text = text[:entity["start"]] + token + text[entity["end"]:]
                # Store mapping for potential reversal
                self.tokenization_map[token] = entity["text"]

            return text

        return data

    def _generate_token(self, value: str, entity_type: str) -> str:
        """Generate a deterministic token for a value"""
        hash_input = f"{value}{entity_type}{self.salt}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        return f"TOK_{entity_type}_{hash_value}"

    def _apply_differential_privacy(self, data: Any, entities: List[Dict], format: str) -> Any:
        """Apply differential privacy with noise addition"""
        if format == "text":
            # For text, redact and add noise to numeric values
            text = self._redact(data, entities, format)

            # Add Laplace noise to any numeric values
            import re
            numeric_pattern = re.compile(r'\b\d+\.?\d*\b')

            def add_noise(match):
                value = float(match.group())
                # Add Laplace noise with epsilon=1.0
                noise = np.random.laplace(0, 1.0)
                noisy_value = value + noise
                return str(round(noisy_value, 2))

            text = numeric_pattern.sub(add_noise, text)
            return text

        return data

    def _apply_k_anonymity(self, data: Any, entities: List[Dict], format: str, k: int = 3) -> Any:
        """Apply k-anonymity generalization"""
        if format in ["json", "structured"]:
            # Generalize quasi-identifiers
            generalized = self._generalize_data(data, entities, k)
            return generalized

        # For text, use redaction
        return self._redact(data, entities, format)

    def _generalize_data(self, data: Any, entities: List[Dict], k: int) -> Any:
        """Generalize data for k-anonymity"""
        if isinstance(data, dict):
            generalized = data.copy()

            for entity in entities:
                if entity["type"] == "DATE_OF_BIRTH":
                    # Generalize to year only
                    generalized = self._replace_in_dict(
                        generalized,
                        entity["text"],
                        entity["text"][:4] + "-**-**"
                    )
                elif entity["type"] == "ZIP_CODE":
                    # Generalize to first 3 digits
                    generalized = self._replace_in_dict(
                        generalized,
                        entity["text"],
                        entity["text"][:3] + "**"
                    )
                elif entity["type"] == "AGE":
                    # Generalize to age range
                    age = int(entity["text"])
                    range_start = (age // 10) * 10
                    generalized = self._replace_in_dict(
                        generalized,
                        entity["text"],
                        f"{range_start}-{range_start + 9}"
                    )

            return generalized

        return data

    def _replace_in_dict(self, obj: Any, old: str, new: str) -> Any:
        """Recursively replace values in dictionary"""
        if isinstance(obj, dict):
            return {k: self._replace_in_dict(v, old, new) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_in_dict(item, old, new) for item in obj]
        elif isinstance(obj, str):
            return obj.replace(old, new)
        return obj

    def _generate_synthetic(self, data: Any, entities: List[Dict], format: str) -> Any:
        """Generate synthetic data preserving structure"""
        if format == "text":
            text = str(data)
            sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)

            for entity in sorted_entities:
                synthetic = self._create_synthetic_value(entity["type"])
                text = text[:entity["start"]] + synthetic + text[entity["end"]:]

            return text

        return data

    def _create_synthetic_value(self, entity_type: str) -> str:
        """Create realistic synthetic data based on type"""
        if entity_type == "PERSON_NAME":
            return fake.name()
        elif entity_type == "EMAIL":
            return fake.email()
        elif entity_type == "PHONE":
            return fake.phone_number()
        elif entity_type == "ADDRESS":
            return fake.address().replace("\n", ", ")
        elif entity_type == "CREDIT_CARD":
            return fake.credit_card_number()
        elif entity_type == "SSN":
            return fake.ssn()
        elif entity_type == "DATE_OF_BIRTH":
            return fake.date_of_birth().strftime("%Y-%m-%d")
        elif entity_type == "COMPANY":
            return fake.company()
        else:
            return f"[SYNTHETIC_{entity_type}]"

    def reverse_tokenization(self, token: str) -> Optional[str]:
        """Reverse a tokenization (if mapping exists)"""
        return self.tokenization_map.get(token)

    async def cleanup(self):
        """Cleanup resources"""
        self.tokenization_map.clear()
        self._initialized = False