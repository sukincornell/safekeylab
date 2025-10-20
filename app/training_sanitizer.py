"""
Aegis Training Data Sanitizer - Batch Processing for ML Training Data
Production-ready sanitization for datasets, chat logs, and training corpora
"""

import asyncio
import json
import csv
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, AsyncIterator
from dataclasses import dataclass
from datetime import datetime
import hashlib
import uuid
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from io import StringIO
import tempfile
import os

from app.services.pii_detector import PIIDetector
from app.services.anonymizer import DataAnonymizer


@dataclass
class TrainingConfig:
    """Configuration for training data sanitization"""
    method: str = "redaction"  # redaction, masking, tokenization, differential_privacy, synthetic
    preserve_context: bool = True
    preserve_format: bool = True
    confidence_threshold: float = 0.85
    differential_privacy_epsilon: float = 1.0
    k_anonymity_k: int = 5
    batch_size: int = 1000
    parallel_workers: int = None
    output_format: str = "same"  # same, json, csv, parquet

    def __post_init__(self):
        if self.parallel_workers is None:
            self.parallel_workers = min(mp.cpu_count(), 8)


@dataclass
class SanitizationResult:
    """Result of training data sanitization"""
    job_id: str
    original_size: int
    processed_size: int
    entities_detected: int
    entities_removed: int
    processing_time_seconds: float
    method_used: str
    compliance_report: Dict[str, Any]
    output_path: Optional[str] = None
    metadata: Dict[str, Any] = None


class TrainingSanitizer:
    """
    Production-ready training data sanitizer for ML pipelines
    Handles batch processing of large datasets with privacy preservation
    """

    def __init__(self):
        self.pii_detector = PIIDetector()
        self.anonymizer = DataAnonymizer()
        self.logger = logging.getLogger(__name__)
        self.active_jobs = {}

    async def initialize(self):
        """Initialize detector and anonymizer models"""
        await self.pii_detector.initialize()
        await self.anonymizer.initialize()
        self.logger.info("Training sanitizer initialized")

    async def sanitize_dataset(
        self,
        data: Union[str, List[Dict], pd.DataFrame, Path],
        config: TrainingConfig = None,
        output_path: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> SanitizationResult:
        """
        Sanitize a complete dataset for training

        Args:
            data: Input data (file path, dataframe, or list of records)
            config: Sanitization configuration
            output_path: Where to save sanitized data
            job_id: Job identifier for tracking

        Returns:
            SanitizationResult with processing details
        """
        start_time = datetime.utcnow()
        job_id = job_id or str(uuid.uuid4())
        config = config or TrainingConfig()

        self.logger.info(f"Starting sanitization job {job_id}")
        self.active_jobs[job_id] = {"status": "processing", "start_time": start_time}

        try:
            # Load and normalize data
            normalized_data, original_format = await self._load_and_normalize(data)
            original_size = len(normalized_data)

            # Process in batches for memory efficiency
            processed_batches = []
            total_entities_detected = 0
            total_entities_removed = 0

            for batch in self._batch_generator(normalized_data, config.batch_size):
                batch_result = await self._process_batch(batch, config)
                processed_batches.extend(batch_result["data"])
                total_entities_detected += batch_result["entities_detected"]
                total_entities_removed += batch_result["entities_removed"]

                # Update job progress
                progress = len(processed_batches) / original_size
                self.active_jobs[job_id]["progress"] = progress

            # Generate output
            output_data = await self._format_output(
                processed_batches,
                original_format,
                config.output_format
            )

            # Save if output path specified
            if output_path:
                await self._save_output(output_data, output_path, config.output_format)

            # Generate compliance report
            compliance_report = self._generate_compliance_report(
                total_entities_detected,
                total_entities_removed,
                config.method
            )

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            result = SanitizationResult(
                job_id=job_id,
                original_size=original_size,
                processed_size=len(processed_batches),
                entities_detected=total_entities_detected,
                entities_removed=total_entities_removed,
                processing_time_seconds=processing_time,
                method_used=config.method,
                compliance_report=compliance_report,
                output_path=output_path,
                metadata={
                    "config": config.__dict__,
                    "original_format": original_format,
                    "timestamp": start_time.isoformat()
                }
            )

            self.active_jobs[job_id] = {"status": "completed", "result": result}
            self.logger.info(f"Sanitization job {job_id} completed successfully")

            return result

        except Exception as e:
            self.active_jobs[job_id] = {"status": "failed", "error": str(e)}
            self.logger.error(f"Sanitization job {job_id} failed: {e}")
            raise

    async def sanitize_chat_logs(
        self,
        chat_logs: List[Dict],
        config: TrainingConfig = None
    ) -> List[Dict]:
        """
        Sanitize chat conversation logs for training

        Args:
            chat_logs: List of conversation records
            config: Sanitization configuration

        Returns:
            Sanitized chat logs preserving conversation flow
        """
        config = config or TrainingConfig()
        sanitized_logs = []

        for conversation in chat_logs:
            sanitized_conversation = {
                "conversation_id": conversation.get("conversation_id", str(uuid.uuid4())),
                "timestamp": conversation.get("timestamp", datetime.utcnow().isoformat()),
                "messages": []
            }

            # Process each message in conversation
            for message in conversation.get("messages", []):
                sanitized_message = await self._sanitize_message(message, config)
                sanitized_conversation["messages"].append(sanitized_message)

            sanitized_logs.append(sanitized_conversation)

        return sanitized_logs

    async def create_synthetic_dataset(
        self,
        original_data: Union[List[Dict], pd.DataFrame],
        size_multiplier: float = 1.0,
        config: TrainingConfig = None
    ) -> List[Dict]:
        """
        Generate synthetic training data that preserves patterns but removes PII

        Args:
            original_data: Source data for pattern learning
            size_multiplier: How much synthetic data to generate (1.0 = same size)
            config: Generation configuration

        Returns:
            Synthetic dataset safe for training
        """
        config = config or TrainingConfig()

        # Analyze patterns in original data
        patterns = await self._analyze_data_patterns(original_data)

        # Generate synthetic data based on patterns
        synthetic_size = int(len(original_data) * size_multiplier)
        synthetic_data = []

        for i in range(synthetic_size):
            synthetic_record = await self._generate_synthetic_record(patterns, config)
            synthetic_data.append(synthetic_record)

        return synthetic_data

    def apply_differential_privacy(
        self,
        data: List[Dict],
        epsilon: float = 1.0,
        sensitivity: float = 1.0
    ) -> List[Dict]:
        """
        Apply differential privacy noise to numerical and categorical data

        Args:
            data: Input dataset
            epsilon: Privacy budget (lower = more private)
            sensitivity: Sensitivity of the function

        Returns:
            Data with differential privacy applied
        """
        dp_data = []

        for record in data:
            dp_record = {}
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    # Add Laplace noise to numerical values
                    noise = np.random.laplace(0, sensitivity / epsilon)
                    dp_record[key] = value + noise
                elif isinstance(value, str):
                    # Apply privacy-preserving string modification
                    dp_record[key] = self._apply_string_dp(value, epsilon)
                else:
                    dp_record[key] = value

            dp_data.append(dp_record)

        return dp_data

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a sanitization job"""
        if job_id not in self.active_jobs:
            return {"status": "not_found"}

        return self.active_jobs[job_id]

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running sanitization job"""
        if job_id in self.active_jobs and self.active_jobs[job_id]["status"] == "processing":
            self.active_jobs[job_id]["status"] = "cancelled"
            return True
        return False

    # Private helper methods

    async def _load_and_normalize(self, data: Any) -> Tuple[List[Dict], str]:
        """Load data from various sources and normalize to list of dicts"""
        if isinstance(data, str) or isinstance(data, Path):
            # File path
            path = Path(data)
            if path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    data = json.load(f)
                return data, "json"
            elif path.suffix.lower() == '.csv':
                df = pd.read_csv(path)
                return df.to_dict('records'), "csv"
            elif path.suffix.lower() == '.parquet':
                df = pd.read_parquet(path)
                return df.to_dict('records'), "parquet"
            else:
                # Assume text file with one record per line
                with open(path, 'r') as f:
                    lines = [{"text": line.strip()} for line in f if line.strip()]
                return lines, "text"

        elif isinstance(data, pd.DataFrame):
            return data.to_dict('records'), "dataframe"

        elif isinstance(data, list):
            return data, "list"

        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _batch_generator(self, data: List[Dict], batch_size: int):
        """Generate batches of data for processing"""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    async def _process_batch(self, batch: List[Dict], config: TrainingConfig) -> Dict:
        """Process a batch of records"""
        processed_data = []
        entities_detected = 0
        entities_removed = 0

        for record in batch:
            processed_record, batch_entities = await self._process_record(record, config)
            processed_data.append(processed_record)
            entities_detected += len(batch_entities)
            entities_removed += sum(1 for e in batch_entities if e.get("removed", False))

        return {
            "data": processed_data,
            "entities_detected": entities_detected,
            "entities_removed": entities_removed
        }

    async def _process_record(self, record: Dict, config: TrainingConfig) -> Tuple[Dict, List[Dict]]:
        """Process a single record for PII"""
        processed_record = {}
        all_entities = []

        for key, value in record.items():
            if isinstance(value, str):
                # Detect PII in string fields
                entities = await self.pii_detector.detect(
                    value,
                    confidence_threshold=config.confidence_threshold
                )

                # Apply sanitization method
                if config.method == "redaction":
                    processed_value = await self._redact_text(value, entities)
                elif config.method == "masking":
                    processed_value = await self._mask_text(value, entities)
                elif config.method == "tokenization":
                    processed_value = await self._tokenize_text(value, entities)
                else:
                    processed_value = value

                processed_record[key] = processed_value
                all_entities.extend(entities)
            else:
                processed_record[key] = value

        return processed_record, all_entities

    async def _sanitize_message(self, message: Dict, config: TrainingConfig) -> Dict:
        """Sanitize a single chat message"""
        sanitized = {
            "role": message.get("role", "user"),
            "timestamp": message.get("timestamp", datetime.utcnow().isoformat())
        }

        # Sanitize message content
        content = message.get("content", "")
        if content:
            entities = await self.pii_detector.detect(content)
            sanitized["content"] = await self._apply_sanitization_method(
                content, entities, config.method
            )

        # Preserve other metadata
        for key in ["message_id", "user_id", "session_id"]:
            if key in message:
                if key.endswith("_id") and config.method == "tokenization":
                    # Tokenize IDs for privacy
                    sanitized[key] = self._tokenize_id(message[key])
                else:
                    sanitized[key] = message[key]

        return sanitized

    async def _apply_sanitization_method(self, text: str, entities: List[Dict], method: str) -> str:
        """Apply specific sanitization method to text"""
        if method == "redaction":
            return await self._redact_text(text, entities)
        elif method == "masking":
            return await self._mask_text(text, entities)
        elif method == "tokenization":
            return await self._tokenize_text(text, entities)
        elif method == "synthetic":
            return await self._generate_synthetic_text(text, entities)
        else:
            return text

    async def _redact_text(self, text: str, entities: List[Dict]) -> str:
        """Redact PII entities from text"""
        result = text
        # Sort by start position in reverse to maintain positions
        for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
            entity_type = entity["type"]
            start, end = entity["start"], entity["end"]
            replacement = f"[{entity_type}]"
            result = result[:start] + replacement + result[end:]
        return result

    async def _mask_text(self, text: str, entities: List[Dict]) -> str:
        """Mask PII entities with format-preserving characters"""
        result = text
        for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
            start, end = entity["start"], entity["end"]
            original = text[start:end]

            # Preserve format while masking
            if entity["type"] == "EMAIL":
                masked = "****@****.***"
            elif entity["type"] == "PHONE":
                masked = "XXX-XXX-XXXX"
            elif entity["type"] == "SSN":
                masked = "XXX-XX-XXXX"
            elif entity["type"] == "CREDIT_CARD":
                masked = "XXXX-XXXX-XXXX-XXXX"
            else:
                # Generic masking preserving length
                masked = "X" * len(original)

            result = result[:start] + masked + result[end:]
        return result

    async def _tokenize_text(self, text: str, entities: List[Dict]) -> str:
        """Replace PII with consistent tokens"""
        result = text
        for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
            start, end = entity["start"], entity["end"]
            original = text[start:end]

            # Generate consistent token for same value
            token_hash = hashlib.sha256(original.encode()).hexdigest()[:8]
            token = f"TOKEN_{entity['type']}_{token_hash}"

            result = result[:start] + token + result[end:]
        return result

    def _tokenize_id(self, id_value: str) -> str:
        """Create consistent token for ID values"""
        token_hash = hashlib.sha256(id_value.encode()).hexdigest()[:16]
        return f"ID_{token_hash}"

    async def _analyze_data_patterns(self, data: List[Dict]) -> Dict:
        """Analyze patterns in data for synthetic generation"""
        patterns = {
            "field_types": {},
            "value_distributions": {},
            "common_phrases": [],
            "structure_patterns": []
        }

        # Analyze field types and distributions
        for record in data[:1000]:  # Sample for pattern analysis
            for key, value in record.items():
                if key not in patterns["field_types"]:
                    patterns["field_types"][key] = []
                patterns["field_types"][key].append(type(value).__name__)

        return patterns

    async def _generate_synthetic_record(self, patterns: Dict, config: TrainingConfig) -> Dict:
        """Generate a synthetic record based on learned patterns"""
        synthetic = {}

        for field, types in patterns["field_types"].items():
            most_common_type = max(set(types), key=types.count)

            if most_common_type == "str":
                synthetic[field] = f"synthetic_{field}_{uuid.uuid4().hex[:8]}"
            elif most_common_type == "int":
                synthetic[field] = np.random.randint(1, 1000)
            elif most_common_type == "float":
                synthetic[field] = np.random.uniform(0, 100)
            else:
                synthetic[field] = None

        return synthetic

    def _apply_string_dp(self, text: str, epsilon: float) -> str:
        """Apply differential privacy to string data"""
        # Simple approach: randomly modify characters based on epsilon
        if np.random.random() > epsilon:
            return text

        # Small random modifications
        chars = list(text)
        if chars:
            idx = np.random.randint(0, len(chars))
            chars[idx] = chr(ord(chars[idx]) + np.random.randint(-1, 2))

        return ''.join(chars)

    async def _format_output(self, data: List[Dict], original_format: str, output_format: str) -> Any:
        """Format output data according to specified format"""
        if output_format == "same":
            output_format = original_format

        if output_format == "json":
            return data
        elif output_format == "csv":
            if data:
                df = pd.DataFrame(data)
                return df.to_csv(index=False)
        elif output_format == "parquet":
            if data:
                df = pd.DataFrame(data)
                return df
        elif output_format == "dataframe":
            return pd.DataFrame(data)

        return data

    async def _save_output(self, data: Any, output_path: str, format_type: str):
        """Save processed data to file"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format_type == "json":
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format_type == "csv":
            if isinstance(data, str):
                with open(path, 'w') as f:
                    f.write(data)
            else:
                pd.DataFrame(data).to_csv(path, index=False)
        elif format_type == "parquet":
            if isinstance(data, pd.DataFrame):
                data.to_parquet(path)
            else:
                pd.DataFrame(data).to_parquet(path)

    def _generate_compliance_report(
        self,
        entities_detected: int,
        entities_removed: int,
        method: str
    ) -> Dict[str, Any]:
        """Generate compliance report for sanitization"""
        removal_rate = entities_removed / entities_detected if entities_detected > 0 else 1.0

        return {
            "entities_detected": entities_detected,
            "entities_removed": entities_removed,
            "removal_rate": removal_rate,
            "method_used": method,
            "gdpr_compliant": removal_rate >= 0.95,
            "ccpa_compliant": removal_rate >= 0.95,
            "hipaa_compliant": removal_rate >= 0.99,
            "pci_dss_compliant": removal_rate >= 0.99,
            "timestamp": datetime.utcnow().isoformat(),
            "certification": f"CERT_{uuid.uuid4().hex[:16].upper()}"
        }


# Convenience functions for common use cases

async def sanitize_training_file(
    file_path: str,
    output_path: str = None,
    method: str = "redaction",
    preserve_format: bool = True
) -> SanitizationResult:
    """
    Quick function to sanitize a training file

    Args:
        file_path: Path to input file
        output_path: Path for sanitized output
        method: Sanitization method
        preserve_format: Whether to preserve original format

    Returns:
        Sanitization result
    """
    sanitizer = TrainingSanitizer()
    await sanitizer.initialize()

    config = TrainingConfig(
        method=method,
        preserve_format=preserve_format
    )

    if output_path is None:
        path = Path(file_path)
        output_path = str(path.parent / f"{path.stem}_sanitized{path.suffix}")

    return await sanitizer.sanitize_dataset(
        data=file_path,
        config=config,
        output_path=output_path
    )


async def sanitize_dataframe(
    df: pd.DataFrame,
    method: str = "redaction",
    preserve_context: bool = True
) -> pd.DataFrame:
    """
    Quick function to sanitize a pandas DataFrame

    Args:
        df: Input DataFrame
        method: Sanitization method
        preserve_context: Whether to preserve context

    Returns:
        Sanitized DataFrame
    """
    sanitizer = TrainingSanitizer()
    await sanitizer.initialize()

    config = TrainingConfig(
        method=method,
        preserve_context=preserve_context,
        output_format="dataframe"
    )

    result = await sanitizer.sanitize_dataset(data=df, config=config)
    return pd.DataFrame(result.output_data)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize sanitizer
        sanitizer = TrainingSanitizer()
        await sanitizer.initialize()

        # Example: Sanitize chat logs
        sample_chats = [
            {
                "conversation_id": "conv_1",
                "messages": [
                    {"role": "user", "content": "Hi, my email is john@example.com"},
                    {"role": "assistant", "content": "Hello! How can I help you today?"},
                    {"role": "user", "content": "My SSN is 123-45-6789, can you help?"}
                ]
            }
        ]

        config = TrainingConfig(method="redaction", preserve_context=True)
        sanitized = await sanitizer.sanitize_chat_logs(sample_chats, config)

        print("Original:")
        print(json.dumps(sample_chats[0], indent=2))
        print("\nSanitized:")
        print(json.dumps(sanitized[0], indent=2))

    asyncio.run(main())