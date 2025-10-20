"""
Aegis Python SDK - Enterprise Privacy Shield for AI
"""

import os
import time
import json
import hashlib
import hmac
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import httpx
from httpx import AsyncClient, Client

__version__ = "1.0.0"

class PrivacyMethod(Enum):
    """Available privacy protection methods"""
    REDACTION = "redaction"
    MASKING = "masking"
    TOKENIZATION = "tokenization"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    K_ANONYMITY = "k_anonymity"
    SYNTHETIC = "synthetic"
    AUTO = "auto"

class DataFormat(Enum):
    """Supported data formats"""
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    STRUCTURED = "structured"

@dataclass
class ProcessedResult:
    """Result from processing data"""
    request_id: str
    processed_data: Union[str, Dict, List]
    entities_detected: List[Dict]
    processing_time_ms: float
    method_used: str
    compliance: Dict[str, bool]
    risk_score: float
    timestamp: str

@dataclass
class UsageStats:
    """API usage statistics"""
    requests: int
    data_processed_mb: float
    entities_detected: int
    average_latency_ms: float
    requests_remaining: int

class AegisException(Exception):
    """Base exception for Aegis SDK"""
    pass

class AuthenticationError(AegisException):
    """Authentication failed"""
    pass

class RateLimitError(AegisException):
    """Rate limit exceeded"""
    pass

class ProcessingError(AegisException):
    """Data processing failed"""
    pass

class AegisClient:
    """
    Aegis API Client

    Example:
        >>> from aegis_sdk import AegisClient
        >>> client = AegisClient(api_key="sk_your_api_key")
        >>>
        >>> # Process text
        >>> result = client.process("John's email is john@example.com")
        >>> print(result.processed_data)
        '[PERSON_NAME]'s email is [EMAIL_REDACTED]'
        >>>
        >>> # Detect only
        >>> entities = client.detect("SSN: 123-45-6789")
        >>> print(entities)
        [{'type': 'SSN', 'text': '123-45-6789', 'confidence': 0.95}]
    """

    DEFAULT_BASE_URL = "https://api.aegis-shield.ai"

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize Aegis client

        Args:
            api_key: Your Aegis API key (or set AEGIS_API_KEY env var)
            base_url: API base URL (default: https://api.aegis-shield.ai)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key or os.environ.get("AEGIS_API_KEY")
        if not self.api_key:
            raise AuthenticationError("API key required. Set AEGIS_API_KEY or pass api_key parameter")

        self.base_url = base_url or os.environ.get("AEGIS_BASE_URL", self.DEFAULT_BASE_URL)
        self.timeout = timeout
        self.max_retries = max_retries

        self._client = Client(
            base_url=self.base_url,
            headers=self._get_headers(),
            timeout=timeout
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "X-API-Key": self.api_key,
            "User-Agent": f"aegis-python-sdk/{__version__}",
            "Content-Type": "application/json"
        }

    def _handle_response(self, response: httpx.Response) -> Dict:
        """Handle API response"""
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        elif response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        elif response.status_code >= 500:
            raise ProcessingError(f"Server error: {response.text}")
        else:
            raise AegisException(f"API error: {response.status_code} - {response.text}")

    def process(
        self,
        data: Union[str, Dict, List],
        method: PrivacyMethod = PrivacyMethod.AUTO,
        format: DataFormat = DataFormat.TEXT,
        confidence_threshold: float = 0.85,
        custom_patterns: Optional[List[Dict]] = None
    ) -> ProcessedResult:
        """
        Process data to remove/protect PII

        Args:
            data: Data to process
            method: Privacy method to use
            format: Data format
            confidence_threshold: Minimum confidence for detection
            custom_patterns: Additional patterns to detect

        Returns:
            ProcessedResult with anonymized data
        """
        payload = {
            "data": data,
            "method": method.value,
            "format": format.value,
            "confidence_threshold": confidence_threshold,
            "detect_only": False,
            "return_metrics": True
        }

        if custom_patterns:
            payload["custom_patterns"] = custom_patterns

        response = self._client.post("/v1/process", json=payload)
        result = self._handle_response(response)

        return ProcessedResult(
            request_id=result["request_id"],
            processed_data=result["processed_data"],
            entities_detected=result["entities_detected"],
            processing_time_ms=result["processing_time_ms"],
            method_used=result["method_used"],
            compliance=result["compliance"],
            risk_score=result["risk_score"],
            timestamp=result["timestamp"]
        )

    def detect(
        self,
        data: Union[str, Dict, List],
        format: DataFormat = DataFormat.TEXT,
        confidence_threshold: float = 0.85
    ) -> List[Dict]:
        """
        Detect PII without modifying data

        Args:
            data: Data to analyze
            format: Data format
            confidence_threshold: Minimum confidence for detection

        Returns:
            List of detected entities
        """
        payload = {
            "data": data,
            "format": format.value,
            "confidence_threshold": confidence_threshold,
            "detect_only": True
        }

        response = self._client.post("/v1/process", json=payload)
        result = self._handle_response(response)

        return result["entities_detected"]

    def batch_process(
        self,
        items: List[Union[str, Dict]],
        method: PrivacyMethod = PrivacyMethod.AUTO,
        format: DataFormat = DataFormat.TEXT
    ) -> List[ProcessedResult]:
        """
        Process multiple items in batch

        Args:
            items: List of items to process
            method: Privacy method to use
            format: Data format

        Returns:
            List of ProcessedResult
        """
        requests = [
            {
                "data": item,
                "method": method.value,
                "format": format.value
            }
            for item in items
        ]

        response = self._client.post("/v1/batch", json={"requests": requests})
        result = self._handle_response(response)

        return [
            ProcessedResult(**r) for r in result["results"]
        ]

    def get_usage(self) -> UsageStats:
        """
        Get current usage statistics

        Returns:
            UsageStats with current month's usage
        """
        response = self._client.get("/v1/usage")
        result = self._handle_response(response)

        return UsageStats(
            requests=result["current_month"]["requests"],
            data_processed_mb=result["current_month"]["data_processed_mb"],
            entities_detected=result["current_month"]["entities_detected"],
            average_latency_ms=result["current_month"]["average_latency_ms"],
            requests_remaining=result["quota"]["requests_remaining"]
        )

    def get_compliance_report(self, request_id: str) -> Dict:
        """
        Get compliance report for a processed request

        Args:
            request_id: Request ID from process() result

        Returns:
            Compliance report with certifications
        """
        response = self._client.get(f"/v1/compliance/{request_id}")
        return self._handle_response(response)

    def get_patterns(self) -> Dict:
        """
        Get available detection patterns

        Returns:
            Dictionary of supported patterns and entities
        """
        response = self._client.get("/v1/patterns")
        return self._handle_response(response)

    def close(self):
        """Close the client connection"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class AsyncAegisClient:
    """
    Async version of Aegis client

    Example:
        >>> import asyncio
        >>> from aegis_sdk import AsyncAegisClient
        >>>
        >>> async def main():
        >>>     async with AsyncAegisClient(api_key="sk_your_key") as client:
        >>>         result = await client.process("PII text here")
        >>>         print(result.processed_data)
        >>>
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        self.api_key = api_key or os.environ.get("AEGIS_API_KEY")
        if not self.api_key:
            raise AuthenticationError("API key required")

        self.base_url = base_url or os.environ.get("AEGIS_BASE_URL", AegisClient.DEFAULT_BASE_URL)
        self.timeout = timeout
        self.max_retries = max_retries

        self._client = AsyncClient(
            base_url=self.base_url,
            headers=self._get_headers(),
            timeout=timeout
        )

    def _get_headers(self) -> Dict[str, str]:
        return {
            "X-API-Key": self.api_key,
            "User-Agent": f"aegis-python-sdk/{__version__}",
            "Content-Type": "application/json"
        }

    async def _handle_response(self, response: httpx.Response) -> Dict:
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        elif response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        else:
            raise AegisException(f"API error: {response.status_code}")

    async def process(
        self,
        data: Union[str, Dict, List],
        method: PrivacyMethod = PrivacyMethod.AUTO,
        format: DataFormat = DataFormat.TEXT,
        confidence_threshold: float = 0.85
    ) -> ProcessedResult:
        """Async version of process()"""
        payload = {
            "data": data,
            "method": method.value,
            "format": format.value,
            "confidence_threshold": confidence_threshold,
            "detect_only": False
        }

        response = await self._client.post("/v1/process", json=payload)
        result = await self._handle_response(response)

        return ProcessedResult(**result)

    async def close(self):
        """Close the async client"""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Convenience functions
def quick_protect(text: str, api_key: str = None) -> str:
    """
    Quick function to protect text

    Example:
        >>> from aegis_sdk import quick_protect
        >>> protected = quick_protect("John's SSN is 123-45-6789")
        >>> print(protected)
        '[NAME_REDACTED]'s SSN is [SSN_REDACTED]'
    """
    with AegisClient(api_key=api_key) as client:
        result = client.process(text)
        return result.processed_data

def quick_detect(text: str, api_key: str = None) -> List[Dict]:
    """
    Quick function to detect PII

    Example:
        >>> from aegis_sdk import quick_detect
        >>> entities = quick_detect("Email: john@example.com")
        >>> print(entities)
        [{'type': 'EMAIL', 'text': 'john@example.com', 'confidence': 0.95}]
    """
    with AegisClient(api_key=api_key) as client:
        return client.detect(text)