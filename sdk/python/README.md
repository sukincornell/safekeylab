# Aegis Python SDK

Official Python SDK for Aegis - Enterprise Privacy Shield for AI Systems

## Installation

```bash
pip install aegis-shield
```

## Quick Start

```python
from aegis_sdk import AegisClient

# Initialize client
client = AegisClient(api_key="sk_your_api_key")

# Protect sensitive data
text = "John's email is john@example.com, SSN: 123-45-6789"
result = client.process(text)
print(result.processed_data)
# Output: "[PERSON_NAME]'s email is [EMAIL_REDACTED], SSN: [SSN_REDACTED]"
```

## Features

- 🛡️ **PII Detection**: Identify 25+ types of sensitive data
- 🔒 **Multiple Privacy Methods**: Redaction, masking, tokenization, and more
- ⚡ **High Performance**: <50ms latency with automatic retries
- 🔄 **Batch Processing**: Process multiple items efficiently
- 📊 **Usage Tracking**: Monitor your API usage in real-time
- 🏢 **Enterprise Ready**: Full async support and type hints

## Documentation

Full documentation available at [https://docs.aegis-shield.ai](https://docs.aegis-shield.ai)

## Support

- Email: support@aegis-shield.ai
- Enterprise: enterprise@aegis-shield.ai