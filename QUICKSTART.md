# 🚀 Aegis Quick Start Guide

## Launch in 30 Seconds

### Prerequisites
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- Docker Compose installed ([Get Docker Compose](https://docs.docker.com/compose/install/))

### 1. Clone and Launch

```bash
# Clone the repository (or use existing)
cd /Users/sukinyang/aegis

# Launch Aegis
./launch.sh
```

Choose option 1 for "Quick Start" - this launches immediately without any dependencies.

### 2. Access Services

- 🌐 **Website**: http://localhost
- 🔌 **API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs

### 3. Test the API

```bash
# Test PII protection
curl -X POST http://localhost:8000/v1/process \
  -H "X-API-Key: sk_test_key" \
  -H "Content-Type: application/json" \
  -d '{"data": "John email is john@example.com, SSN 123-45-6789"}'
```

Expected response:
```json
{
  "processed_data": "John email is [EMAIL_REDACTED], SSN [SSN_REDACTED]",
  "entities_detected": [
    {"type": "EMAIL", "text": "john@example.com"},
    {"type": "SSN", "text": "123-45-6789"}
  ]
}
```

### 4. Run Full Test Suite

```bash
python3 test_api.py
```

## Using the Python SDK

### Install
```bash
pip install httpx pydantic
```

### Use
```python
import sys
sys.path.append('sdk/python')
from aegis_sdk import AegisClient

client = AegisClient(api_key="sk_test_key", base_url="http://localhost:8000")
result = client.process("My SSN is 123-45-6789")
print(result.processed_data)  # "My SSN is [SSN_REDACTED]"
```

## Production Deployment

### Full Stack with Database

```bash
# Edit configuration
cp .env.example .env
# Edit .env with your settings

# Launch full stack
./launch.sh
# Choose option 2

# Services available:
# - API: http://localhost:8000
# - Website: http://localhost
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9090
```

### Deploy to Cloud

```bash
# Build production image
docker build -t aegis-api:latest -f Dockerfile.simple .

# Push to registry
docker tag aegis-api:latest your-registry/aegis-api:latest
docker push your-registry/aegis-api:latest

# Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml
```

## Architecture

```
Quick Start Mode:
┌─────────────┐     ┌─────────────┐
│   Website   │────▶│   API       │
│  (Nginx)    │     │ (FastAPI)   │
└─────────────┘     └─────────────┘
   Port 80            Port 8000

Full Stack Mode:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Website   │────▶│   API       │────▶│  PostgreSQL │
│  (Nginx)    │     │ (FastAPI)   │     └─────────────┘
└─────────────┘     └─────────────┘            │
   Port 80            Port 8000                ▼
                           │              ┌─────────────┐
                           └─────────────▶│   Redis     │
                                         └─────────────┘
```

## Troubleshooting

### Port Already in Use
```bash
# Stop existing services
docker-compose down

# Or change ports in docker-compose.simple.yml
```

### API Not Responding
```bash
# Check logs
docker-compose logs -f api

# Restart services
docker-compose restart
```

### Permission Denied
```bash
# Make scripts executable
chmod +x launch.sh test_api.py
```

## Next Steps

1. **Get API Key**: Contact sales@aegis-shield.ai
2. **Configure Production**: Edit `.env` with real credentials
3. **Deploy to Cloud**: Use Kubernetes manifests in `/k8s`
4. **Monitor Usage**: Access Grafana at http://localhost:3000
5. **Scale Up**: Increase replicas in docker-compose.yml

## Support

- 📧 Email: support@aegis-shield.ai
- 💼 Enterprise: enterprise@aegis-shield.ai
- 📚 Docs: https://docs.aegis-shield.ai

---

**Ready to protect your AI from $600M GDPR fines? Aegis is running! ⚡**