# AEGIS SECURITY IMPLEMENTATION GUIDE
## Enterprise-Grade Authentication & Access Control

---

## 🔒 SECURITY LAYERS

### Layer 1: Authentication Methods

#### **BASIC** - Development/Testing
```bash
curl -X POST https://api.aegis.ai/v3/protect \
  -H "X-API-Key: ak_test_xxxxx"
```
- Single API key
- Rate limited to 100 requests/minute
- For development only

#### **STANDARD** - Small/Medium Business
```bash
curl -X POST https://api.aegis.ai/v3/protect \
  -H "X-API-Key: ak_live_xxxxx" \
  -H "X-API-Secret: as_live_yyyyy"
```
- API Key + Secret pair
- Rate limited to 1,000 requests/minute
- Audit logging enabled

#### **ENHANCED** - Enterprise
```bash
curl -X POST https://api.aegis.ai/v3/protect \
  -H "X-API-Key: ak_prod_xxxxx" \
  -H "X-API-Secret: as_prod_yyyyy" \
  -H "X-Client-IP: 192.168.1.100"
```
- API Key + Secret + IP Whitelist
- Rate limited to 10,000 requests/minute
- Full audit trail
- Webhook notifications

#### **MAXIMUM** - Financial/Healthcare
```bash
# First, get JWT token with MFA
TOKEN=$(curl -X POST https://api.aegis.ai/auth/login \
  -d '{
    "email": "admin@company.com",
    "password": "SecurePass123!",
    "mfa_token": "123456"
  }' | jq -r '.access_token')

# Then make API call with token
curl -X POST https://api.aegis.ai/v3/protect \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-API-Key: ak_secure_xxxxx" \
  -H "X-API-Secret: as_secure_yyyyy" \
  -H "X-Request-Signature: sha256_signature"
```
- JWT + API Key + Secret + MFA + Request Signing
- Custom rate limits
- Real-time anomaly detection
- Complete compliance logging

---

## 🔐 IMPLEMENTATION OPTIONS

### Option 1: Direct API Integration
**Best for:** SaaS companies, APIs, web services

```python
import requests
import hmac
import hashlib
import time

class AegisClient:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.aegis.ai/v3"

    def protect_data(self, text):
        # Generate request signature
        timestamp = str(int(time.time()))
        message = f"{self.api_key}:{timestamp}:{text}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        # Make secure request
        response = requests.post(
            f"{self.base_url}/protect",
            headers={
                "X-API-Key": self.api_key,
                "X-API-Secret": self.api_secret,
                "X-Timestamp": timestamp,
                "X-Signature": signature
            },
            json={"data": text}
        )

        return response.json()

# Usage
client = AegisClient("ak_prod_xxx", "as_prod_yyy")
protected = client.protect_data("John Doe, SSN 123-45-6789")
```

### Option 2: SDK Integration
**Best for:** Enterprise applications

```python
from aegis import AegisSDK

# Initialize with enhanced security
aegis = AegisSDK(
    api_key="ak_prod_xxx",
    api_secret="as_prod_yyy",
    security_level="enhanced",
    ip_whitelist=["192.168.1.0/24"],
    enable_caching=True,
    retry_on_failure=True
)

# Protect data with context
result = aegis.protect(
    data="Customer data here...",
    context={
        "user_id": "user123",
        "session_id": "sess456",
        "compliance": ["GDPR", "HIPAA"]
    }
)
```

### Option 3: Proxy Server
**Best for:** Microservices, legacy systems

```yaml
# docker-compose.yml
version: '3.8'
services:
  aegis-proxy:
    image: aegis/privacy-proxy:latest
    environment:
      AEGIS_API_KEY: ${AEGIS_API_KEY}
      AEGIS_API_SECRET: ${AEGIS_API_SECRET}
      UPSTREAM_URL: http://your-api:8080
      SECURITY_LEVEL: enhanced
    ports:
      - "8443:8443"
    volumes:
      - ./config/whitelist.json:/etc/aegis/whitelist.json
      - ./logs:/var/log/aegis
```

### Option 4: On-Premise Deployment
**Best for:** Highly regulated industries, air-gapped environments

```bash
# Deploy Aegis on-premise with Kubernetes
kubectl create namespace aegis

# Create secrets
kubectl create secret generic aegis-license \
  --from-literal=key=${LICENSE_KEY} \
  -n aegis

# Deploy with Helm
helm install aegis ./charts/aegis \
  --namespace aegis \
  --values values-production.yaml \
  --set security.level=maximum \
  --set persistence.enabled=true \
  --set monitoring.enabled=true
```

---

## 🛡️ SECURITY FEATURES

### 1. Multi-Factor Authentication (MFA)
```javascript
// Setup MFA for account
const { secret, qr_code } = await aegis.auth.enableMFA({
  method: "totp",  // or "sms", "email", "hardware"
  backup_codes: 10
});

// Verify MFA token
const session = await aegis.auth.login({
  email: "admin@company.com",
  password: "SecurePass123!",
  mfa_token: "123456"  // From authenticator app
});
```

### 2. API Key Rotation
```bash
# Rotate keys every 90 days
curl -X POST https://api.aegis.ai/auth/api-key/rotate \
  -H "X-API-Key: ak_old_xxxxx" \
  -H "X-API-Secret: as_old_yyyyy"

# Response
{
  "api_key": "ak_new_zzzzz",
  "api_secret": "as_new_wwwww",
  "expires_at": "2025-01-15T00:00:00Z",
  "grace_period_hours": 24
}
```

### 3. IP Whitelisting
```python
# Configure IP whitelist
aegis.security.update_whitelist([
    "192.168.1.0/24",    # Office network
    "10.0.0.0/8",        # VPN range
    "52.44.128.0/20"     # AWS NAT Gateway
])

# Enable geo-blocking
aegis.security.set_geo_restrictions({
    "allowed_countries": ["US", "CA", "GB"],
    "blocked_countries": ["RU", "CN", "KP"]
})
```

### 4. Request Signing
```python
import hmac
import hashlib
import base64

def sign_request(method, path, body, secret):
    """Generate HMAC-SHA256 signature for request"""
    message = f"{method}:{path}:{body}"
    signature = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode()

# Add signature to request
signature = sign_request("POST", "/v3/protect", json.dumps(data), api_secret)
headers["X-Request-Signature"] = signature
```

### 5. Audit Logging
```python
# Query audit logs
logs = aegis.audit.query({
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "event_types": ["api_call", "auth_failure", "key_rotation"],
    "customer_id": "cust_123"
})

# Set up real-time alerts
aegis.audit.create_alert({
    "condition": "failed_auth_attempts > 5",
    "window": "5m",
    "action": "email",
    "recipients": ["security@company.com"]
})
```

---

## 📊 MONITORING & COMPLIANCE

### Real-time Dashboard
```python
# Get security metrics
metrics = aegis.monitoring.get_metrics({
    "period": "24h",
    "metrics": [
        "api_calls_total",
        "auth_failures",
        "data_processed_gb",
        "threats_blocked"
    ]
})

# Security score
score = aegis.compliance.get_security_score()
print(f"Security Score: {score}/100")
```

### Compliance Reports
```python
# Generate compliance report
report = aegis.compliance.generate_report({
    "regulation": "GDPR",
    "period": "Q1-2024",
    "format": "pdf"
})

# Automated compliance checks
aegis.compliance.schedule_audit({
    "frequency": "weekly",
    "regulations": ["GDPR", "CCPA", "HIPAA"],
    "notify": ["compliance@company.com"]
})
```

---

## 🚨 INCIDENT RESPONSE

### Automatic Threat Response
```python
# Configure auto-response
aegis.security.set_threat_response({
    "sql_injection": "block_and_alert",
    "prompt_injection": "sanitize_and_log",
    "data_exfiltration": "block_ip_24h",
    "brute_force": "rate_limit_progressive"
})

# Manual incident handling
if threat_detected:
    aegis.incident.create({
        "severity": "high",
        "type": "data_leak_attempt",
        "source_ip": request.client_ip,
        "action_taken": "blocked",
        "notify_team": True
    })
```

---

## 🔧 BEST PRACTICES

### 1. **Never expose secrets in code**
```python
# ❌ BAD
api_key = "ak_prod_xxx"

# ✅ GOOD
api_key = os.environ.get("AEGIS_API_KEY")
```

### 2. **Use environment-specific keys**
```python
# Development
if ENV == "development":
    api_key = "ak_test_xxx"

# Production
elif ENV == "production":
    api_key = vault.get_secret("aegis/prod/api_key")
```

### 3. **Implement retry logic**
```python
@retry(max_attempts=3, backoff=2)
def protect_with_retry(data):
    return aegis.protect(data)
```

### 4. **Cache responses when appropriate**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def detect_pii_cached(text_hash):
    return aegis.detect(text_hash)
```

### 5. **Monitor rate limits**
```python
response = aegis.protect(data)
remaining = response.headers.get("X-RateLimit-Remaining")
if int(remaining) < 100:
    logger.warning(f"Rate limit low: {remaining}")
```

---

## 📱 CUSTOMER PORTAL

### Self-Service Dashboard
Customers get access to:
- **Real-time usage metrics**
- **API key management**
- **Security configuration**
- **Billing & invoices**
- **Audit logs**
- **Compliance reports**

### Portal URL
```
https://dashboard.aegis.ai

Features:
- SSO integration (SAML, OAuth2)
- Role-based access control
- Custom dashboards
- API playground
- Documentation
```

---

## 💳 BILLING INTEGRATION

### Stripe Integration
```python
# Automatic usage-based billing
aegis.billing.configure({
    "provider": "stripe",
    "customer_id": "cus_xxx",
    "subscription_id": "sub_yyy",
    "billing_cycle": "monthly",
    "payment_method": "invoice"  # or "card"
})

# Usage automatically tracked and billed
result = aegis.protect(data)  # Counted toward usage
```

### Usage Alerts
```python
# Set usage alerts
aegis.billing.set_alerts([
    {"threshold": "80%", "action": "email"},
    {"threshold": "100%", "action": "throttle"},
    {"threshold": "150%", "action": "upgrade_plan"}
])
```

---

## 🆘 SUPPORT TIERS

### Community (Free tier)
- Documentation access
- Community forum
- 48-hour email response

### Standard ($99/month)
- Email support (24-hour SLA)
- Technical documentation
- Monthly webinars

### Priority ($499/month)
- 4-hour support SLA
- Dedicated Slack channel
- Quarterly business reviews

### Enterprise (Custom)
- 1-hour SLA
- Dedicated support engineer
- 24/7 phone support
- Custom training

---

## 📞 CONTACT

**Security Team**
security@aegis.ai

**Enterprise Sales**
enterprise@aegis.ai

**24/7 Support Hotline** (Enterprise only)
+1-800-AEGIS-AI

**Bug Bounty Program**
https://aegis.ai/security/bugbounty