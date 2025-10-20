"""
Monitoring and Metrics for Aegis API
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from datetime import datetime
import time
from dataclasses import dataclass
from typing import Dict, Any
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
requests_total = Counter('aegis_requests_total', 'Total number of requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('aegis_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
active_connections = Gauge('aegis_active_connections', 'Number of active connections')
entities_detected_total = Counter('aegis_entities_detected_total', 'Total entities detected', ['entity_type'])
processing_errors = Counter('aegis_processing_errors_total', 'Total processing errors', ['error_type'])
api_keys_active = Gauge('aegis_api_keys_active', 'Number of active API keys')
data_processed_bytes = Counter('aegis_data_processed_bytes_total', 'Total bytes processed')

@dataclass
class Metrics:
    """Application metrics"""
    requests_total: int = 0
    entities_detected: int = 0
    errors_total: int = 0
    avg_latency_ms: float = 0
    active_customers: int = 0
    data_processed_gb: float = 0

# Global metrics instance
metrics = Metrics()

def log_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration_ms: float,
    customer_id: str = None,
    request_id: str = None
):
    """Log API request metrics"""
    # Update Prometheus metrics
    requests_total.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
    request_duration.labels(method=method, endpoint=endpoint).observe(duration_ms / 1000)

    # Log to application logs
    logger.info(json.dumps({
        "type": "api_request",
        "timestamp": datetime.utcnow().isoformat(),
        "method": method,
        "endpoint": endpoint,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "customer_id": customer_id,
        "request_id": request_id
    }))

def log_entity_detection(entity_type: str, count: int = 1):
    """Log entity detection metrics"""
    entities_detected_total.labels(entity_type=entity_type).inc(count)
    metrics.entities_detected += count

def log_error(error_type: str, error_message: str, customer_id: str = None):
    """Log processing errors"""
    processing_errors.labels(error_type=error_type).inc()
    metrics.errors_total += 1

    logger.error(json.dumps({
        "type": "processing_error",
        "timestamp": datetime.utcnow().isoformat(),
        "error_type": error_type,
        "error_message": error_message,
        "customer_id": customer_id
    }))

def log_data_processed(bytes_processed: int):
    """Log data processing metrics"""
    data_processed_bytes.inc(bytes_processed)
    metrics.data_processed_gb += bytes_processed / (1024 ** 3)

class PerformanceTracker:
    """Track API performance metrics"""

    def __init__(self):
        self.latencies = []
        self.error_rates = {}
        self.throughput = {}

    def record_latency(self, latency_ms: float):
        """Record request latency"""
        self.latencies.append(latency_ms)
        # Keep only last 1000 measurements
        if len(self.latencies) > 1000:
            self.latencies.pop(0)

    def get_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles"""
        if not self.latencies:
            return {"p50": 0, "p95": 0, "p99": 0}

        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)

        return {
            "p50": sorted_latencies[int(n * 0.5)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)]
        }

    def record_error(self, endpoint: str):
        """Record error for endpoint"""
        if endpoint not in self.error_rates:
            self.error_rates[endpoint] = {"errors": 0, "total": 0}

        self.error_rates[endpoint]["errors"] += 1
        self.error_rates[endpoint]["total"] += 1

    def record_success(self, endpoint: str):
        """Record successful request"""
        if endpoint not in self.error_rates:
            self.error_rates[endpoint] = {"errors": 0, "total": 0}

        self.error_rates[endpoint]["total"] += 1

    def get_error_rate(self, endpoint: str = None) -> float:
        """Calculate error rate"""
        if endpoint:
            if endpoint in self.error_rates:
                data = self.error_rates[endpoint]
                if data["total"] > 0:
                    return data["errors"] / data["total"]
            return 0

        # Overall error rate
        total_errors = sum(d["errors"] for d in self.error_rates.values())
        total_requests = sum(d["total"] for d in self.error_rates.values())

        if total_requests > 0:
            return total_errors / total_requests
        return 0

# Global performance tracker
performance = PerformanceTracker()

class HealthChecker:
    """Service health monitoring"""

    def __init__(self):
        self.checks = {}
        self.last_check = {}

    async def check_database(self) -> bool:
        """Check database connectivity"""
        try:
            from app.models.database import engine
            async with engine.connect() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def check_redis(self) -> bool:
        """Check Redis connectivity"""
        try:
            from app.core.security import get_redis
            redis = await get_redis()
            await redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def check_ml_models(self) -> bool:
        """Check ML model availability"""
        try:
            # Check if models are loaded
            # In production, verify model endpoints
            return True
        except Exception as e:
            logger.error(f"ML model health check failed: {e}")
            return False

    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        checks = {
            "database": await self.check_database(),
            "redis": await self.check_redis(),
            "ml_models": await self.check_ml_models()
        }

        all_healthy = all(checks.values())

        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks,
            "metrics": {
                "requests_total": metrics.requests_total,
                "error_rate": performance.get_error_rate(),
                "latency_p50": performance.get_percentiles()["p50"],
                "latency_p99": performance.get_percentiles()["p99"]
            }
        }

# Global health checker
health = HealthChecker()

def get_prometheus_metrics():
    """Get Prometheus metrics for scraping"""
    return generate_latest()

class AuditLogger:
    """Audit trail for compliance"""

    @staticmethod
    def log_data_access(
        customer_id: str,
        request_id: str,
        data_type: str,
        action: str,
        ip_address: str = None
    ):
        """Log data access for audit trail"""
        audit_entry = {
            "type": "data_access",
            "timestamp": datetime.utcnow().isoformat(),
            "customer_id": customer_id,
            "request_id": request_id,
            "data_type": data_type,
            "action": action,
            "ip_address": ip_address
        }

        logger.info(f"AUDIT: {json.dumps(audit_entry)}")

    @staticmethod
    def log_compliance_event(
        event_type: str,
        regulation: str,
        status: str,
        details: Dict = None
    ):
        """Log compliance-related events"""
        compliance_entry = {
            "type": "compliance",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "regulation": regulation,
            "status": status,
            "details": details or {}
        }

        logger.info(f"COMPLIANCE: {json.dumps(compliance_entry)}")

# Global audit logger
audit = AuditLogger()