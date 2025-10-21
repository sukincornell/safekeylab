"""
Aegis API Server - Production-Ready Privacy Shield for AI
"""

from fastapi import FastAPI, HTTPException, Depends, Security, Request, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import hmac
import time
import uuid
import asyncio
from functools import lru_cache

from app.core.config import settings
from app.core.security import verify_api_key, create_access_token
from app.core.rate_limiter import RateLimiter
from app.services.pii_detector import PIIDetector
from app.services.anonymizer import DataAnonymizer
from app.models.database import get_db, APIKey, UsageLog
from app.monitoring import metrics, log_request
from app.training_sanitizer import TrainingSanitizer, TrainingConfig, SanitizationResult
from app.onboarding import OnboardingOrchestrator, CustomerProfile, OnboardingState, IntegrationType
from app.dashboard import DashboardController
from app.multimodal_api import router as multimodal_router

# Initialize FastAPI app
app = FastAPI(
    title="Aegis API",
    description="Enterprise Privacy Shield for AI Systems - Now with Multimodal Support",
    version="2.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Include multimodal router
app.include_router(multimodal_router)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Initialize services
pii_detector = PIIDetector()
anonymizer = DataAnonymizer()
rate_limiter = RateLimiter()
training_sanitizer = TrainingSanitizer()
onboarding = OnboardingOrchestrator()
dashboard = DashboardController()

# API Key Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Request/Response Models
class PrivacyMethod(str, Enum):
    REDACTION = "redaction"
    MASKING = "masking"
    TOKENIZATION = "tokenization"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    K_ANONYMITY = "k_anonymity"
    SYNTHETIC = "synthetic"
    AUTO = "auto"

class DataFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    STRUCTURED = "structured"

class ProcessRequest(BaseModel):
    data: Union[str, Dict, List]
    method: PrivacyMethod = PrivacyMethod.AUTO
    format: DataFormat = DataFormat.TEXT
    detect_only: bool = False
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    custom_patterns: Optional[List[Dict[str, str]]] = None
    return_metrics: bool = True

    class Config:
        example = {
            "data": "John Smith's email is john@example.com and his SSN is 123-45-6789",
            "method": "redaction",
            "format": "text",
            "confidence_threshold": 0.9
        }

class DetectedEntity(BaseModel):
    text: str
    type: str
    start: int
    end: int
    confidence: float
    replacement: Optional[str] = None

class ProcessResponse(BaseModel):
    request_id: str
    timestamp: datetime
    original_length: int
    processed_data: Union[str, Dict, List]
    entities_detected: List[DetectedEntity]
    processing_time_ms: float
    method_used: str
    compliance: Dict[str, bool]
    risk_score: float

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    requests_processed: int

# Training Data Models
class TrainingDataRequest(BaseModel):
    data: Union[List[Dict], str]  # File path or data directly
    method: PrivacyMethod = PrivacyMethod.REDACTION
    preserve_context: bool = True
    preserve_format: bool = True
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    output_format: str = "same"
    batch_size: int = 1000

    class Config:
        example = {
            "data": [
                {"message": "Hi, my email is john@example.com"},
                {"message": "My SSN is 123-45-6789"}
            ],
            "method": "redaction",
            "preserve_context": True
        }

class TrainingJobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    estimated_completion: Optional[str] = None
    result: Optional[SanitizationResult] = None

# Onboarding Models
class QuickStartRequest(BaseModel):
    company_name: str
    email: str
    use_case: str
    integration_type: str = "openai"
    industry: Optional[str] = "general"

class OnboardingActionRequest(BaseModel):
    action: str
    data: Optional[Dict] = None

# Dashboard Models
class DashboardRequest(BaseModel):
    time_range: str = "24h"
    metrics: List[str] = ["all"]

# Dependency for API key validation
async def validate_api_key(api_key: str = Security(api_key_header)) -> Dict:
    if not api_key:
        raise HTTPException(status_code=403, detail="API key required")

    key_data = await verify_api_key(api_key)
    if not key_data:
        raise HTTPException(status_code=403, detail="Invalid API key")

    # Check rate limits
    if not await rate_limiter.check_limit(key_data["customer_id"]):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return key_data

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        uptime_seconds=time.time() - app.state.start_time,
        requests_processed=metrics.requests_total
    )

# Main processing endpoint
@app.post("/v1/process", response_model=ProcessResponse)
async def process_data(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
    api_key_data: Dict = Depends(validate_api_key)
):
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        # Detect PII entities
        entities = await pii_detector.detect(
            request.data,
            format=request.format,
            confidence_threshold=request.confidence_threshold,
            custom_patterns=request.custom_patterns
        )

        # If detect_only, return entities without processing
        if request.detect_only:
            processed_data = request.data
        else:
            # Apply privacy protection
            processed_data = await anonymizer.process(
                request.data,
                entities=entities,
                method=request.method,
                format=request.format
            )

        # Calculate metrics
        processing_time = (time.time() - start_time) * 1000
        risk_score = calculate_risk_score(entities)
        compliance = check_compliance(entities, processed_data)

        # Log usage in background
        background_tasks.add_task(
            log_usage,
            api_key_data["customer_id"],
            request_id,
            len(str(request.data)),
            len(entities),
            processing_time
        )

        # Update metrics
        metrics.requests_total += 1
        metrics.entities_detected += len(entities)

        return ProcessResponse(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            original_length=len(str(request.data)),
            processed_data=processed_data,
            entities_detected=[
                DetectedEntity(
                    text=e["text"],
                    type=e["type"],
                    start=e["start"],
                    end=e["end"],
                    confidence=e["confidence"],
                    replacement=e.get("replacement")
                ) for e in entities
            ],
            processing_time_ms=processing_time,
            method_used=request.method if request.method != PrivacyMethod.AUTO else "redaction",
            compliance=compliance,
            risk_score=risk_score
        )

    except Exception as e:
        metrics.errors_total += 1
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Batch processing endpoint
@app.post("/v1/batch")
async def batch_process(
    requests: List[ProcessRequest],
    background_tasks: BackgroundTasks,
    api_key_data: Dict = Depends(validate_api_key)
):
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 items per batch")

    results = []
    for req in requests:
        result = await process_data(req, background_tasks, api_key_data)
        results.append(result)

    return {"batch_id": str(uuid.uuid4()), "results": results}

# Detection patterns endpoint
@app.get("/v1/patterns")
async def get_detection_patterns(api_key_data: Dict = Depends(validate_api_key)):
    return {
        "default_patterns": pii_detector.get_patterns(),
        "supported_entities": [
            "EMAIL", "PHONE", "SSN", "CREDIT_CARD", "IP_ADDRESS",
            "PERSON_NAME", "ADDRESS", "DATE_OF_BIRTH", "PASSPORT",
            "DRIVER_LICENSE", "BANK_ACCOUNT", "API_KEY", "PASSWORD"
        ],
        "custom_patterns_supported": True
    }

# Usage statistics endpoint
@app.get("/v1/usage")
async def get_usage_stats(api_key_data: Dict = Depends(validate_api_key)):
    # Get usage from database
    usage = await get_customer_usage(api_key_data["customer_id"])
    return {
        "customer_id": api_key_data["customer_id"],
        "current_month": {
            "requests": usage["requests"],
            "data_processed_mb": usage["data_mb"],
            "entities_detected": usage["entities"],
            "average_latency_ms": usage["avg_latency"]
        },
        "quota": {
            "requests_limit": api_key_data["requests_limit"],
            "requests_remaining": api_key_data["requests_limit"] - usage["requests"]
        }
    }

# Compliance report endpoint
@app.get("/v1/compliance/{request_id}")
async def get_compliance_report(
    request_id: str,
    api_key_data: Dict = Depends(validate_api_key)
):
    # Retrieve processing details from cache/database
    report = await get_processing_report(request_id)
    if not report:
        raise HTTPException(status_code=404, detail="Request not found")

    return {
        "request_id": request_id,
        "gdpr_compliant": report["gdpr"],
        "ccpa_compliant": report["ccpa"],
        "hipaa_compliant": report["hipaa"],
        "pci_dss_compliant": report["pci"],
        "audit_trail": report["audit"],
        "certificate_url": f"https://api.aegis-shield.ai/certificates/{request_id}"
    }

# ==================== TRAINING DATA PROTECTION ENDPOINTS ====================

@app.post("/v1/training/sanitize", response_model=TrainingJobStatusResponse)
async def sanitize_training_data(
    request: TrainingDataRequest,
    background_tasks: BackgroundTasks,
    api_key_data: Dict = Depends(validate_api_key)
):
    """
    Sanitize training data for safe model training
    Supports batch processing of large datasets
    """
    try:
        # Create training configuration
        config = TrainingConfig(
            method=request.method.value,
            preserve_context=request.preserve_context,
            preserve_format=request.preserve_format,
            confidence_threshold=request.confidence_threshold,
            output_format=request.output_format,
            batch_size=request.batch_size
        )

        # Start sanitization job
        result = await training_sanitizer.sanitize_dataset(
            data=request.data,
            config=config
        )

        # Log usage
        background_tasks.add_task(
            log_training_usage,
            api_key_data["customer_id"],
            result.job_id,
            result.original_size,
            result.entities_detected,
            result.processing_time_seconds
        )

        return TrainingJobStatusResponse(
            job_id=result.job_id,
            status="completed",
            progress=100.0,
            result=result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training sanitization error: {str(e)}")

@app.post("/v1/training/chat-logs")
async def sanitize_chat_logs(
    chat_logs: List[Dict],
    method: PrivacyMethod = PrivacyMethod.REDACTION,
    preserve_context: bool = True,
    api_key_data: Dict = Depends(validate_api_key)
):
    """
    Sanitize chat conversation logs for training
    Preserves conversation flow while removing PII
    """
    try:
        config = TrainingConfig(
            method=method.value,
            preserve_context=preserve_context
        )

        sanitized_logs = await training_sanitizer.sanitize_chat_logs(chat_logs, config)

        return {
            "status": "success",
            "original_conversations": len(chat_logs),
            "sanitized_conversations": len(sanitized_logs),
            "sanitized_data": sanitized_logs,
            "method_used": method.value
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat log sanitization error: {str(e)}")

@app.post("/v1/training/synthetic")
async def create_synthetic_dataset(
    original_data: List[Dict],
    size_multiplier: float = 1.0,
    method: str = "synthetic",
    api_key_data: Dict = Depends(validate_api_key)
):
    """
    Generate synthetic training data that preserves patterns but removes PII
    Safe for model training without privacy risks
    """
    try:
        config = TrainingConfig(method=method)

        synthetic_data = await training_sanitizer.create_synthetic_dataset(
            original_data=original_data,
            size_multiplier=size_multiplier,
            config=config
        )

        return {
            "status": "success",
            "original_size": len(original_data),
            "synthetic_size": len(synthetic_data),
            "size_multiplier": size_multiplier,
            "synthetic_data": synthetic_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthetic data generation error: {str(e)}")

@app.get("/v1/training/jobs/{job_id}")
async def get_training_job_status(
    job_id: str,
    api_key_data: Dict = Depends(validate_api_key)
):
    """Get status of a training data sanitization job"""
    try:
        job_status = await training_sanitizer.get_job_status(job_id)

        if job_status["status"] == "not_found":
            raise HTTPException(status_code=404, detail="Job not found")

        return job_status

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job status error: {str(e)}")

@app.delete("/v1/training/jobs/{job_id}")
async def cancel_training_job(
    job_id: str,
    api_key_data: Dict = Depends(validate_api_key)
):
    """Cancel a running training data sanitization job"""
    try:
        success = await training_sanitizer.cancel_job(job_id)

        if not success:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled")

        return {"status": "cancelled", "job_id": job_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job cancellation error: {str(e)}")

# ==================== ONBOARDING ENDPOINTS ====================

@app.post("/v1/onboarding/quick-start", response_model=OnboardingState)
async def quick_start_onboarding(request: QuickStartRequest):
    """
    Quick start onboarding for immediate access
    Get API key and integration code in minutes
    """
    try:
        profile = CustomerProfile(
            company_name=request.company_name,
            industry=request.industry,
            use_case=request.use_case,
            data_types=["text"],
            compliance_requirements=["GDPR"],
            integration_type=IntegrationType(request.integration_type),
            expected_volume="medium",
            technical_contact={"email": request.email},
            business_contact={"email": request.email}
        )

        state = await onboarding.start_onboarding(profile, source="api")

        return state

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Onboarding error: {str(e)}")

@app.post("/v1/onboarding/{customer_id}/continue")
async def continue_onboarding(
    customer_id: str,
    request: OnboardingActionRequest
):
    """Continue onboarding process with customer action"""
    try:
        state = await onboarding.continue_onboarding(
            customer_id=customer_id,
            action=request.action,
            data=request.data
        )

        return state

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Onboarding continuation error: {str(e)}")

@app.get("/v1/onboarding/{customer_id}/status")
async def get_onboarding_status(customer_id: str):
    """Get complete onboarding status and next steps"""
    try:
        status = await onboarding.get_onboarding_status(customer_id)
        return status

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Onboarding status error: {str(e)}")

@app.get("/v1/onboarding/{customer_id}/integration-code")
async def get_integration_code(
    customer_id: str,
    integration_type: IntegrationType
):
    """Get personalized integration code for customer"""
    try:
        code = await onboarding.get_integration_code(customer_id, integration_type)

        return {
            "integration_type": integration_type.value,
            "code": code,
            "documentation": f"https://docs.aegis-shield.ai/{integration_type.value}",
            "support": "support@aegis-shield.ai"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Integration code error: {str(e)}")

@app.post("/v1/onboarding/{customer_id}/analyze-data")
async def analyze_sample_data(
    customer_id: str,
    sample_data: List[str]
):
    """
    Analyze customer's sample data for automatic configuration
    Returns optimized settings and recommendations
    """
    try:
        analysis = await onboarding.auto_analyze_sample_data(customer_id, sample_data)

        return {
            "customer_id": customer_id,
            "analysis": analysis,
            "auto_configured": True,
            "next_steps": ["Test integration", "Go live"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data analysis error: {str(e)}")

# ==================== DASHBOARD ENDPOINTS ====================

@app.get("/v1/dashboard/overview")
async def get_dashboard_overview(api_key_data: Dict = Depends(validate_api_key)):
    """Get complete dashboard overview with all metrics"""
    try:
        overview = await dashboard.get_dashboard_overview(api_key_data["customer_id"])
        return overview

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")

@app.get("/v1/dashboard/metrics/realtime")
async def get_realtime_metrics(api_key_data: Dict = Depends(validate_api_key)):
    """Get real-time metrics for live dashboard updates"""
    try:
        metrics = await dashboard.get_real_time_metrics(api_key_data["customer_id"])
        return metrics

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Real-time metrics error: {str(e)}")

@app.get("/v1/dashboard/compliance")
async def get_compliance_dashboard(api_key_data: Dict = Depends(validate_api_key)):
    """Get comprehensive compliance dashboard"""
    try:
        compliance = await dashboard.get_compliance_dashboard(api_key_data["customer_id"])
        return compliance

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compliance dashboard error: {str(e)}")

@app.get("/v1/dashboard/training")
async def get_training_dashboard(api_key_data: Dict = Depends(validate_api_key)):
    """Get training data protection dashboard"""
    try:
        training_dash = await dashboard.get_training_data_dashboard(api_key_data["customer_id"])
        return training_dash

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training dashboard error: {str(e)}")

# Helper functions
def calculate_risk_score(entities: List[Dict]) -> float:
    """Calculate privacy risk score based on detected entities"""
    if not entities:
        return 0.0

    high_risk_types = ["SSN", "CREDIT_CARD", "PASSPORT", "BANK_ACCOUNT"]
    medium_risk_types = ["EMAIL", "PHONE", "DATE_OF_BIRTH", "DRIVER_LICENSE"]

    score = 0.0
    for entity in entities:
        if entity["type"] in high_risk_types:
            score += 0.3
        elif entity["type"] in medium_risk_types:
            score += 0.1
        else:
            score += 0.05

    return min(score, 1.0)

def check_compliance(entities: List[Dict], processed_data: Any) -> Dict[str, bool]:
    """Check compliance with various regulations"""
    has_pii = len(entities) > 0
    all_removed = not any(e["text"] in str(processed_data) for e in entities)

    return {
        "gdpr": all_removed or not has_pii,
        "ccpa": all_removed or not has_pii,
        "hipaa": all_removed or not any(e["type"] in ["SSN", "MEDICAL_RECORD"] for e in entities),
        "pci_dss": all_removed or not any(e["type"] == "CREDIT_CARD" for e in entities)
    }

async def log_usage(customer_id: str, request_id: str, data_size: int, entities: int, latency: float):
    """Log API usage to database"""
    # Implementation would save to database
    pass

async def log_training_usage(customer_id: str, job_id: str, data_size: int, entities: int, processing_time: float):
    """Log training data usage to database"""
    # Implementation would save training usage to database
    pass

async def get_customer_usage(customer_id: str) -> Dict:
    """Get customer usage statistics"""
    # Implementation would query database
    return {
        "requests": 150000,
        "data_mb": 1024,
        "entities": 450000,
        "avg_latency": 45.2
    }

async def get_processing_report(request_id: str) -> Optional[Dict]:
    """Retrieve processing report from cache/database"""
    # Implementation would query cache/database
    return {
        "gdpr": True,
        "ccpa": True,
        "hipaa": True,
        "pci": True,
        "audit": []
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    app.state.start_time = time.time()
    # Initialize database connections
    # Load ML models and services
    await pii_detector.initialize()
    await anonymizer.initialize()
    await training_sanitizer.initialize()
    await onboarding.initialize()
    await dashboard.initialize()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    # Clean up resources
    await pii_detector.cleanup()
    await anonymizer.cleanup()
    # Note: training_sanitizer, onboarding, and dashboard don't need explicit cleanup

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)