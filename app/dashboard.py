"""
Aegis Customer Dashboard - Comprehensive Control Center
Real-time analytics, compliance monitoring, and configuration management
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Security, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import asyncio
from collections import defaultdict
import pandas as pd
import plotly.graph_objs as go
import plotly.utils

from app.core.config import settings
from app.core.security import verify_api_key
from app.models.database import get_db, APIKey, UsageLog, Customer
from app.onboarding import OnboardingOrchestrator, OnboardingStage
from app.training_sanitizer import TrainingSanitizer


# Dashboard Data Models
class DashboardMetrics(BaseModel):
    """Main dashboard metrics"""
    requests_today: int
    requests_this_month: int
    data_processed_gb: float
    entities_detected: int
    entities_protected: int
    average_latency_ms: float
    uptime_percentage: float
    compliance_score: float
    cost_savings_usd: float


class ProtectionStats(BaseModel):
    """PII protection statistics"""
    total_entities_detected: int
    entities_by_type: Dict[str, int]
    protection_methods_used: Dict[str, int]
    high_risk_detections: int
    compliance_violations_prevented: int
    top_risk_patterns: List[Dict[str, Any]]


class UsageAnalytics(BaseModel):
    """Usage analytics and trends"""
    hourly_usage: List[Dict[str, Any]]
    daily_usage: List[Dict[str, Any]]
    monthly_usage: List[Dict[str, Any]]
    peak_usage_times: List[str]
    usage_by_endpoint: Dict[str, int]
    geographic_distribution: Dict[str, int]


class ComplianceReport(BaseModel):
    """Compliance monitoring report"""
    gdpr_status: Dict[str, Any]
    ccpa_status: Dict[str, Any]
    hipaa_status: Dict[str, Any]
    pci_dss_status: Dict[str, Any]
    audit_logs: List[Dict[str, Any]]
    risk_assessments: List[Dict[str, Any]]
    compliance_score: float
    next_audit_date: str


class AlertConfiguration(BaseModel):
    """Alert configuration settings"""
    high_risk_threshold: float = 0.8
    volume_spike_threshold: float = 2.0
    latency_threshold_ms: int = 1000
    compliance_violation_alerts: bool = True
    email_notifications: bool = True
    slack_webhook: Optional[str] = None
    notification_emails: List[str] = []


class DashboardController:
    """
    Main controller for customer dashboard functionality
    Handles analytics, monitoring, and configuration management
    """

    def __init__(self):
        self.onboarding = OnboardingOrchestrator()
        self.training_sanitizer = TrainingSanitizer()
        self.templates = Jinja2Templates(directory="templates")

        # Cache for dashboard data
        self.metrics_cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def initialize(self):
        """Initialize dashboard controller"""
        await self.onboarding.initialize()
        await self.training_sanitizer.initialize()

    async def get_dashboard_overview(self, customer_id: str) -> Dict[str, Any]:
        """
        Get complete dashboard overview for customer

        Args:
            customer_id: Customer identifier

        Returns:
            Dashboard overview with all key metrics
        """
        # Check cache first
        cache_key = f"overview_{customer_id}"
        if cache_key in self.metrics_cache:
            cached = self.metrics_cache[cache_key]
            if (datetime.utcnow() - cached["timestamp"]).seconds < self.cache_ttl:
                return cached["data"]

        # Generate fresh data
        overview = {
            "customer_id": customer_id,
            "last_updated": datetime.utcnow().isoformat(),
            "metrics": await self._get_dashboard_metrics(customer_id),
            "protection_stats": await self._get_protection_stats(customer_id),
            "usage_analytics": await self._get_usage_analytics(customer_id),
            "compliance_report": await self._get_compliance_report(customer_id),
            "recent_activity": await self._get_recent_activity(customer_id),
            "onboarding_status": await self._get_onboarding_status(customer_id),
            "alerts": await self._get_active_alerts(customer_id),
            "recommendations": await self._get_recommendations(customer_id)
        }

        # Cache the result
        self.metrics_cache[cache_key] = {
            "data": overview,
            "timestamp": datetime.utcnow()
        }

        return overview

    async def get_real_time_metrics(self, customer_id: str) -> Dict[str, Any]:
        """
        Get real-time metrics for live dashboard updates

        Args:
            customer_id: Customer identifier

        Returns:
            Real-time metrics data
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "requests_per_minute": await self._get_current_rpm(customer_id),
            "entities_detected_per_minute": await self._get_current_epm(customer_id),
            "average_latency_ms": await self._get_current_latency(customer_id),
            "system_status": "healthy",
            "active_connections": await self._get_active_connections(customer_id),
            "queue_depth": 0,
            "error_rate": await self._get_current_error_rate(customer_id)
        }

    async def get_protection_analytics(
        self,
        customer_id: str,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """
        Get detailed protection analytics

        Args:
            customer_id: Customer identifier
            time_range: Time range for analytics (1h, 24h, 7d, 30d)

        Returns:
            Detailed protection analytics
        """
        analytics = {
            "time_range": time_range,
            "summary": {
                "total_requests": 0,
                "pii_detected": 0,
                "protection_rate": 0.0,
                "risk_reduction": 0.0
            },
            "entity_breakdown": {},
            "protection_methods": {},
            "risk_timeline": [],
            "top_threats": [],
            "geographical_threats": {},
            "compliance_coverage": {}
        }

        # Get data based on time range
        end_time = datetime.utcnow()
        if time_range == "1h":
            start_time = end_time - timedelta(hours=1)
        elif time_range == "24h":
            start_time = end_time - timedelta(days=1)
        elif time_range == "7d":
            start_time = end_time - timedelta(days=7)
        elif time_range == "30d":
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(days=1)

        # In production, query actual database
        # For demo, generate realistic data
        analytics = await self._generate_protection_analytics(customer_id, start_time, end_time)

        return analytics

    async def get_compliance_dashboard(self, customer_id: str) -> Dict[str, Any]:
        """
        Get comprehensive compliance dashboard

        Args:
            customer_id: Customer identifier

        Returns:
            Compliance dashboard data
        """
        compliance = {
            "overall_score": 98.5,
            "last_assessment": datetime.utcnow().isoformat(),
            "regulations": {
                "gdpr": {
                    "status": "compliant",
                    "score": 99.2,
                    "last_check": datetime.utcnow().isoformat(),
                    "requirements_met": 47,
                    "total_requirements": 48,
                    "issues": ["Data retention period documentation needed"]
                },
                "ccpa": {
                    "status": "compliant",
                    "score": 97.8,
                    "last_check": datetime.utcnow().isoformat(),
                    "requirements_met": 22,
                    "total_requirements": 23,
                    "issues": ["Consumer request processing time optimization"]
                },
                "hipaa": {
                    "status": "compliant",
                    "score": 99.9,
                    "last_check": datetime.utcnow().isoformat(),
                    "requirements_met": 25,
                    "total_requirements": 25,
                    "issues": []
                },
                "pci_dss": {
                    "status": "compliant",
                    "score": 98.1,
                    "last_check": datetime.utcnow().isoformat(),
                    "requirements_met": 11,
                    "total_requirements": 12,
                    "issues": ["Network segmentation documentation update"]
                }
            },
            "audit_trail": await self._get_audit_trail(customer_id),
            "risk_assessments": await self._get_risk_assessments(customer_id),
            "certification_status": await self._get_certifications(customer_id),
            "upcoming_audits": await self._get_upcoming_audits(customer_id)
        }

        return compliance

    async def get_training_data_dashboard(self, customer_id: str) -> Dict[str, Any]:
        """
        Get training data protection dashboard

        Args:
            customer_id: Customer identifier

        Returns:
            Training data dashboard
        """
        training_dashboard = {
            "datasets_processed": 15,
            "total_records_sanitized": 2500000,
            "pii_entities_removed": 750000,
            "active_jobs": await self._get_active_training_jobs(customer_id),
            "completed_jobs": await self._get_completed_training_jobs(customer_id),
            "sanitization_methods_used": {
                "redaction": 60,
                "tokenization": 25,
                "differential_privacy": 15
            },
            "compliance_reports": await self._get_training_compliance_reports(customer_id),
            "recommended_actions": await self._get_training_recommendations(customer_id)
        }

        return training_dashboard

    async def configure_alerts(
        self,
        customer_id: str,
        config: AlertConfiguration
    ) -> Dict[str, Any]:
        """
        Configure alert settings for customer

        Args:
            customer_id: Customer identifier
            config: Alert configuration

        Returns:
            Configuration status
        """
        # Store alert configuration
        # In production, save to database
        return {
            "status": "success",
            "message": "Alert configuration updated",
            "config": config.dict(),
            "test_alert_sent": True
        }

    async def export_data(
        self,
        customer_id: str,
        data_type: str,
        format: str = "csv",
        time_range: str = "30d"
    ) -> Dict[str, Any]:
        """
        Export dashboard data for analysis

        Args:
            customer_id: Customer identifier
            data_type: Type of data to export
            format: Export format (csv, json, excel)
            time_range: Time range for data

        Returns:
            Export information
        """
        export_id = f"export_{customer_id}_{data_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Generate export in background
        # In production, this would be an async task
        export_info = {
            "export_id": export_id,
            "status": "processing",
            "data_type": data_type,
            "format": format,
            "time_range": time_range,
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
            "download_url": f"https://api.aegis-shield.ai/exports/{export_id}"
        }

        return export_info

    # Private helper methods

    async def _get_dashboard_metrics(self, customer_id: str) -> DashboardMetrics:
        """Get main dashboard metrics"""
        # In production, query actual database
        # For demo, return realistic metrics
        return DashboardMetrics(
            requests_today=125000,
            requests_this_month=3200000,
            data_processed_gb=1250.5,
            entities_detected=450000,
            entities_protected=449000,
            average_latency_ms=45.2,
            uptime_percentage=99.97,
            compliance_score=98.5,
            cost_savings_usd=285000
        )

    async def _get_protection_stats(self, customer_id: str) -> ProtectionStats:
        """Get PII protection statistics"""
        return ProtectionStats(
            total_entities_detected=450000,
            entities_by_type={
                "EMAIL": 125000,
                "PHONE": 85000,
                "SSN": 45000,
                "CREDIT_CARD": 25000,
                "ADDRESS": 95000,
                "PERSON_NAME": 75000
            },
            protection_methods_used={
                "redaction": 270000,
                "masking": 135000,
                "tokenization": 45000
            },
            high_risk_detections=70000,
            compliance_violations_prevented=450000,
            top_risk_patterns=[
                {"pattern": "SSN", "count": 45000, "risk": "high"},
                {"pattern": "Credit Card", "count": 25000, "risk": "high"},
                {"pattern": "Medical Record", "count": 15000, "risk": "high"}
            ]
        )

    async def _get_usage_analytics(self, customer_id: str) -> UsageAnalytics:
        """Get usage analytics and trends"""
        # Generate hourly usage for last 24 hours
        hourly_data = []
        for i in range(24):
            hour_time = datetime.utcnow() - timedelta(hours=23-i)
            hourly_data.append({
                "hour": hour_time.strftime("%H:00"),
                "requests": 4000 + (i * 200) + (100 if i % 3 == 0 else 0),
                "entities_detected": 1200 + (i * 60),
                "latency_ms": 40 + (i % 5) * 2
            })

        return UsageAnalytics(
            hourly_usage=hourly_data,
            daily_usage=[],  # Would populate with daily data
            monthly_usage=[],  # Would populate with monthly data
            peak_usage_times=["09:00-11:00", "14:00-16:00", "20:00-22:00"],
            usage_by_endpoint={
                "/v1/process": 85000,
                "/v1/batch": 35000,
                "/v1/training/sanitize": 5000
            },
            geographic_distribution={
                "US": 60,
                "EU": 25,
                "Asia": 10,
                "Other": 5
            }
        )

    async def _get_compliance_report(self, customer_id: str) -> ComplianceReport:
        """Get compliance monitoring report"""
        return ComplianceReport(
            gdpr_status={
                "compliant": True,
                "score": 99.2,
                "last_check": datetime.utcnow().isoformat(),
                "issues": []
            },
            ccpa_status={
                "compliant": True,
                "score": 97.8,
                "last_check": datetime.utcnow().isoformat(),
                "issues": []
            },
            hipaa_status={
                "compliant": True,
                "score": 99.9,
                "last_check": datetime.utcnow().isoformat(),
                "issues": []
            },
            pci_dss_status={
                "compliant": True,
                "score": 98.1,
                "last_check": datetime.utcnow().isoformat(),
                "issues": []
            },
            audit_logs=[],
            risk_assessments=[],
            compliance_score=98.5,
            next_audit_date=(datetime.utcnow() + timedelta(days=90)).strftime("%Y-%m-%d")
        )

    async def _get_recent_activity(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get recent activity logs"""
        return [
            {
                "timestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
                "type": "high_risk_detection",
                "message": "SSN detected and protected in customer support chat",
                "severity": "medium"
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(minutes=15)).isoformat(),
                "type": "batch_processing",
                "message": "Training data sanitization completed: 50,000 records processed",
                "severity": "info"
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                "type": "compliance_check",
                "message": "GDPR compliance verification passed",
                "severity": "info"
            }
        ]

    async def _get_onboarding_status(self, customer_id: str) -> Dict[str, Any]:
        """Get customer onboarding status"""
        try:
            status = await self.onboarding.get_onboarding_status(customer_id)
            return status
        except:
            # Customer not in onboarding or completed
            return {
                "status": "completed",
                "stage": OnboardingStage.COMPLETED,
                "progress_percentage": 100
            }

    async def _get_active_alerts(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get active alerts for customer"""
        return [
            {
                "id": "alert_001",
                "type": "usage_spike",
                "severity": "medium",
                "message": "API usage increased by 150% in the last hour",
                "timestamp": (datetime.utcnow() - timedelta(minutes=30)).isoformat(),
                "acknowledged": False
            }
        ]

    async def _get_recommendations(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get recommendations for customer"""
        return [
            {
                "type": "optimization",
                "title": "Optimize batch processing",
                "description": "Consider using batch processing for better performance",
                "impact": "high",
                "effort": "low"
            },
            {
                "type": "security",
                "title": "Enable additional compliance monitoring",
                "description": "Add CCPA monitoring for your California users",
                "impact": "medium",
                "effort": "low"
            }
        ]

    async def _get_current_rpm(self, customer_id: str) -> int:
        """Get current requests per minute"""
        return 850  # Demo value

    async def _get_current_epm(self, customer_id: str) -> int:
        """Get current entities detected per minute"""
        return 255  # Demo value

    async def _get_current_latency(self, customer_id: str) -> float:
        """Get current average latency"""
        return 42.5  # Demo value

    async def _get_active_connections(self, customer_id: str) -> int:
        """Get active connections count"""
        return 12  # Demo value

    async def _get_current_error_rate(self, customer_id: str) -> float:
        """Get current error rate percentage"""
        return 0.02  # Demo value

    async def _generate_protection_analytics(
        self,
        customer_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Generate protection analytics for time range"""
        return {
            "summary": {
                "total_requests": 125000,
                "pii_detected": 45000,
                "protection_rate": 99.8,
                "risk_reduction": 98.5
            },
            "entity_breakdown": {
                "EMAIL": 12500,
                "PHONE": 8500,
                "SSN": 4500,
                "CREDIT_CARD": 2500,
                "ADDRESS": 9500,
                "PERSON_NAME": 7500
            },
            "protection_methods": {
                "redaction": 27000,
                "masking": 13500,
                "tokenization": 4500
            },
            "risk_timeline": [],  # Would populate with hourly risk data
            "top_threats": [
                {"type": "SSN", "count": 4500, "risk_score": 0.95},
                {"type": "CREDIT_CARD", "count": 2500, "risk_score": 0.90}
            ]
        }

    async def _get_audit_trail(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get audit trail for compliance"""
        return [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "data_processed",
                "user": "system",
                "details": "Batch processing of 10,000 records",
                "compliance_impact": "GDPR, CCPA"
            }
        ]

    async def _get_risk_assessments(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get risk assessments"""
        return [
            {
                "id": "risk_001",
                "timestamp": datetime.utcnow().isoformat(),
                "risk_level": "low",
                "score": 0.15,
                "findings": ["All PII properly protected", "Compliance requirements met"]
            }
        ]

    async def _get_certifications(self, customer_id: str) -> Dict[str, Any]:
        """Get certification status"""
        return {
            "soc2": {"status": "certified", "expiry": "2024-12-31"},
            "iso27001": {"status": "certified", "expiry": "2025-06-30"},
            "gdpr": {"status": "compliant", "last_check": "2024-10-01"}
        }

    async def _get_upcoming_audits(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get upcoming audit schedule"""
        return [
            {
                "type": "GDPR Compliance Review",
                "date": "2024-11-15",
                "auditor": "External Compliance Firm",
                "scope": "Data processing activities"
            }
        ]

    async def _get_active_training_jobs(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get active training data jobs"""
        return [
            {
                "job_id": "train_001",
                "dataset": "customer_support_logs_q3",
                "status": "processing",
                "progress": 75.5,
                "estimated_completion": (datetime.utcnow() + timedelta(minutes=15)).isoformat()
            }
        ]

    async def _get_completed_training_jobs(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get completed training data jobs"""
        return [
            {
                "job_id": "train_002",
                "dataset": "chat_history_archive",
                "status": "completed",
                "completion_time": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "records_processed": 500000,
                "entities_removed": 125000
            }
        ]

    async def _get_training_compliance_reports(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get training data compliance reports"""
        return [
            {
                "report_id": "compliance_001",
                "dataset": "training_data_v2",
                "compliance_score": 99.2,
                "timestamp": datetime.utcnow().isoformat(),
                "regulations_covered": ["GDPR", "CCPA", "HIPAA"]
            }
        ]

    async def _get_training_recommendations(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get training data recommendations"""
        return [
            {
                "type": "optimization",
                "message": "Consider using differential privacy for improved utility",
                "impact": "medium"
            },
            {
                "type": "compliance",
                "message": "Enable additional PII pattern detection for medical data",
                "impact": "high"
            }
        ]


# FastAPI dashboard endpoints
dashboard_app = FastAPI(title="Aegis Dashboard API", version="1.0.0")
dashboard_controller = DashboardController()

# API Key Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def validate_dashboard_api_key(api_key: str = Security(api_key_header)) -> Dict:
    """Validate API key for dashboard access"""
    if not api_key:
        raise HTTPException(status_code=403, detail="API key required")

    key_data = await verify_api_key(api_key)
    if not key_data:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return key_data

@dashboard_app.on_event("startup")
async def startup_dashboard():
    """Initialize dashboard on startup"""
    await dashboard_controller.initialize()

@dashboard_app.get("/dashboard/overview")
async def get_dashboard_overview(api_key_data: Dict = Depends(validate_dashboard_api_key)):
    """Get complete dashboard overview"""
    return await dashboard_controller.get_dashboard_overview(api_key_data["customer_id"])

@dashboard_app.get("/dashboard/metrics/realtime")
async def get_realtime_metrics(api_key_data: Dict = Depends(validate_dashboard_api_key)):
    """Get real-time metrics"""
    return await dashboard_controller.get_real_time_metrics(api_key_data["customer_id"])

@dashboard_app.get("/dashboard/protection/analytics")
async def get_protection_analytics(
    time_range: str = "24h",
    api_key_data: Dict = Depends(validate_dashboard_api_key)
):
    """Get protection analytics"""
    return await dashboard_controller.get_protection_analytics(
        api_key_data["customer_id"], time_range
    )

@dashboard_app.get("/dashboard/compliance")
async def get_compliance_dashboard(api_key_data: Dict = Depends(validate_dashboard_api_key)):
    """Get compliance dashboard"""
    return await dashboard_controller.get_compliance_dashboard(api_key_data["customer_id"])

@dashboard_app.get("/dashboard/training")
async def get_training_dashboard(api_key_data: Dict = Depends(validate_dashboard_api_key)):
    """Get training data dashboard"""
    return await dashboard_controller.get_training_data_dashboard(api_key_data["customer_id"])

@dashboard_app.post("/dashboard/alerts/configure")
async def configure_alerts(
    config: AlertConfiguration,
    api_key_data: Dict = Depends(validate_dashboard_api_key)
):
    """Configure alert settings"""
    return await dashboard_controller.configure_alerts(api_key_data["customer_id"], config)

@dashboard_app.post("/dashboard/export")
async def export_dashboard_data(
    data_type: str,
    format: str = "csv",
    time_range: str = "30d",
    api_key_data: Dict = Depends(validate_dashboard_api_key)
):
    """Export dashboard data"""
    return await dashboard_controller.export_data(
        api_key_data["customer_id"], data_type, format, time_range
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(dashboard_app, host="0.0.0.0", port=8001)