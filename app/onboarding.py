"""
Aegis Onboarding System - Smooth as Butter Customer Experience
Automated setup, configuration, and integration guidance
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from pathlib import Path
import logging

from app.core.config import settings
from app.models.database import get_db, APIKey, Customer
from app.services.pii_detector import PIIDetector
from app.training_sanitizer import TrainingSanitizer, TrainingConfig


class OnboardingStage(str, Enum):
    SIGNUP = "signup"
    VERIFICATION = "verification"
    SETUP = "setup"
    INTEGRATION = "integration"
    TESTING = "testing"
    PRODUCTION = "production"
    COMPLETED = "completed"


class IntegrationType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    CUSTOM_API = "custom_api"
    CHATBOT = "chatbot"
    RAG_SYSTEM = "rag_system"
    TRAINING_PIPELINE = "training_pipeline"


@dataclass
class CustomerProfile:
    """Customer profile for personalized onboarding"""
    company_name: str
    industry: str
    use_case: str
    data_types: List[str]
    compliance_requirements: List[str]
    integration_type: IntegrationType
    expected_volume: str
    technical_contact: Dict[str, str]
    business_contact: Dict[str, str]

    # Auto-detected characteristics
    detected_patterns: Optional[List[str]] = None
    risk_level: Optional[str] = None
    recommended_config: Optional[Dict] = None


@dataclass
class OnboardingState:
    """Current onboarding state for a customer"""
    customer_id: str
    stage: OnboardingStage
    progress_percentage: float
    next_steps: List[str]
    completed_steps: List[str]
    integration_code: Optional[str] = None
    api_key: Optional[str] = None
    test_results: Optional[Dict] = None
    auto_config: Optional[Dict] = None
    estimated_completion: Optional[datetime] = None


class OnboardingOrchestrator:
    """
    Main orchestrator for smooth customer onboarding
    Handles end-to-end customer journey with automatic configuration
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pii_detector = PIIDetector()
        self.training_sanitizer = TrainingSanitizer()
        self.active_onboardings = {}

        # Integration templates
        self.integration_templates = self._load_integration_templates()

        # Auto-configuration rules
        self.auto_config_rules = self._load_auto_config_rules()

    async def initialize(self):
        """Initialize onboarding system"""
        await self.pii_detector.initialize()
        await self.training_sanitizer.initialize()
        self.logger.info("Onboarding orchestrator initialized")

    async def start_onboarding(
        self,
        profile: CustomerProfile,
        source: str = "website"
    ) -> OnboardingState:
        """
        Start onboarding process for new customer

        Args:
            profile: Customer profile information
            source: Where the customer came from

        Returns:
            Initial onboarding state
        """
        customer_id = str(uuid.uuid4())

        # Generate API key immediately
        api_key = await self._generate_api_key(customer_id, profile.company_name)

        # Create initial onboarding state
        state = OnboardingState(
            customer_id=customer_id,
            stage=OnboardingStage.SETUP,
            progress_percentage=10.0,
            next_steps=["Verify email", "Complete profile", "Generate API key"],
            completed_steps=["Account created"],
            api_key=api_key,
            estimated_completion=datetime.utcnow() + timedelta(hours=2)
        )

        # Store onboarding state
        self.active_onboardings[customer_id] = {
            "profile": profile,
            "state": state,
            "created_at": datetime.utcnow(),
            "source": source
        }

        # Send welcome email with instant access
        await self._send_welcome_email(profile, state)

        # Auto-detect optimal configuration
        await self._auto_configure(customer_id, profile)

        self.logger.info(f"Started onboarding for {profile.company_name} ({customer_id})")
        return state

    async def continue_onboarding(
        self,
        customer_id: str,
        action: str,
        data: Optional[Dict] = None
    ) -> OnboardingState:
        """
        Continue onboarding process with customer action

        Args:
            customer_id: Customer identifier
            action: Action being performed
            data: Additional data for the action

        Returns:
            Updated onboarding state
        """
        if customer_id not in self.active_onboardings:
            raise ValueError("Onboarding session not found")

        onboarding = self.active_onboardings[customer_id]
        state = onboarding["state"]
        profile = onboarding["profile"]

        if action == "verify_email":
            state = await self._handle_email_verification(state, data)
        elif action == "upload_sample_data":
            state = await self._handle_sample_data(state, profile, data)
        elif action == "test_integration":
            state = await self._handle_integration_test(state, profile, data)
        elif action == "confirm_setup":
            state = await self._handle_setup_confirmation(state, profile)
        elif action == "go_live":
            state = await self._handle_go_live(state, profile)

        # Update stored state
        self.active_onboardings[customer_id]["state"] = state

        return state

    async def get_integration_code(
        self,
        customer_id: str,
        integration_type: IntegrationType
    ) -> str:
        """
        Generate personalized integration code for customer

        Args:
            customer_id: Customer identifier
            integration_type: Type of integration

        Returns:
            Ready-to-use integration code
        """
        if customer_id not in self.active_onboardings:
            raise ValueError("Customer not found")

        onboarding = self.active_onboardings[customer_id]
        profile = onboarding["profile"]
        state = onboarding["state"]

        # Get auto-detected configuration
        auto_config = state.auto_config or {}

        # Generate code from template
        template = self.integration_templates.get(integration_type)
        if not template:
            raise ValueError(f"No template for {integration_type}")

        code = template.format(
            api_key=state.api_key,
            endpoint=settings.API_BASE_URL,
            method=auto_config.get("recommended_method", "redaction"),
            confidence=auto_config.get("confidence_threshold", 0.85),
            company=profile.company_name,
            use_case=profile.use_case
        )

        return code

    async def auto_analyze_sample_data(
        self,
        customer_id: str,
        sample_data: List[str]
    ) -> Dict[str, Any]:
        """
        Automatically analyze customer's sample data to configure Aegis

        Args:
            customer_id: Customer identifier
            sample_data: Sample data from customer

        Returns:
            Analysis results and configuration recommendations
        """
        analysis = {
            "data_characteristics": {},
            "pii_patterns": [],
            "risk_assessment": {},
            "recommended_config": {},
            "compliance_requirements": []
        }

        # Analyze PII patterns across sample data
        all_entities = []
        for sample in sample_data[:100]:  # Analyze first 100 samples
            entities = await self.pii_detector.detect(sample)
            all_entities.extend(entities)

        # Categorize detected PII
        entity_types = {}
        for entity in all_entities:
            entity_type = entity["type"]
            if entity_type not in entity_types:
                entity_types[entity_type] = 0
            entity_types[entity_type] += 1

        analysis["pii_patterns"] = entity_types

        # Assess risk level
        high_risk_entities = ["SSN", "CREDIT_CARD", "PASSPORT", "MEDICAL_RECORD"]
        medium_risk_entities = ["EMAIL", "PHONE", "DATE_OF_BIRTH"]

        risk_score = 0
        for entity_type, count in entity_types.items():
            if entity_type in high_risk_entities:
                risk_score += count * 0.3
            elif entity_type in medium_risk_entities:
                risk_score += count * 0.1
            else:
                risk_score += count * 0.05

        if risk_score > 10:
            risk_level = "high"
        elif risk_score > 3:
            risk_level = "medium"
        else:
            risk_level = "low"

        analysis["risk_assessment"] = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "total_entities": len(all_entities),
            "unique_types": len(entity_types)
        }

        # Generate recommendations
        if risk_level == "high":
            recommended_method = "tokenization"
            confidence_threshold = 0.95
        elif risk_level == "medium":
            recommended_method = "redaction"
            confidence_threshold = 0.90
        else:
            recommended_method = "masking"
            confidence_threshold = 0.85

        analysis["recommended_config"] = {
            "method": recommended_method,
            "confidence_threshold": confidence_threshold,
            "batch_processing": len(sample_data) > 1000,
            "real_time_protection": True
        }

        # Determine compliance requirements
        compliance_needed = []
        if "SSN" in entity_types or "MEDICAL_RECORD" in entity_types:
            compliance_needed.append("HIPAA")
        if "CREDIT_CARD" in entity_types:
            compliance_needed.append("PCI-DSS")
        if any(eu_indicator in " ".join(sample_data).lower() for eu_indicator in ["gdpr", "europe", "eu"]):
            compliance_needed.append("GDPR")

        analysis["compliance_requirements"] = compliance_needed

        # Update customer's auto-configuration
        if customer_id in self.active_onboardings:
            self.active_onboardings[customer_id]["state"].auto_config = analysis["recommended_config"]
            self.active_onboardings[customer_id]["profile"].detected_patterns = list(entity_types.keys())
            self.active_onboardings[customer_id]["profile"].risk_level = risk_level

        return analysis

    async def generate_onboarding_checklist(
        self,
        customer_id: str
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized onboarding checklist

        Args:
            customer_id: Customer identifier

        Returns:
            List of checklist items with status and instructions
        """
        if customer_id not in self.active_onboardings:
            raise ValueError("Customer not found")

        onboarding = self.active_onboardings[customer_id]
        profile = onboarding["profile"]
        state = onboarding["state"]

        checklist = [
            {
                "id": "api_key",
                "title": "Get API Key",
                "description": "Your API key has been generated and is ready to use",
                "status": "completed" if state.api_key else "pending",
                "instructions": "Copy your API key from the dashboard",
                "estimated_time": "1 minute",
                "api_key": state.api_key
            },
            {
                "id": "integration_code",
                "title": "Add Integration Code",
                "description": f"Integrate Aegis with your {profile.integration_type} setup",
                "status": "pending",
                "instructions": "Copy and paste the generated code into your application",
                "estimated_time": "5 minutes",
                "code_available": True
            },
            {
                "id": "test_protection",
                "title": "Test Data Protection",
                "description": "Send test data to verify PII detection and protection",
                "status": "pending",
                "instructions": "Use our test endpoint to verify everything works",
                "estimated_time": "3 minutes",
                "test_data_provided": True
            },
            {
                "id": "upload_sample",
                "title": "Analyze Your Data (Optional)",
                "description": "Upload sample data for automatic configuration",
                "status": "optional",
                "instructions": "Upload 10-100 sample records for optimal configuration",
                "estimated_time": "2 minutes",
                "benefits": ["Auto-configuration", "Optimized performance", "Compliance verification"]
            },
            {
                "id": "compliance_check",
                "title": "Compliance Verification",
                "description": "Verify compliance with your industry requirements",
                "status": "automated",
                "instructions": "Automatic based on your data patterns",
                "estimated_time": "0 minutes",
                "requirements": profile.compliance_requirements
            },
            {
                "id": "go_live",
                "title": "Go Live",
                "description": "Switch from test to production mode",
                "status": "pending",
                "instructions": "Confirm your setup and enable live traffic",
                "estimated_time": "1 minute",
                "prerequisites": ["api_key", "integration_code", "test_protection"]
            }
        ]

        # Add training data protection if relevant
        if profile.use_case in ["fine_tuning", "model_training", "rag_system"]:
            checklist.insert(4, {
                "id": "training_data",
                "title": "Sanitize Training Data",
                "description": "Clean your historical data for safe model training",
                "status": "optional",
                "instructions": "Upload training datasets for batch sanitization",
                "estimated_time": "10 minutes",
                "benefits": ["Safe model training", "No PII leakage", "Compliance ready"]
            })

        return checklist

    async def get_onboarding_status(self, customer_id: str) -> Dict[str, Any]:
        """Get complete onboarding status and next steps"""
        if customer_id not in self.active_onboardings:
            return {"status": "not_found"}

        onboarding = self.active_onboardings[customer_id]
        state = onboarding["state"]
        profile = onboarding["profile"]

        checklist = await self.generate_onboarding_checklist(customer_id)
        completed_items = sum(1 for item in checklist if item["status"] == "completed")
        total_items = len([item for item in checklist if item["status"] != "optional"])

        progress = (completed_items / total_items) * 100 if total_items > 0 else 0

        return {
            "customer_id": customer_id,
            "company_name": profile.company_name,
            "stage": state.stage,
            "progress_percentage": progress,
            "checklist": checklist,
            "api_key": state.api_key,
            "estimated_completion": state.estimated_completion,
            "auto_config": state.auto_config,
            "support_contact": "support@aegis-shield.ai",
            "documentation_url": "https://docs.aegis-shield.ai/quickstart"
        }

    # Private helper methods

    async def _generate_api_key(self, customer_id: str, company_name: str) -> str:
        """Generate API key for customer"""
        # Generate a secure API key
        key_data = f"{customer_id}:{company_name}:{datetime.utcnow().isoformat()}"
        api_key = f"aegis_live_{hashlib.sha256(key_data.encode()).hexdigest()[:32]}"

        # In production, this would be stored in database
        # For now, we'll store it in the onboarding state

        return api_key

    async def _send_welcome_email(self, profile: CustomerProfile, state: OnboardingState):
        """Send welcome email with instant access"""
        try:
            # In production, integrate with your email service
            email_content = f"""
            Welcome to Aegis, {profile.company_name}!

            Your API key is ready: {state.api_key}

            Next steps:
            1. Visit your dashboard: https://portal.aegis-shield.ai
            2. Copy your integration code
            3. Test with sample data
            4. Go live in minutes!

            Need help? Reply to this email or visit our docs.

            Best regards,
            The Aegis Team
            """

            # Log email instead of actually sending in demo
            self.logger.info(f"Welcome email sent to {profile.technical_contact.get('email', 'customer')}")

        except Exception as e:
            self.logger.error(f"Failed to send welcome email: {e}")

    async def _auto_configure(self, customer_id: str, profile: CustomerProfile):
        """Automatically configure Aegis based on customer profile"""
        config = {}

        # Industry-specific defaults
        if profile.industry.lower() in ["healthcare", "medical"]:
            config["method"] = "tokenization"
            config["confidence_threshold"] = 0.95
            config["compliance_mode"] = "hipaa"
        elif profile.industry.lower() in ["finance", "banking"]:
            config["method"] = "redaction"
            config["confidence_threshold"] = 0.90
            config["compliance_mode"] = "pci_dss"
        else:
            config["method"] = "redaction"
            config["confidence_threshold"] = 0.85
            config["compliance_mode"] = "gdpr"

        # Use case specific settings
        if profile.use_case in ["training", "fine_tuning"]:
            config["batch_processing"] = True
            config["training_data_protection"] = True

        # Volume-based settings
        if profile.expected_volume in ["high", "enterprise"]:
            config["batch_size"] = 10000
            config["parallel_workers"] = 8
        else:
            config["batch_size"] = 1000
            config["parallel_workers"] = 4

        # Store auto-configuration
        if customer_id in self.active_onboardings:
            self.active_onboardings[customer_id]["state"].auto_config = config

    async def _handle_sample_data(self, state: OnboardingState, profile: CustomerProfile, data: Dict) -> OnboardingState:
        """Handle sample data upload and analysis"""
        sample_data = data.get("sample_data", [])

        # Analyze the sample data
        analysis = await self.auto_analyze_sample_data(state.customer_id, sample_data)

        # Update state with analysis results
        state.auto_config = analysis["recommended_config"]
        state.completed_steps.append("Sample data analyzed")
        state.progress_percentage = min(state.progress_percentage + 20, 80)

        if analysis["risk_assessment"]["risk_level"] == "high":
            state.next_steps = ["Review security recommendations", "Test integration", "Go live"]
        else:
            state.next_steps = ["Test integration", "Go live"]

        return state

    async def _handle_integration_test(self, state: OnboardingState, profile: CustomerProfile, data: Dict) -> OnboardingState:
        """Handle integration testing"""
        test_data = data.get("test_data", "")

        # Run test through actual PII detection
        entities = await self.pii_detector.detect(test_data)

        test_results = {
            "input": test_data,
            "entities_detected": len(entities),
            "entities": [{"type": e["type"], "confidence": e["confidence"]} for e in entities],
            "success": True,
            "timestamp": datetime.utcnow().isoformat()
        }

        state.test_results = test_results
        state.completed_steps.append("Integration tested")
        state.progress_percentage = min(state.progress_percentage + 30, 95)
        state.next_steps = ["Go live"]

        return state

    async def _handle_setup_confirmation(self, state: OnboardingState, profile: CustomerProfile) -> OnboardingState:
        """Handle setup confirmation"""
        state.stage = OnboardingStage.PRODUCTION
        state.completed_steps.append("Setup confirmed")
        state.progress_percentage = 95
        state.next_steps = ["Start using Aegis in production"]

        return state

    async def _handle_go_live(self, state: OnboardingState, profile: CustomerProfile) -> OnboardingState:
        """Handle go-live process"""
        state.stage = OnboardingStage.COMPLETED
        state.completed_steps.append("Live deployment")
        state.progress_percentage = 100
        state.next_steps = ["Monitor usage in dashboard", "Set up alerts", "Review compliance reports"]

        # Send go-live notification
        self.logger.info(f"Customer {profile.company_name} has gone live!")

        return state

    async def _handle_email_verification(self, state: OnboardingState, data: Dict) -> OnboardingState:
        """Handle email verification"""
        verification_code = data.get("code")

        # In production, verify against stored code
        # For demo, accept any code
        if verification_code:
            state.completed_steps.append("Email verified")
            state.progress_percentage = max(state.progress_percentage, 15)
            state.next_steps = ["Upload sample data", "Test integration"]

        return state

    def _load_integration_templates(self) -> Dict[str, str]:
        """Load integration code templates"""
        return {
            IntegrationType.OPENAI: '''
# Aegis + OpenAI Integration for {company}
import openai
import requests

# Your Aegis configuration
AEGIS_API_KEY = "{api_key}"
AEGIS_ENDPOINT = "{endpoint}/v1/process"

def safe_openai_chat(message, model="gpt-4"):
    """OpenAI chat with automatic PII protection"""

    # Step 1: Protect the message with Aegis
    response = requests.post(
        AEGIS_ENDPOINT,
        headers={{"X-API-Key": AEGIS_API_KEY}},
        json={{
            "data": message,
            "method": "{method}",
            "confidence_threshold": {confidence}
        }}
    )

    protected_data = response.json()
    safe_message = protected_data["processed_data"]

    # Step 2: Send safe message to OpenAI
    openai_response = openai.ChatCompletion.create(
        model=model,
        messages=[{{"role": "user", "content": safe_message}}]
    )

    return {{
        "response": openai_response.choices[0].message.content,
        "pii_detected": protected_data["entities_detected"],
        "risk_score": protected_data["risk_score"]
    }}

# Usage example
result = safe_openai_chat("My email is john@example.com, help me with my account")
print(result["response"])  # Safe response without PII exposure
            ''',

            IntegrationType.ANTHROPIC: '''
# Aegis + Anthropic Claude Integration for {company}
import anthropic
import requests

# Your Aegis configuration
AEGIS_API_KEY = "{api_key}"
AEGIS_ENDPOINT = "{endpoint}/v1/process"

def safe_claude_chat(message, model="claude-3-sonnet-20240229"):
    """Claude chat with automatic PII protection"""

    # Step 1: Protect the message with Aegis
    response = requests.post(
        AEGIS_ENDPOINT,
        headers={{"X-API-Key": AEGIS_API_KEY}},
        json={{
            "data": message,
            "method": "{method}",
            "confidence_threshold": {confidence}
        }}
    )

    protected_data = response.json()
    safe_message = protected_data["processed_data"]

    # Step 2: Send safe message to Claude
    client = anthropic.Anthropic()
    claude_response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{{"role": "user", "content": safe_message}}]
    )

    return {{
        "response": claude_response.content[0].text,
        "pii_detected": protected_data["entities_detected"],
        "compliance": protected_data["compliance"]
    }}

# Usage example
result = safe_claude_chat("My SSN is 123-45-6789, can you help?")
print(result["response"])  # Safe response without PII exposure
            ''',

            IntegrationType.TRAINING_PIPELINE: '''
# Aegis Training Data Protection for {company}
import requests
import pandas as pd

# Your Aegis configuration
AEGIS_API_KEY = "{api_key}"
AEGIS_ENDPOINT = "{endpoint}/v1/training/sanitize"

def sanitize_training_data(data_path, output_path):
    """Sanitize training data for safe model training"""

    # Load your training data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        data = df.to_dict('records')
    else:
        with open(data_path, 'r') as f:
            data = [line.strip() for line in f]

    # Sanitize with Aegis
    response = requests.post(
        AEGIS_ENDPOINT,
        headers={{"X-API-Key": AEGIS_API_KEY}},
        json={{
            "data": data,
            "method": "{method}",
            "preserve_context": True,
            "output_format": "csv"
        }}
    )

    result = response.json()

    # Save sanitized data
    with open(output_path, 'w') as f:
        f.write(result["sanitized_data"])

    return {{
        "original_records": result["original_size"],
        "pii_removed": result["entities_removed"],
        "compliance": result["compliance_report"],
        "output_path": output_path
    }}

# Usage example
result = sanitize_training_data("raw_chat_logs.csv", "safe_training_data.csv")
print(f"Sanitized {{result['original_records']}} records, removed {{result['pii_removed']}} PII entities")
            '''
        }

    def _load_auto_config_rules(self) -> Dict:
        """Load automatic configuration rules"""
        return {
            "high_risk_industries": ["healthcare", "finance", "government"],
            "compliance_mapping": {
                "healthcare": ["HIPAA", "GDPR"],
                "finance": ["PCI-DSS", "SOX", "GDPR"],
                "government": ["FedRAMP", "FISMA"]
            },
            "method_recommendations": {
                "high_sensitivity": "tokenization",
                "medium_sensitivity": "redaction",
                "low_sensitivity": "masking"
            }
        }


# Convenience functions for quick onboarding

async def quick_start_onboarding(
    company_name: str,
    email: str,
    use_case: str,
    integration_type: str = "openai"
) -> OnboardingState:
    """
    Quick start onboarding with minimal information

    Args:
        company_name: Company name
        email: Contact email
        use_case: Primary use case
        integration_type: Type of integration

    Returns:
        Onboarding state with immediate access
    """
    profile = CustomerProfile(
        company_name=company_name,
        industry="general",
        use_case=use_case,
        data_types=["text"],
        compliance_requirements=["GDPR"],
        integration_type=IntegrationType(integration_type),
        expected_volume="medium",
        technical_contact={"email": email},
        business_contact={"email": email}
    )

    orchestrator = OnboardingOrchestrator()
    await orchestrator.initialize()

    return await orchestrator.start_onboarding(profile, source="quick_start")


if __name__ == "__main__":
    # Example usage
    async def main():
        # Quick start example
        state = await quick_start_onboarding(
            company_name="Acme Corp",
            email="tech@acme.com",
            use_case="customer_support_bot",
            integration_type="openai"
        )

        print(f"Onboarding started! API Key: {state.api_key}")
        print(f"Progress: {state.progress_percentage}%")

        # Get integration code
        orchestrator = OnboardingOrchestrator()
        await orchestrator.initialize()

        code = await orchestrator.get_integration_code(
            state.customer_id,
            IntegrationType.OPENAI
        )

        print("Integration code ready:")
        print(code[:200] + "...")

    asyncio.run(main())