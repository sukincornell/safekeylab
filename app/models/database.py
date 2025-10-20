"""
Database Models for Aegis API
"""

from sqlalchemy import Column, String, Integer, DateTime, Boolean, Float, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.core.config import settings

Base = declarative_base()

# Create async engine
engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

class Customer(Base):
    __tablename__ = "customers"

    id = Column(String, primary_key=True, default=lambda: f"cus_{uuid.uuid4().hex[:12]}")
    email = Column(String, unique=True, index=True, nullable=False)
    company_name = Column(String, nullable=False)
    plan = Column(String, default="starter")  # starter, growth, enterprise
    stripe_customer_id = Column(String, unique=True, nullable=True)
    stripe_subscription_id = Column(String, unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    api_keys = relationship("APIKey", back_populates="customer")
    usage_logs = relationship("UsageLog", back_populates="customer")

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False)
    public_key = Column(String, unique=True, index=True, nullable=False)
    secret_key_hash = Column(String, nullable=False)  # Hashed version
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    permissions = Column(JSON, default=dict)  # Store feature permissions

    # Relationships
    customer = relationship("Customer", back_populates="api_keys")

    __table_args__ = (
        Index("idx_api_key_customer", "customer_id"),
    )

class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False)
    request_id = Column(String, unique=True, index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    endpoint = Column(String, nullable=False)
    method = Column(String, nullable=False)  # Privacy method used
    data_size_bytes = Column(Integer, nullable=False)
    entities_detected = Column(Integer, default=0)
    processing_time_ms = Column(Float, nullable=False)
    response_code = Column(Integer, nullable=False)
    ip_address = Column(String, nullable=True)

    # Relationships
    customer = relationship("Customer", back_populates="usage_logs")

    __table_args__ = (
        Index("idx_usage_customer_timestamp", "customer_id", "timestamp"),
    )

class ProcessingReport(Base):
    __tablename__ = "processing_reports"

    id = Column(String, primary_key=True)
    request_id = Column(String, unique=True, index=True, nullable=False)
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    original_data_hash = Column(String, nullable=False)  # Hash of original data
    processed_data_hash = Column(String, nullable=False)  # Hash of processed data
    entities = Column(JSON, nullable=False)  # List of detected entities
    compliance_status = Column(JSON, nullable=False)  # Compliance checks
    risk_score = Column(Float, nullable=False)
    method_used = Column(String, nullable=False)

    __table_args__ = (
        Index("idx_report_customer", "customer_id"),
        Index("idx_report_timestamp", "timestamp"),
    )

class BillingRecord(Base):
    __tablename__ = "billing_records"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False)
    billing_period_start = Column(DateTime, nullable=False)
    billing_period_end = Column(DateTime, nullable=False)
    total_requests = Column(Integer, nullable=False)
    total_data_gb = Column(Float, nullable=False)
    base_cost = Column(Float, nullable=False)
    overage_cost = Column(Float, default=0)
    total_cost = Column(Float, nullable=False)
    stripe_invoice_id = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending, paid, failed
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_billing_customer_period", "customer_id", "billing_period_start"),
    )

# Database session dependency
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

# Initialize database
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# CRUD Operations
class CustomerCRUD:
    @staticmethod
    async def create_customer(db: AsyncSession, email: str, company_name: str, plan: str = "starter"):
        customer = Customer(
            email=email,
            company_name=company_name,
            plan=plan
        )
        db.add(customer)
        await db.commit()
        await db.refresh(customer)
        return customer

    @staticmethod
    async def get_customer_by_email(db: AsyncSession, email: str):
        result = await db.execute(
            "SELECT * FROM customers WHERE email = :email",
            {"email": email}
        )
        return result.first()

    @staticmethod
    async def update_customer_plan(db: AsyncSession, customer_id: str, plan: str):
        result = await db.execute(
            "UPDATE customers SET plan = :plan WHERE id = :id",
            {"plan": plan, "id": customer_id}
        )
        await db.commit()
        return result.rowcount > 0

class APIKeyCRUD:
    @staticmethod
    async def create_api_key(db: AsyncSession, customer_id: str, public_key: str, secret_key_hash: str, name: str = None):
        api_key = APIKey(
            customer_id=customer_id,
            public_key=public_key,
            secret_key_hash=secret_key_hash,
            name=name
        )
        db.add(api_key)
        await db.commit()
        await db.refresh(api_key)
        return api_key

    @staticmethod
    async def get_api_key(db: AsyncSession, public_key: str):
        result = await db.execute(
            "SELECT * FROM api_keys WHERE public_key = :key AND is_active = true",
            {"key": public_key}
        )
        return result.first()

    @staticmethod
    async def update_last_used(db: AsyncSession, api_key_id: str):
        await db.execute(
            "UPDATE api_keys SET last_used_at = :now WHERE id = :id",
            {"now": datetime.utcnow(), "id": api_key_id}
        )
        await db.commit()

class UsageCRUD:
    @staticmethod
    async def log_usage(db: AsyncSession, **kwargs):
        usage_log = UsageLog(**kwargs)
        db.add(usage_log)
        await db.commit()
        return usage_log

    @staticmethod
    async def get_customer_usage(db: AsyncSession, customer_id: str, start_date: datetime, end_date: datetime):
        result = await db.execute(
            """
            SELECT
                COUNT(*) as requests,
                SUM(data_size_bytes) / 1024 / 1024 as data_mb,
                SUM(entities_detected) as entities,
                AVG(processing_time_ms) as avg_latency
            FROM usage_logs
            WHERE customer_id = :customer_id
            AND timestamp BETWEEN :start AND :end
            """,
            {
                "customer_id": customer_id,
                "start": start_date,
                "end": end_date
            }
        )
        return result.first()