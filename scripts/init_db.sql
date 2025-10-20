-- Aegis Database Initialization Script
-- PostgreSQL setup for production

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS aegis_production;

\c aegis_production;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS api;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS metrics;

-- Set search path
SET search_path TO api, public;

-- Organizations table
CREATE TABLE IF NOT EXISTS api.organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB DEFAULT '{}'::jsonb
);

-- API Keys table
CREATE TABLE IF NOT EXISTS api.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES api.organizations(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    key_prefix VARCHAR(20) NOT NULL,
    scopes TEXT[] DEFAULT ARRAY['detect', 'anonymize', 'process'],
    rate_limit INTEGER DEFAULT 10000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active',
    metadata JSONB DEFAULT '{}'::jsonb,
    INDEX idx_key_hash (key_hash),
    INDEX idx_organization (organization_id),
    INDEX idx_status (status)
);

-- Usage logs table (partitioned by month)
CREATE TABLE IF NOT EXISTS api.usage_logs (
    id BIGSERIAL,
    api_key_id UUID REFERENCES api.api_keys(id) ON DELETE SET NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    request_size INTEGER,
    response_size INTEGER,
    status_code INTEGER,
    latency_ms INTEGER,
    pii_detected INTEGER DEFAULT 0,
    pii_removed INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    error_message TEXT,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create monthly partitions for usage logs
CREATE TABLE IF NOT EXISTS api.usage_logs_2024_01 PARTITION OF api.usage_logs
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE IF NOT EXISTS api.usage_logs_2024_02 PARTITION OF api.usage_logs
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Add more partitions as needed...

-- PII Detection results cache
CREATE TABLE IF NOT EXISTS api.pii_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text_hash VARCHAR(255) UNIQUE NOT NULL,
    detected_entities JSONB NOT NULL,
    model_version VARCHAR(50),
    confidence_scores JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP + INTERVAL '24 hours',
    INDEX idx_text_hash (text_hash),
    INDEX idx_expires (expires_at)
);

-- Audit log for compliance
CREATE TABLE IF NOT EXISTS audit.activity_logs (
    id BIGSERIAL PRIMARY KEY,
    organization_id UUID,
    api_key_id UUID,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    actor_type VARCHAR(50) DEFAULT 'api',
    actor_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    request_data JSONB,
    response_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_organization (organization_id),
    INDEX idx_created_at (created_at),
    INDEX idx_action (action)
);

-- Metrics aggregation tables
CREATE TABLE IF NOT EXISTS metrics.hourly_stats (
    id BIGSERIAL PRIMARY KEY,
    organization_id UUID,
    hour TIMESTAMP NOT NULL,
    total_requests BIGINT DEFAULT 0,
    successful_requests BIGINT DEFAULT 0,
    failed_requests BIGINT DEFAULT 0,
    total_pii_detected BIGINT DEFAULT 0,
    total_pii_removed BIGINT DEFAULT 0,
    avg_latency_ms NUMERIC(10,2),
    p95_latency_ms NUMERIC(10,2),
    p99_latency_ms NUMERIC(10,2),
    unique_api_keys INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(organization_id, hour),
    INDEX idx_hour (hour),
    INDEX idx_org_hour (organization_id, hour)
);

-- Webhook configurations
CREATE TABLE IF NOT EXISTS api.webhooks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES api.organizations(id) ON DELETE CASCADE,
    url VARCHAR(500) NOT NULL,
    events TEXT[] NOT NULL,
    secret VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_triggered_at TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    INDEX idx_organization (organization_id),
    INDEX idx_status (status)
);

-- Functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to tables with updated_at
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON api.organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_webhooks_updated_at BEFORE UPDATE ON api.webhooks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to cleanup expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM api.pii_cache WHERE expires_at < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_usage_logs_created_at ON api.usage_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_usage_logs_api_key ON api.usage_logs(api_key_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit.activity_logs(created_at DESC);

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA api TO aegis;
GRANT ALL PRIVILEGES ON SCHEMA audit TO aegis;
GRANT ALL PRIVILEGES ON SCHEMA metrics TO aegis;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA api TO aegis;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO aegis;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA metrics TO aegis;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA api TO aegis;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA audit TO aegis;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA metrics TO aegis;

-- Insert default organization for testing
INSERT INTO api.organizations (name, slug, metadata)
VALUES ('Default Organization', 'default', '{"type": "trial", "tier": "startup"}')
ON CONFLICT (slug) DO NOTHING;

-- Success message
SELECT 'Database initialization completed successfully!' as status;