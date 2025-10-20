# Aegis Enterprise Service Level Agreement (SLA)

**Effective Date**: January 1, 2024
**Version**: 2.0
**Classification**: Enterprise Agreement

## 1. Service Availability Guarantee

### 1.1 Uptime Commitment

| Service Tier | Monthly Uptime SLA | Annual Uptime SLA | Maximum Downtime/Month |
|-------------|-------------------|-------------------|------------------------|
| **Enterprise** | 99.99% | 99.99% | 4.32 minutes |
| **Enterprise Plus** | 99.995% | 99.995% | 2.16 minutes |
| **Unlimited** | 99.999% | 99.999% | 26 seconds |

### 1.2 Service Credits

| Monthly Uptime Percentage | Service Credit |
|---------------------------|----------------|
| 99.9% - 99.99% | 10% |
| 99.0% - 99.9% | 25% |
| 95.0% - 99.0% | 50% |
| < 95.0% | 100% |

## 2. Performance Guarantees

### 2.1 API Response Time

| Metric | P50 | P95 | P99 | P99.9 |
|--------|-----|-----|-----|-------|
| **Response Latency** | < 25ms | < 50ms | < 100ms | < 250ms |
| **Processing Time** | < 35ms | < 75ms | < 150ms | < 500ms |
| **End-to-End** | < 50ms | < 100ms | < 250ms | < 1000ms |

### 2.2 Throughput Guarantees

| Service Tier | Requests/Second | Requests/Month | Burst Capacity |
|-------------|-----------------|----------------|----------------|
| **Enterprise** | 10,000 | 10 Billion | 50,000 RPS |
| **Enterprise Plus** | 50,000 | 50 Billion | 200,000 RPS |
| **Unlimited** | 100,000+ | Unlimited | 500,000 RPS |

### 2.3 Data Processing Capacity

- **Maximum Request Size**: 100 MB
- **Batch Processing**: Up to 10,000 items per batch
- **Concurrent Connections**: 100,000+
- **Geographic Latency**: < 50ms from any major region

## 3. Data Protection & Security

### 3.1 Encryption Standards

- **Data at Rest**: AES-256-GCM
- **Data in Transit**: TLS 1.3
- **Key Management**: FIPS 140-2 Level 3 HSM
- **Key Rotation**: Automatic every 90 days

### 3.2 Compliance Certifications

| Framework | Certification | Audit Frequency |
|-----------|--------------|-----------------|
| **SOC 2 Type II** | ✅ Certified | Annual |
| **ISO 27001** | ✅ Certified | Annual |
| **ISO 27018** | ✅ Certified | Annual |
| **HIPAA** | ✅ Compliant | Continuous |
| **PCI DSS Level 1** | ✅ Certified | Annual |
| **GDPR** | ✅ Compliant | Continuous |
| **CCPA** | ✅ Compliant | Continuous |
| **FedRAMP** | In Progress | 2024 Q2 |

### 3.3 Data Residency

- **Available Regions**: US, EU, UK, CA, AU, JP, SG
- **Data Sovereignty**: Guaranteed within selected region
- **Cross-Region Replication**: Optional with explicit consent
- **Data Retention**: Configurable 1-2555 days

## 4. Support Services

### 4.1 Support Tiers

| Support Level | Response Time (Critical) | Response Time (High) | Response Time (Normal) | Response Time (Low) |
|--------------|-------------------------|---------------------|----------------------|-------------------|
| **Enterprise** | 15 minutes | 1 hour | 4 hours | 1 business day |
| **Enterprise Plus** | 5 minutes | 30 minutes | 2 hours | 4 hours |
| **Unlimited** | Immediate | 15 minutes | 1 hour | 2 hours |

### 4.2 Support Channels

- **24/7/365 Phone Support**: All enterprise tiers
- **Dedicated Slack Channel**: Enterprise Plus and above
- **Technical Account Manager**: Enterprise Plus and above
- **Executive Escalation**: Unlimited tier
- **On-Site Support**: Available for Unlimited tier

### 4.3 Incident Management

| Severity | Definition | Response Time | Update Frequency |
|----------|-----------|---------------|-------------------|
| **Critical (P1)** | Complete service outage | 5 minutes | Every 30 minutes |
| **High (P2)** | Major feature unavailable | 30 minutes | Every 2 hours |
| **Medium (P3)** | Minor feature degraded | 2 hours | Every 8 hours |
| **Low (P4)** | Cosmetic issues | 1 business day | As needed |

## 5. Service Level Objectives (SLOs)

### 5.1 Reliability Metrics

- **API Success Rate**: > 99.99%
- **Data Durability**: 99.999999999% (11 nines)
- **PII Detection Accuracy**: > 99.9%
- **False Positive Rate**: < 0.1%
- **False Negative Rate**: < 0.01%

### 5.2 Operational Metrics

- **Deployment Frequency**: Daily
- **Lead Time for Changes**: < 1 hour
- **Mean Time to Recovery (MTTR)**: < 5 minutes
- **Mean Time Between Failures (MTBF)**: > 720 hours
- **Change Failure Rate**: < 0.1%

## 6. Disaster Recovery

### 6.1 Recovery Objectives

| Metric | Enterprise | Enterprise Plus | Unlimited |
|--------|-----------|-----------------|-----------|
| **Recovery Time Objective (RTO)** | 1 hour | 15 minutes | 5 minutes |
| **Recovery Point Objective (RPO)** | 1 hour | 15 minutes | 1 minute |
| **Backup Frequency** | Hourly | Every 15 min | Continuous |
| **Backup Retention** | 35 days | 90 days | 365 days |

### 6.2 Business Continuity

- **Multi-Region Failover**: Automatic within 60 seconds
- **Data Center Redundancy**: N+2 across all regions
- **Network Path Diversity**: Minimum 3 independent paths
- **Disaster Recovery Testing**: Quarterly with customer participation

## 7. Maintenance Windows

### 7.1 Scheduled Maintenance

- **Frequency**: Monthly
- **Duration**: Maximum 2 hours
- **Notification**: 7 days advance notice
- **Time**: 2:00 AM - 4:00 AM UTC (customizable per region)
- **Zero-Downtime Deployments**: Yes for application updates

### 7.2 Emergency Maintenance

- **Notification**: Minimum 4 hours (critical security only)
- **Approval Required**: Yes for Enterprise Plus and above
- **Service Credit**: 5% for any emergency maintenance

## 8. Data Processing Agreement (DPA)

### 8.1 Data Controller Responsibilities

- Customer remains the data controller
- Aegis acts solely as data processor
- No data usage for training or improvement without explicit consent
- Complete data isolation between customers

### 8.2 Data Subject Rights

- **Access**: Within 24 hours
- **Rectification**: Within 48 hours
- **Erasure**: Within 72 hours
- **Portability**: Within 5 business days
- **Audit Rights**: Annual with 30 days notice

## 9. Financial Terms

### 9.1 Pricing Stability

- **Price Protection**: 24 months from contract signing
- **Annual Increase Cap**: Maximum 5%
- **Volume Discounts**: Automatic at thresholds
- **No Hidden Fees**: All costs disclosed upfront

### 9.2 Billing Accuracy

- **Billing Disputes**: Resolved within 30 days
- **Overcharge Protection**: 2x credit for any overcharge
- **Usage Transparency**: Real-time dashboard access
- **Cost Alerts**: Configurable at any threshold

## 10. Termination & Data Export

### 10.1 Contract Terms

- **Minimum Term**: 12 months
- **Renewal**: Automatic unless 90 days notice
- **Termination for Convenience**: 90 days notice
- **Termination for Cause**: Immediate with cause

### 10.2 Data Export

- **Export Format**: JSON, CSV, Parquet, or custom
- **Export Timeline**: Within 30 days of termination
- **Data Deletion**: Certified deletion within 90 days
- **No Vendor Lock-in**: Full API compatibility maintained

## 11. Liability & Indemnification

### 11.1 Liability Cap

| Service Tier | Direct Damages | Indirect Damages |
|-------------|---------------|------------------|
| **Enterprise** | 12 months of fees | Excluded |
| **Enterprise Plus** | 24 months of fees | 6 months of fees |
| **Unlimited** | Uncapped | 12 months of fees |

### 11.2 Indemnification

- **IP Indemnification**: Full coverage
- **Data Breach**: $50M cyber insurance
- **Regulatory Fines**: Covered if due to Aegis fault
- **Legal Defense**: Included at no cost

## 12. Governance

### 12.1 Service Reviews

- **Quarterly Business Reviews**: All tiers
- **Executive Reviews**: Semi-annual for Enterprise Plus
- **Board Presentations**: Annual for Unlimited

### 12.2 Change Management

- **API Versioning**: Minimum 12 months deprecation notice
- **Breaking Changes**: Require customer approval
- **Feature Requests**: Considered within 30 days
- **Roadmap Access**: Full visibility for Enterprise Plus

## 13. Exclusions

This SLA does not apply to:

- Customer-caused outages
- Force majeure events
- Scheduled maintenance windows
- Customer exceeding rate limits
- Issues with customer's infrastructure
- Beta or preview features

## 14. SLA Monitoring

### 14.1 Transparency

- **Public Status Page**: https://status.aegis-shield.ai
- **Real-time Metrics**: Available via API
- **Monthly Reports**: Automated delivery
- **Annual Audits**: Third-party verified

### 14.2 Measurement

All metrics measured from Aegis monitoring systems. Customer may request third-party verification at any time.

## 15. Contact Information

### Enterprise Support
- **Phone**: +1-888-AEGIS-AI (24/7/365)
- **Email**: enterprise-support@aegis-shield.ai
- **Portal**: https://support.aegis-shield.ai

### Executive Escalation
- **Email**: executive@aegis-shield.ai
- **Direct Line**: Available for Unlimited tier

### Legal & Compliance
- **Email**: legal@aegis-shield.ai
- **Address**: Aegis Security, Inc., 1 Market Street, San Francisco, CA 94105

---

**Signature Block**

_By using Aegis services under an Enterprise agreement, you acknowledge and agree to these SLA terms._

**Aegis Security, Inc.**

_____________________
Chief Executive Officer
Date: _____________

**Customer**

_____________________
Authorized Signatory
Date: _____________

---

**Document Control**
- Last Updated: January 1, 2024
- Review Cycle: Quarterly
- Next Review: April 1, 2024
- Document ID: AEGIS-SLA-ENT-2024-v2.0