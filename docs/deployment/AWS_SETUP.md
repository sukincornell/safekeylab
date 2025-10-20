# Aegis AWS Deployment Guide

## 1. AWS Account Setup (30 minutes)

### Prerequisites Installation
```bash
# Install AWS CLI
brew install awscli

# Install Terraform
brew install terraform

# Install kubectl
brew install kubectl

# Install Helm
brew install helm

# Verify installations
aws --version
terraform --version
kubectl version --client
helm version
```

### AWS Account Configuration

1. **Create AWS Account** (if needed)
   - Go to https://aws.amazon.com
   - Click "Create an AWS Account"
   - Use enterprise email: aegis@yourcompany.com

2. **Enable Required Services**
   ```bash
   # These services need to be enabled in your account:
   - Amazon EKS
   - Amazon RDS (Aurora)
   - Amazon ElastiCache
   - Amazon S3
   - Amazon VPC
   - AWS KMS
   - AWS CloudWatch
   - AWS IAM
   ```

3. **Create IAM User for Deployment**
   ```bash
   # Create deployment user with programmatic access
   aws iam create-user --user-name aegis-deployer

   # Attach AdministratorAccess policy (for initial setup)
   aws iam attach-user-policy \
     --user-name aegis-deployer \
     --policy-arn arn:aws:iam::aws:policy/AdministratorAccess

   # Create access keys
   aws iam create-access-key --user-name aegis-deployer
   ```

4. **Configure AWS CLI**
   ```bash
   aws configure
   # Enter:
   # AWS Access Key ID: [from step 3]
   # AWS Secret Access Key: [from step 3]
   # Default region: us-east-1
   # Default output format: json
   ```

5. **Set Service Quotas** (for Fortune 500 scale)
   ```bash
   # Request quota increases via AWS Console:
   - EC2 Instances: 1000
   - EKS Nodes: 200
   - RDS Instances: 50
   - ElastiCache Nodes: 100
   - S3 Buckets: 500
   - Elastic IPs: 50
   ```

6. **Create S3 Bucket for Terraform State**
   ```bash
   aws s3 mb s3://aegis-terraform-state --region us-east-1
   aws s3api put-bucket-versioning \
     --bucket aegis-terraform-state \
     --versioning-configuration Status=Enabled
   aws s3api put-bucket-encryption \
     --bucket aegis-terraform-state \
     --server-side-encryption-configuration '{
       "Rules": [{
         "ApplyServerSideEncryptionByDefault": {
           "SSEAlgorithm": "AES256"
         }
       }]
     }'
   ```

7. **Create DynamoDB Table for State Locking**
   ```bash
   aws dynamodb create-table \
     --table-name aegis-terraform-locks \
     --attribute-definitions AttributeName=LockID,AttributeType=S \
     --key-schema AttributeName=LockID,KeyType=HASH \
     --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
     --region us-east-1
   ```

## 2. Run Deployment (20-30 minutes)

```bash
cd /Users/sukinyang/aegis

# Make script executable
chmod +x enterprise_launch.sh

# Run deployment
./enterprise_launch.sh

# Select option 1: Full Enterprise Deployment
```

## 3. Configure DNS (10 minutes)

After deployment completes, you'll receive load balancer endpoints. Configure your DNS:

### If using Route 53:
```bash
# Create hosted zone
aws route53 create-hosted-zone --name aegis-shield.ai --caller-reference $(date +%s)

# Note the nameservers and update your domain registrar
```

### DNS Records to Create:
```
api.aegis-shield.ai         → CNAME → [ELB endpoint from deployment]
api-us.aegis-shield.ai      → CNAME → [US ELB endpoint]
api-eu.aegis-shield.ai      → CNAME → [EU ELB endpoint]
api-ap.aegis-shield.ai      → CNAME → [AP ELB endpoint]
dashboard.aegis-shield.ai   → CNAME → [Dashboard ELB endpoint]
```

## 4. Validation Tests (Automatic)

The deployment script automatically runs validation tests. Manual verification:

```bash
# Test health endpoint
curl https://api.aegis-shield.ai/health

# Test PII detection
curl -X POST https://api.aegis-shield.ai/v2/process \
  -H "X-API-Key: sk_test_enterprise" \
  -H "Content-Type: application/json" \
  -d '{
    "data": "John Doe, john@example.com, SSN 123-45-6789",
    "compliance_mode": "gdpr"
  }'

# Check cluster status
kubectl get pods -n aegis-production

# View metrics
kubectl top nodes
kubectl top pods -n aegis-production
```

## 5. Post-Deployment Checklist

- [ ] All pods running (minimum 10 per region)
- [ ] Database cluster healthy (3 nodes)
- [ ] Redis cluster operational (6 nodes)
- [ ] SSL certificates active
- [ ] WAF rules enabled
- [ ] CloudWatch alarms configured
- [ ] Backup jobs scheduled
- [ ] Monitoring dashboards accessible

## 6. Enterprise Customer Onboarding

1. **Generate API Keys**
   ```bash
   # Generate customer-specific API key
   openssl rand -hex 32
   ```

2. **Configure Data Residency**
   - Set customer's preferred region
   - Enable cross-region replication if needed

3. **Set Up SSO/SAML**
   - Configure identity provider
   - Map user roles and permissions

4. **Schedule Training**
   - Technical integration session
   - Best practices review
   - Compliance documentation walkthrough

## Support Contacts

- **24/7 Support**: +1-888-AEGIS-AI
- **Email**: enterprise-support@aegis-shield.ai
- **Slack**: aegis-support.slack.com
- **Status Page**: https://status.aegis-shield.ai

## Troubleshooting

If deployment fails, check:
1. AWS service quotas
2. IAM permissions
3. Network connectivity
4. CloudWatch logs: `/aws/eks/aegis-production/cluster`

## Cost Estimate

Monthly AWS infrastructure costs for Fortune 500 deployment:
- EKS Cluster (100 nodes): ~$7,200
- RDS Aurora (6 instances): ~$4,800
- ElastiCache (6 nodes): ~$1,800
- Load Balancers: ~$500
- Data Transfer: ~$2,000
- S3 Storage: ~$500
- **Total**: ~$16,800/month

This infrastructure supports $75,000+/month enterprise contracts with ~78% gross margin.