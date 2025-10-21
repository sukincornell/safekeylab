# GitHub Actions Secrets Setup

To enable full CI/CD functionality, add these secrets to your GitHub repository:

## Required Secrets

Go to: Settings → Secrets and variables → Actions → New repository secret

### 1. Docker Hub (Optional - for container registry)
- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub access token

### 2. Vercel (Optional - for website deployment)
- `VERCEL_TOKEN`: Get from https://vercel.com/account/tokens
- `VERCEL_ORG_ID`: Found in `.vercel/project.json`
- `VERCEL_PROJECT_ID`: Found in `.vercel/project.json`

### 3. AWS (Optional - for cloud deployment)
- `AWS_ACCESS_KEY_ID`: AWS IAM access key
- `AWS_SECRET_ACCESS_KEY`: AWS IAM secret key

## Current Status

The CI/CD pipeline will run with these features:
- ✅ **Testing**: Runs automatically (no secrets needed)
- ✅ **Linting**: Runs automatically (no secrets needed)
- ✅ **Security Scanning**: Runs automatically (no secrets needed)
- ✅ **Benchmarks**: Runs automatically (no secrets needed)
- ⚠️ **Docker Build**: Requires Docker Hub secrets
- ⚠️ **Website Deploy**: Requires Vercel secrets
- ⚠️ **API Deploy**: Requires AWS secrets

## Quick Start (Minimal Setup)

No secrets are required for basic CI functionality. The pipelines will:
1. Run all tests
2. Check code quality
3. Scan for vulnerabilities
4. Generate benchmark reports

Deployment steps will be skipped if secrets are not configured.