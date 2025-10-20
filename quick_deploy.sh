#!/bin/bash

# Aegis Quick Deployment Script
# This script helps you deploy Aegis quickly

set -e

echo "======================================"
echo "   AEGIS ENTERPRISE QUICK DEPLOY"
echo "======================================"
echo ""

# Check if running the full deployment or demo
echo "Select deployment mode:"
echo "1) Local Demo (No AWS required)"
echo "2) Full AWS Deployment (Requires AWS account)"
echo ""
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        echo ""
        echo "Starting local demo..."
        echo "This will run a simplified version locally for testing."
        echo ""

        # Check Python
        if ! command -v python3 &> /dev/null; then
            echo "Python 3 is required. Please install it first."
            exit 1
        fi

        # Install dependencies
        echo "Installing Python dependencies..."
        pip3 install fastapi uvicorn pydantic python-jose passlib redis aioredis asyncpg sqlalchemy prometheus-client

        # Run the demo
        echo "Starting Aegis API server..."
        cd /Users/sukinyang/aegis
        python3 app/main_enterprise.py
        ;;

    2)
        echo ""
        echo "Full AWS Deployment"
        echo "==================="
        echo ""
        echo "Prerequisites:"
        echo "1. AWS Account with appropriate permissions"
        echo "2. AWS CLI configured (aws configure)"
        echo "3. Domain name for DNS configuration"
        echo ""
        read -p "Are prerequisites ready? (y/n): " ready

        if [ "$ready" != "y" ]; then
            echo ""
            echo "Please complete these steps first:"
            echo ""
            echo "1. Configure AWS CLI:"
            echo "   aws configure"
            echo "   (Enter your Access Key ID, Secret Key, and region)"
            echo ""
            echo "2. Ensure you have a domain ready for DNS configuration"
            echo ""
            echo "3. Run this script again when ready"
            exit 0
        fi

        echo ""
        echo "Starting deployment..."
        ./enterprise_launch.sh
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac