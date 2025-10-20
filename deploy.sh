#!/bin/bash

# Aegis Production Deployment Script
# Complete setup and deployment for API-based chatbot privacy shield

set -e

echo "================================================"
echo "🛡️  AEGIS API PRODUCTION DEPLOYMENT"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root (recommended for production)
if [[ $EUID -ne 0 ]]; then
   echo -e "${YELLOW}⚠️  Warning: Not running as root. Some operations may require sudo.${NC}"
fi

# Step 1: Environment Check
echo "📋 Checking environment..."
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo "   Creating from .env.example..."
    cp .env.example .env
    echo -e "${YELLOW}   Please edit .env with your configuration${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Environment file found${NC}"
fi

# Step 2: Docker Check
echo ""
echo "📋 Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not installed!${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not installed!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose installed${NC}"

# Step 3: Build Images
echo ""
echo "🔨 Building Docker images..."
docker-compose -f docker-compose.prod.yml build

# Step 4: Initialize Database
echo ""
echo "🗄️  Initializing database..."
docker-compose -f docker-compose.prod.yml up -d postgres redis
sleep 5  # Wait for services to start

# Run database initialization
docker-compose -f docker-compose.prod.yml exec -T postgres psql -U aegis -d aegis_production < scripts/init_db.sql 2>/dev/null || true

# Step 5: Generate API Keys
echo ""
echo "🔑 Generating initial API key..."
python3 scripts/generate_api_key.py --name "Default Client" --save || true

# Step 6: SSL Setup
echo ""
read -p "Do you want to set up SSL certificates? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter your domain (default: api.aegis-shield.ai): " DOMAIN
    DOMAIN=${DOMAIN:-api.aegis-shield.ai}

    read -p "Enter your email for Let's Encrypt: " EMAIL

    read -p "Environment (production/development): " ENV
    ENV=${ENV:-development}

    ./scripts/setup_ssl.sh "$DOMAIN" "$EMAIL" "$ENV"
fi

# Step 7: Start Services
echo ""
echo "🚀 Starting all services..."
docker-compose -f docker-compose.prod.yml up -d

# Step 8: Health Check
echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check service health
SERVICES=("postgres" "redis" "app" "nginx")
ALL_HEALTHY=true

for service in "${SERVICES[@]}"; do
    if docker-compose -f docker-compose.prod.yml exec -T $service echo "OK" &>/dev/null; then
        echo -e "${GREEN}✅ $service is healthy${NC}"
    else
        echo -e "${RED}❌ $service is not responding${NC}"
        ALL_HEALTHY=false
    fi
done

# Step 9: Display Status
echo ""
echo "================================================"
if $ALL_HEALTHY; then
    echo -e "${GREEN}🎉 DEPLOYMENT SUCCESSFUL!${NC}"
else
    echo -e "${YELLOW}⚠️  DEPLOYMENT COMPLETED WITH WARNINGS${NC}"
fi
echo "================================================"
echo ""

# Get container IPs
API_URL=$(docker-compose -f docker-compose.prod.yml port nginx 80 2>/dev/null | cut -d: -f1)
API_URL=${API_URL:-localhost}

echo "📊 Service URLs:"
echo "   • API Endpoint: http://$API_URL/v1/"
echo "   • Health Check: http://$API_URL/health"
echo "   • Prometheus: http://localhost:9090"
echo "   • Grafana: http://localhost:3000 (admin/GrafanaAdmin2024!)"
echo ""

echo "📋 Useful Commands:"
echo "   • View logs: docker-compose -f docker-compose.prod.yml logs -f"
echo "   • Stop services: docker-compose -f docker-compose.prod.yml down"
echo "   • Restart services: docker-compose -f docker-compose.prod.yml restart"
echo "   • Generate API key: python3 scripts/generate_api_key.py --name 'Client Name'"
echo "   • Database shell: docker-compose -f docker-compose.prod.yml exec postgres psql -U aegis -d aegis_production"
echo ""

echo "🔒 Security Reminders:"
echo "   1. Change default passwords in .env"
echo "   2. Set up SSL certificates for production"
echo "   3. Configure firewall rules"
echo "   4. Enable log aggregation and monitoring"
echo "   5. Set up regular backups"
echo ""

# Test API
echo "🧪 Testing API endpoint..."
curl -s -o /dev/null -w "   HTTP Status: %{http_code}\n" http://$API_URL/health || echo -e "${RED}   API test failed${NC}"

echo ""
echo "Your Aegis API is ready for chatbot integration! 🚀"