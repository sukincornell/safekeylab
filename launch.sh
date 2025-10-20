#!/bin/bash

# Aegis Launch Script
# This script starts the Aegis platform

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║                         ⚡ A E G I S ⚡                              ║"
echo "║                                                                      ║"
echo "║               Enterprise Privacy Shield for AI                       ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "   Please edit .env file with your configuration"
fi

# Choose launch mode
echo "Select launch mode:"
echo "1) Quick Start (Simplified - No dependencies)"
echo "2) Full Stack (With database, Redis, monitoring)"
echo ""
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting Aegis in Quick Start mode..."
        echo ""

        # Build and start simplified version
        docker-compose -f docker-compose.simple.yml up --build -d

        echo ""
        echo "✅ Aegis is running!"
        echo ""
        echo "🌐 Website: http://localhost"
        echo "🔌 API: http://localhost:8000"
        echo "📚 API Docs: http://localhost:8000/docs"
        echo ""
        echo "Test the API with:"
        echo 'curl -X POST http://localhost:8000/v1/process \
  -H "X-API-Key: sk_test_key" \
  -H "Content-Type: application/json" \
  -d "{\"data\": \"John email is john@example.com\"}"'
        ;;

    2)
        echo ""
        echo "🚀 Starting Aegis Full Stack..."
        echo ""

        # Check for required environment variables
        if [ -z "$SECRET_KEY" ]; then
            echo "⚠️  Generating SECRET_KEY..."
            SECRET_KEY=$(openssl rand -hex 32)
            echo "SECRET_KEY=$SECRET_KEY" >> .env
        fi

        # Build and start full stack
        docker-compose up --build -d

        echo ""
        echo "✅ Aegis Full Stack is running!"
        echo ""
        echo "🌐 Website: http://localhost"
        echo "🔌 API: http://localhost:8000"
        echo "📚 API Docs: http://localhost:8000/docs"
        echo "📊 Grafana: http://localhost:3000 (admin/admin)"
        echo "📈 Prometheus: http://localhost:9090"
        echo ""
        ;;

    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "To stop Aegis, run:"
echo "  docker-compose down"
echo ""
echo "To view logs, run:"
echo "  docker-compose logs -f"
echo ""
echo "⚡ Aegis is ready to protect your AI!"