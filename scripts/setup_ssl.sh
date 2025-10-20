#!/bin/bash

# SSL Certificate Setup Script for Aegis API
# Supports Let's Encrypt (production) and self-signed (development)

set -e

echo "========================================"
echo "🔐 Aegis SSL Certificate Setup"
echo "========================================"

# Configuration
DOMAIN="${1:-api.aegis-shield.ai}"
EMAIL="${2:-admin@aegis-shield.ai}"
ENV="${3:-production}"
SSL_DIR="/etc/nginx/ssl"

# Create SSL directory
mkdir -p $SSL_DIR

if [ "$ENV" == "production" ]; then
    echo ""
    echo "📋 Setting up Let's Encrypt SSL for production..."
    echo "   Domain: $DOMAIN"
    echo "   Email: $EMAIL"
    echo ""

    # Install certbot if not present
    if ! command -v certbot &> /dev/null; then
        echo "Installing certbot..."
        if [ -f /etc/debian_version ]; then
            apt-get update && apt-get install -y certbot python3-certbot-nginx
        elif [ -f /etc/redhat-release ]; then
            yum install -y certbot python3-certbot-nginx
        else
            echo "❌ Unsupported OS for automatic certbot installation"
            echo "   Please install certbot manually"
            exit 1
        fi
    fi

    # Get Let's Encrypt certificate
    certbot certonly \
        --standalone \
        --non-interactive \
        --agree-tos \
        --email $EMAIL \
        --domains $DOMAIN \
        --pre-hook "docker-compose stop nginx 2>/dev/null || true" \
        --post-hook "docker-compose start nginx 2>/dev/null || true"

    # Create symbolic links
    ln -sf /etc/letsencrypt/live/$DOMAIN/fullchain.pem $SSL_DIR/fullchain.pem
    ln -sf /etc/letsencrypt/live/$DOMAIN/privkey.pem $SSL_DIR/privkey.pem
    ln -sf /etc/letsencrypt/live/$DOMAIN/chain.pem $SSL_DIR/chain.pem

    # Setup auto-renewal
    echo "Setting up auto-renewal..."
    (crontab -l 2>/dev/null; echo "0 0,12 * * * certbot renew --quiet --post-hook 'docker-compose restart nginx'") | crontab -

    echo ""
    echo "✅ Let's Encrypt SSL certificates installed!"
    echo "   Certificates: $SSL_DIR/"
    echo "   Auto-renewal: Enabled (twice daily)"

else
    echo ""
    echo "📋 Creating self-signed SSL certificate for development..."
    echo "   Domain: $DOMAIN"
    echo ""

    # Generate self-signed certificate
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout $SSL_DIR/privkey.pem \
        -out $SSL_DIR/fullchain.pem \
        -subj "/C=US/ST=State/L=City/O=Aegis/CN=$DOMAIN"

    # Copy as chain for compatibility
    cp $SSL_DIR/fullchain.pem $SSL_DIR/chain.pem

    echo ""
    echo "✅ Self-signed SSL certificate created!"
    echo "   Certificate: $SSL_DIR/fullchain.pem"
    echo "   Private key: $SSL_DIR/privkey.pem"
    echo ""
    echo "⚠️  Warning: This is a self-signed certificate."
    echo "   Browsers will show security warnings."
    echo "   Only use for development/testing!"
fi

# Create Diffie-Hellman parameters for enhanced security
if [ ! -f $SSL_DIR/dhparam.pem ]; then
    echo ""
    echo "Generating Diffie-Hellman parameters (this may take a while)..."
    openssl dhparam -out $SSL_DIR/dhparam.pem 2048
    echo "✅ DH parameters generated"
fi

# Set proper permissions
chmod 600 $SSL_DIR/privkey.pem
chmod 644 $SSL_DIR/fullchain.pem $SSL_DIR/chain.pem

echo ""
echo "========================================"
echo "🎉 SSL Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Update nginx.conf to use SSL certificates"
echo "2. Restart nginx: docker-compose restart nginx"
echo "3. Test HTTPS: curl https://$DOMAIN/health"
echo ""

# Display certificate info
echo "Certificate information:"
openssl x509 -in $SSL_DIR/fullchain.pem -noout -subject -dates