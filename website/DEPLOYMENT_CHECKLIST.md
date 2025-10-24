# Aegis Shield - Production Deployment Checklist

## Pre-Deployment

### 1. Environment Configuration
- [ ] Copy `.env.example` to `.env`
- [ ] Set `AEGIS_API_URL` to your production API server
- [ ] Configure OAuth credentials (GitHub, Google, SSO)
- [ ] Set analytics IDs (Google Analytics, Mixpanel)
- [ ] Configure feature flags appropriately
- [ ] Set `ENVIRONMENT=production` in config

### 2. Security Audit
- [ ] Remove all console.log statements
- [ ] Ensure no API keys in client-side code
- [ ] Enable HTTPS only
- [ ] Configure CSP headers
- [ ] Set secure cookie flags
- [ ] Enable CORS properly
- [ ] Remove development/debug code

### 3. API Backend
- [ ] API server is deployed and accessible
- [ ] Database migrations completed
- [ ] SSL certificates configured
- [ ] Rate limiting enabled
- [ ] Monitoring configured
- [ ] Backup strategy implemented

### 4. Testing
- [ ] All features tested in staging
- [ ] Authentication flow verified
- [ ] Dashboard data loading confirmed
- [ ] Error handling tested
- [ ] Mobile responsiveness verified
- [ ] Cross-browser compatibility checked

## Deployment Steps

### 1. Build Process
```bash
# Install dependencies
npm install

# Run pre-flight checks
npm run preflight

# Build for production
npm run build:prod
```

### 2. Deploy to Vercel
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod
```

### 3. Deploy to AWS S3 + CloudFront
```bash
# Set environment variables
export AWS_BUCKET_NAME=your-bucket-name
export AWS_DISTRIBUTION_ID=your-distribution-id

# Sync to S3
npm run deploy:aws

# Invalidate CloudFront cache
aws cloudfront create-invalidation \
  --distribution-id $AWS_DISTRIBUTION_ID \
  --paths "/*"
```

### 4. Deploy to Netlify
```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy
netlify deploy --prod
```

### 5. Deploy with Docker
```bash
# Build Docker image
docker build -t aegis-shield:latest .

# Run container
docker run -p 80:80 \
  -e AEGIS_API_URL=https://api.production.com \
  -d aegis-shield:latest
```

## Post-Deployment

### 1. Verification
- [ ] Site is accessible via HTTPS
- [ ] Login functionality works
- [ ] Dashboard loads correctly
- [ ] API calls are successful
- [ ] Analytics tracking active
- [ ] Error monitoring active

### 2. Performance
- [ ] Run Lighthouse audit
- [ ] Check Core Web Vitals
- [ ] Verify CDN caching
- [ ] Test load times globally
- [ ] Monitor API response times

### 3. Security
- [ ] SSL certificate valid
- [ ] Security headers present
- [ ] No exposed sensitive data
- [ ] Authentication working
- [ ] Session management secure

### 4. Monitoring Setup
- [ ] Error tracking (Sentry)
- [ ] Performance monitoring
- [ ] Uptime monitoring
- [ ] Log aggregation
- [ ] Alert rules configured

## Configuration Files

### nginx.conf
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/ssl/certs/your-cert.pem;
    ssl_certificate_key /etc/ssl/private/your-key.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Content-Security-Policy "default-src 'self'" always;

    root /var/www/aegis-shield;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass https://api.your-domain.com;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

### Docker Configuration
```dockerfile
FROM nginx:alpine

# Copy built files
COPY . /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Add environment variable injection script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 80 443

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
```

### docker-entrypoint.sh
```bash
#!/bin/sh

# Replace environment variables in config.js
envsubst < /usr/share/nginx/html/js/config.template.js > /usr/share/nginx/html/js/config.js

# Start nginx
exec "$@"
```

## Environment Variables

### Production (.env.production)
```env
AEGIS_API_URL=https://api.aegis-shield.com
GITHUB_CLIENT_ID=prod-github-client-id
GOOGLE_CLIENT_ID=prod-google-client-id
SSO_ENDPOINT=https://sso.company.com
GA_ID=UA-PRODUCTION-ID
MIXPANEL_TOKEN=prod-token
ENABLE_BETA_FEATURES=false
ENABLE_ANALYTICS=true
SESSION_TIMEOUT_MINUTES=60
ENVIRONMENT=production
```

### Staging (.env.staging)
```env
AEGIS_API_URL=https://api-staging.aegis-shield.com
GITHUB_CLIENT_ID=staging-github-client-id
GOOGLE_CLIENT_ID=staging-google-client-id
SSO_ENDPOINT=https://sso-staging.company.com
GA_ID=UA-STAGING-ID
MIXPANEL_TOKEN=staging-token
ENABLE_BETA_FEATURES=true
ENABLE_ANALYTICS=true
SESSION_TIMEOUT_MINUTES=120
ENVIRONMENT=staging
```

## Rollback Procedure

1. **Immediate Rollback**
   ```bash
   # Vercel
   vercel rollback

   # AWS S3
   aws s3 sync s3://bucket-name-backup s3://bucket-name --delete

   # Docker
   docker stop aegis-shield
   docker run -p 80:80 -d aegis-shield:previous-version
   ```

2. **Database Rollback** (if applicable)
   ```bash
   # Restore from backup
   pg_restore -d aegis_production backup.dump
   ```

3. **Notify Team**
   - Send alert to #incidents channel
   - Update status page
   - Notify affected customers

## Support Contacts

- **DevOps Lead**: devops@aegis-shield.com
- **On-Call Engineer**: +1-XXX-XXX-XXXX
- **Security Team**: security@aegis-shield.com
- **Status Page**: https://status.aegis-shield.com

## Notes

- Always deploy to staging first
- Run smoke tests after deployment
- Monitor error rates for 30 minutes post-deployment
- Keep previous version backup for 7 days
- Document any custom configurations

Last Updated: 2024