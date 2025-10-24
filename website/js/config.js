/**
 * SafeKey Lab - Configuration
 *
 * This file contains all configuration settings for the SafeKey Lab application.
 * Update these values based on your deployment environment.
 */

const SafeKeyConfig = {
    // Environment: 'development', 'staging', 'production'
    environment: window.location.hostname === 'localhost' ? 'development' :
                 window.location.hostname.includes('staging') ? 'staging' : 'production',

    // API Configuration
    api: {
        // Base URL for API endpoints - update this to your actual API server
        baseUrl: process.env.SAFEKEY_API_URL || window.SAFEKEY_API_URL ||
                 (window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'https://api.safekeylab.com'),

        // API Version
        version: 'v1',

        // API Timeout (ms)
        timeout: 30000,

        // Retry configuration
        retry: {
            attempts: 3,
            delay: 1000,
            maxDelay: 5000
        }
    },

    // Authentication Configuration
    auth: {
        // Token storage key
        tokenKey: 'safekey_auth_token',

        // Session storage key
        sessionKey: 'safekey_session',

        // Authentication persistence
        persistAuth: true,

        // Session timeout (minutes)
        sessionTimeout: 60,

        // OAuth providers configuration
        oauth: {
            github: {
                clientId: process.env.GITHUB_CLIENT_ID || window.GITHUB_CLIENT_ID || 'your-github-client-id',
                redirectUri: window.location.origin + '/auth/callback/github'
            },
            google: {
                clientId: process.env.GOOGLE_CLIENT_ID || window.GOOGLE_CLIENT_ID || 'your-google-client-id',
                redirectUri: window.location.origin + '/auth/callback/google'
            },
            sso: {
                endpoint: process.env.SSO_ENDPOINT || window.SSO_ENDPOINT || 'https://sso.yourcompany.com',
                clientId: process.env.SSO_CLIENT_ID || window.SSO_CLIENT_ID || 'your-sso-client-id'
            }
        }
    },

    // Feature Flags
    features: {
        // Enable/disable features based on environment or deployment
        dashboard: true,
        analytics: true,
        realTimeMonitoring: true,
        advancedFiltering: true,
        exportData: true,
        teamManagement: true,
        apiKeys: true,
        compliance: true,

        // Beta features
        beta: {
            aiInsights: false,
            predictiveAnalytics: false,
            customModels: false
        }
    },

    // UI Configuration
    ui: {
        // Theme settings
        defaultTheme: 'dark', // 'dark' or 'light'
        allowThemeToggle: true,

        // Pagination
        defaultPageSize: 10,
        pageSizeOptions: [10, 25, 50, 100],

        // Data refresh intervals (seconds)
        refreshIntervals: {
            dashboard: 30,
            detections: 10,
            analytics: 60,
            notifications: 15
        },

        // Chart settings
        charts: {
            animationDuration: 750,
            defaultType: 'line',
            showLegend: true,
            responsive: true
        }
    },

    // Notification Configuration
    notifications: {
        // Enable browser notifications
        browserNotifications: true,

        // Enable in-app notifications
        inAppNotifications: true,

        // Notification channels
        channels: {
            email: true,
            sms: false,
            slack: false,
            webhook: true
        }
    },

    // Analytics & Tracking
    analytics: {
        // Enable analytics
        enabled: true,

        // Google Analytics ID
        googleAnalyticsId: process.env.GA_ID || window.GA_ID || 'UA-XXXXXXXXX-X',

        // Mixpanel token
        mixpanelToken: process.env.MIXPANEL_TOKEN || window.MIXPANEL_TOKEN || null,

        // Custom tracking endpoint
        customEndpoint: null
    },

    // Compliance & Security
    compliance: {
        // Required compliance standards
        standards: ['GDPR', 'CCPA', 'HIPAA', 'SOC2'],

        // Data retention policy (days)
        dataRetention: 90,

        // Audit log retention (days)
        auditLogRetention: 365,

        // Encryption settings
        encryption: {
            algorithm: 'AES-256-GCM',
            keyRotation: 30 // days
        }
    },

    // Support & Documentation
    support: {
        // Support email
        email: 'support@safekeylab.com',

        // Documentation URL
        docsUrl: 'https://docs.safekeylab.com',

        // API documentation
        apiDocsUrl: 'https://api-docs.safekeylab.com',

        // Status page
        statusPageUrl: 'https://status.safekeylab.com',

        // Community forum
        forumUrl: 'https://community.safekeylab.com'
    },

    // Deployment Configuration
    deployment: {
        // CDN URLs for assets
        cdn: {
            enabled: false,
            baseUrl: 'https://cdn.safekeylab.com'
        },

        // Asset versioning
        version: '2.0.0',

        // Build timestamp
        buildTime: new Date().toISOString()
    }
};

// Freeze configuration to prevent accidental modifications
Object.freeze(SafeKeyConfig);

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SafeKeyConfig;
}