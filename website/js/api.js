/**
 * Aegis Shield - API Service Layer
 *
 * Handles all API communications with proper error handling,
 * retry logic, and authentication management.
 */

class AegisAPI {
    constructor(config = AegisConfig) {
        this.config = config;
        this.baseUrl = `${config.api.baseUrl}/api/${config.api.version}`;
        this.token = this.getAuthToken();
    }

    // Authentication token management
    getAuthToken() {
        if (this.config.auth.persistAuth) {
            return localStorage.getItem(this.config.auth.tokenKey);
        }
        return sessionStorage.getItem(this.config.auth.tokenKey);
    }

    setAuthToken(token) {
        this.token = token;
        const storage = this.config.auth.persistAuth ? localStorage : sessionStorage;
        storage.setItem(this.config.auth.tokenKey, token);
    }

    clearAuthToken() {
        this.token = null;
        localStorage.removeItem(this.config.auth.tokenKey);
        sessionStorage.removeItem(this.config.auth.tokenKey);
    }

    // Generic request method with error handling and retry logic
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };

        // Add auth token if available
        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }

        const requestOptions = {
            ...options,
            headers,
            timeout: this.config.api.timeout
        };

        // Retry logic
        let lastError;
        for (let i = 0; i <= this.config.api.retry.attempts; i++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), this.config.api.timeout);

                const response = await fetch(url, {
                    ...requestOptions,
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    if (response.status === 401) {
                        // Handle authentication error
                        this.clearAuthToken();
                        window.location.href = '/login.html';
                        throw new Error('Authentication required');
                    }
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                return data;

            } catch (error) {
                lastError = error;

                // Don't retry on authentication errors
                if (error.message.includes('401') || error.message.includes('Authentication')) {
                    throw error;
                }

                // Calculate retry delay with exponential backoff
                if (i < this.config.api.retry.attempts) {
                    const delay = Math.min(
                        this.config.api.retry.delay * Math.pow(2, i),
                        this.config.api.retry.maxDelay
                    );
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }

        throw lastError;
    }

    // Authentication endpoints
    async login(email, password) {
        const response = await this.request('/auth/login', {
            method: 'POST',
            body: JSON.stringify({ email, password })
        });

        if (response.token) {
            this.setAuthToken(response.token);
        }

        return response;
    }

    async logout() {
        try {
            await this.request('/auth/logout', { method: 'POST' });
        } finally {
            this.clearAuthToken();
            window.location.href = '/login.html';
        }
    }

    async verifySession() {
        return await this.request('/auth/verify');
    }

    // Dashboard data
    async getDashboardStats() {
        return await this.request('/dashboard/stats');
    }

    async getDashboardCharts(timeRange = '7d') {
        return await this.request(`/dashboard/charts?range=${timeRange}`);
    }

    // Detections
    async getDetections(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        return await this.request(`/detections?${queryString}`);
    }

    async getDetectionById(id) {
        return await this.request(`/detections/${id}`);
    }

    async createDetectionRule(rule) {
        return await this.request('/detections/rules', {
            method: 'POST',
            body: JSON.stringify(rule)
        });
    }

    // Analytics
    async getAnalytics(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        return await this.request(`/analytics?${queryString}`);
    }

    // API Keys Management
    async getApiKeys() {
        return await this.request('/api-keys');
    }

    async createApiKey(name, permissions) {
        return await this.request('/api-keys', {
            method: 'POST',
            body: JSON.stringify({ name, permissions })
        });
    }

    async revokeApiKey(id) {
        return await this.request(`/api-keys/${id}`, {
            method: 'DELETE'
        });
    }

    // Team Management
    async getTeamMembers() {
        return await this.request('/team/members');
    }

    async inviteTeamMember(email, role) {
        return await this.request('/team/invite', {
            method: 'POST',
            body: JSON.stringify({ email, role })
        });
    }

    async updateTeamMember(userId, updates) {
        return await this.request(`/team/members/${userId}`, {
            method: 'PATCH',
            body: JSON.stringify(updates)
        });
    }

    // Compliance & Audit
    async getComplianceStatus() {
        return await this.request('/compliance/status');
    }

    async getAuditLogs(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        return await this.request(`/audit-logs?${queryString}`);
    }

    async generateComplianceReport(standard, dateRange) {
        return await this.request('/compliance/reports', {
            method: 'POST',
            body: JSON.stringify({ standard, dateRange })
        });
    }

    // Notifications
    async getNotifications(unreadOnly = false) {
        return await this.request(`/notifications?unread=${unreadOnly}`);
    }

    async markNotificationAsRead(id) {
        return await this.request(`/notifications/${id}/read`, {
            method: 'POST'
        });
    }

    async markAllNotificationsAsRead() {
        return await this.request('/notifications/read-all', {
            method: 'POST'
        });
    }

    // Settings
    async getUserSettings() {
        return await this.request('/settings/user');
    }

    async updateUserSettings(settings) {
        return await this.request('/settings/user', {
            method: 'PATCH',
            body: JSON.stringify(settings)
        });
    }

    async getOrganizationSettings() {
        return await this.request('/settings/organization');
    }

    async updateOrganizationSettings(settings) {
        return await this.request('/settings/organization', {
            method: 'PATCH',
            body: JSON.stringify(settings)
        });
    }

    // Data Protection API
    async protectText(text) {
        return await this.request('/protect/text', {
            method: 'POST',
            body: JSON.stringify({ text })
        });
    }

    async detectPII(data, type = 'text') {
        return await this.request('/detect', {
            method: 'POST',
            body: JSON.stringify({ data, type })
        });
    }

    // Export functionality
    async exportData(type, format = 'csv', filters = {}) {
        const params = new URLSearchParams({ format, ...filters }).toString();
        const response = await this.request(`/export/${type}?${params}`);

        // Handle file download
        if (response.downloadUrl) {
            window.location.href = response.downloadUrl;
        }

        return response;
    }

    // Search functionality
    async search(query, filters = {}) {
        return await this.request('/search', {
            method: 'POST',
            body: JSON.stringify({ query, filters })
        });
    }

    // Real-time monitoring
    connectToRealtime() {
        if (!this.token) {
            throw new Error('Authentication required for real-time connection');
        }

        const wsUrl = this.baseUrl.replace('http', 'ws') + '/realtime';
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            ws.send(JSON.stringify({ type: 'auth', token: this.token }));
        };

        return ws;
    }
}

// Create singleton instance
const aegisAPI = new AegisAPI();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = aegisAPI;
}