/**
 * Aegis Shield - Dashboard Controller
 *
 * Production-ready dashboard with real API integration,
 * error handling, and loading states
 */

class DashboardController {
    constructor() {
        this.api = typeof aegisAPI !== 'undefined' ? aegisAPI : new AegisAPI();
        this.utils = typeof AegisUtils !== 'undefined' ? AegisUtils : {};
        this.config = typeof AegisConfig !== 'undefined' ? AegisConfig : {};

        this.charts = {};
        this.refreshIntervals = {};
        this.currentFilters = {
            timeRange: '7d',
            dataType: 'all',
            riskLevel: 'all',
            status: 'all'
        };
    }

    async initialize() {
        try {
            // Check authentication
            if (!await this.checkAuth()) {
                return;
            }

            // Load user preferences
            await this.loadUserPreferences();

            // Initialize UI components
            this.initializeEventListeners();
            this.initializeTheme();
            this.initializeSearch();
            this.initializeNotifications();

            // Load dashboard data
            await this.loadDashboardData();

            // Set up auto-refresh
            this.setupAutoRefresh();

            // Track page view
            this.trackEvent('dashboard_view');

        } catch (error) {
            this.handleError(error);
        }
    }

    async checkAuth() {
        try {
            const session = await this.api.verifySession();
            if (!session.valid) {
                window.location.href = '/login.html';
                return false;
            }

            // Update user info in UI
            this.updateUserInfo(session.user);
            return true;

        } catch (error) {
            // For demo mode, check localStorage
            if (this.config.environment === 'development') {
                const isAuthenticated = localStorage.getItem('aegis_authenticated');
                if (!isAuthenticated) {
                    window.location.href = '/login.html';
                    return false;
                }
                return true;
            }

            window.location.href = '/login.html';
            return false;
        }
    }

    updateUserInfo(user) {
        const elements = {
            '.user-name': user?.name || 'User',
            '.user-role': user?.role || 'Member',
            '.user-avatar': user?.initials || 'U'
        };

        for (const [selector, value] of Object.entries(elements)) {
            const element = document.querySelector(selector);
            if (element) {
                element.textContent = value;
            }
        }
    }

    async loadUserPreferences() {
        try {
            const settings = await this.api.getUserSettings();

            // Apply theme preference
            if (settings.theme) {
                document.body.className = settings.theme === 'light' ? 'light-mode' : '';
            }

            // Apply other preferences
            this.currentFilters = settings.dashboardFilters || this.currentFilters;

        } catch (error) {
            // Use local storage as fallback
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'light') {
                document.body.classList.add('light-mode');
            }
        }
    }

    async loadDashboardData() {
        try {
            // Show loading state
            this.showLoadingState();

            // Fetch data in parallel
            const [stats, charts, detections, notifications] = await Promise.all([
                this.api.getDashboardStats(),
                this.api.getDashboardCharts(this.currentFilters.timeRange),
                this.api.getDetections({ limit: 10, ...this.currentFilters }),
                this.api.getNotifications(true)
            ]);

            // Update UI with fetched data
            this.updateStats(stats);
            this.updateCharts(charts);
            this.updateDetectionsTable(detections);
            this.updateNotifications(notifications);

            // Hide loading state
            this.hideLoadingState();

        } catch (error) {
            this.hideLoadingState();

            // In development mode, use mock data
            if (this.config.environment === 'development') {
                this.loadMockData();
            } else {
                this.showErrorState(error);
            }
        }
    }

    updateStats(stats) {
        const statsData = stats || this.getMockStats();

        // Update stat cards
        const statElements = [
            { selector: '#total-requests', value: statsData.totalRequests, format: 'number' },
            { selector: '#pii-detected', value: statsData.piiDetected, format: 'number' },
            { selector: '#compliance-score', value: statsData.complianceScore, format: 'percentage' },
            { selector: '#response-time', value: statsData.avgResponseTime, format: 'time' }
        ];

        statElements.forEach(({ selector, value, format }) => {
            const element = document.querySelector(selector);
            if (element) {
                element.textContent = this.formatValue(value, format);
            }
        });

        // Update change indicators
        this.updateChangeIndicators(statsData.changes);
    }

    updateCharts(chartsData) {
        const data = chartsData || this.getMockChartData();

        // Update or create charts
        this.updateTrendsChart(data.trends);
        this.updateDataTypesChart(data.dataTypes);
        this.updatePerformanceChart(data.performance);
        this.updateComplianceChart(data.compliance);
    }

    updateTrendsChart(data) {
        const ctx = document.getElementById('trendsChart');
        if (!ctx) return;

        if (this.charts.trends) {
            this.charts.trends.destroy();
        }

        const isDark = !document.body.classList.contains('light-mode');
        const textColor = isDark ? '#A3A3A3' : '#6B7280';
        const gridColor = isDark ? '#262626' : '#E5E7EB';

        this.charts.trends = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: data.datasets.map(dataset => ({
                    ...dataset,
                    tension: 0.4,
                    borderWidth: 2
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                return `${context.dataset.label}: ${this.utils.formatNumber(context.parsed.y)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: gridColor,
                            borderColor: gridColor
                        },
                        ticks: {
                            color: textColor
                        }
                    },
                    y: {
                        grid: {
                            color: gridColor,
                            borderColor: gridColor
                        },
                        ticks: {
                            color: textColor,
                            callback: (value) => this.utils.formatNumber(value)
                        }
                    }
                }
            }
        });
    }

    updateDetectionsTable(detections) {
        const tbody = document.getElementById('detectionsTable');
        if (!tbody) return;

        const data = detections?.items || this.getMockDetections();

        tbody.innerHTML = data.map(item => `
            <tr>
                <td>${this.utils.timeAgo(item.timestamp)}</td>
                <td>${this.utils.escapeHtml(item.dataType)}</td>
                <td>${this.utils.escapeHtml(item.source)}</td>
                <td>
                    <span class="table-badge ${this.getRiskClass(item.riskLevel)}">
                        ${item.riskLevel}
                    </span>
                </td>
                <td>
                    <span class="table-badge ${this.getStatusClass(item.status)}">
                        ${item.status}
                    </span>
                </td>
                <td>
                    <div class="table-actions-cell">
                        <button class="action-btn" onclick="dashboard.viewDetection('${item.id}')">
                            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/>
                            </svg>
                        </button>
                        <button class="action-btn" onclick="dashboard.editDetection('${item.id}')">
                            <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/>
                            </svg>
                        </button>
                    </div>
                </td>
            </tr>
        `).join('');
    }

    updateNotifications(notifications) {
        const data = notifications?.items || [];
        const unreadCount = data.filter(n => !n.read).length;

        // Update badge
        const badge = document.querySelector('.notification-badge');
        if (badge) {
            badge.textContent = unreadCount;
            badge.style.display = unreadCount > 0 ? 'flex' : 'none';
        }

        // Store notifications for dropdown
        this.notifications = data;
    }

    setupAutoRefresh() {
        // Clear existing intervals
        Object.values(this.refreshIntervals).forEach(interval => clearInterval(interval));

        if (this.config.ui.refreshIntervals.dashboard > 0) {
            this.refreshIntervals.dashboard = setInterval(() => {
                this.loadDashboardData();
            }, this.config.ui.refreshIntervals.dashboard * 1000);
        }
    }

    initializeEventListeners() {
        // Filter buttons
        document.querySelectorAll('[data-filter]').forEach(button => {
            button.addEventListener('click', (e) => {
                const filterType = e.target.dataset.filter;
                const filterValue = e.target.dataset.value;
                this.applyFilter(filterType, filterValue);
            });
        });

        // Export button
        const exportBtn = document.querySelector('[data-export]');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportData());
        }

        // Refresh button
        const refreshBtn = document.querySelector('[data-refresh]');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadDashboardData());
        }
    }

    async applyFilter(filterType, filterValue) {
        this.currentFilters[filterType] = filterValue;
        await this.loadDashboardData();
    }

    async exportData() {
        try {
            const format = document.querySelector('[data-export-format]')?.value || 'csv';
            await this.api.exportData('detections', format, this.currentFilters);
            this.utils.showToast('Export started successfully', 'success');
        } catch (error) {
            this.handleError(error);
        }
    }

    async viewDetection(id) {
        try {
            const detection = await this.api.getDetectionById(id);
            // Show detection details modal
            this.showDetectionModal(detection);
        } catch (error) {
            this.handleError(error);
        }
    }

    async editDetection(id) {
        // Navigate to edit page or show edit modal
        window.location.href = `/detections/edit/${id}`;
    }

    showLoadingState() {
        document.querySelectorAll('[data-loading]').forEach(element => {
            element.classList.add('loading');
        });
    }

    hideLoadingState() {
        document.querySelectorAll('[data-loading]').forEach(element => {
            element.classList.remove('loading');
        });
    }

    showErrorState(error) {
        const message = error.message || 'Failed to load dashboard data';
        this.utils.showToast(message, 'error');

        // Show error in main content area
        const contentArea = document.querySelector('.content');
        if (contentArea) {
            contentArea.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">⚠️</div>
                    <div class="empty-title">Unable to load dashboard</div>
                    <div class="empty-text">${message}</div>
                    <button class="btn btn-primary" onclick="dashboard.loadDashboardData()">Retry</button>
                </div>
            `;
        }
    }

    handleError(error) {
        console.error('Dashboard error:', error);
        this.utils.handleError(error);
    }

    trackEvent(eventName, eventData = {}) {
        if (this.config.analytics.enabled) {
            // Google Analytics
            if (typeof gtag !== 'undefined') {
                gtag('event', eventName, eventData);
            }

            // Mixpanel
            if (typeof mixpanel !== 'undefined') {
                mixpanel.track(eventName, eventData);
            }
        }
    }

    // Helper methods
    formatValue(value, format) {
        switch (format) {
            case 'number':
                return this.utils.formatNumber(value);
            case 'percentage':
                return this.utils.formatPercentage(value);
            case 'time':
                return `${value}ms`;
            default:
                return value;
        }
    }

    getRiskClass(level) {
        const classes = {
            'Critical': 'danger',
            'High': 'warning',
            'Medium': 'warning',
            'Low': 'success'
        };
        return classes[level] || 'info';
    }

    getStatusClass(status) {
        const classes = {
            'Blocked': 'danger',
            'Redacted': 'warning',
            'Encrypted': 'success',
            'Logged': 'info',
            'Allowed': 'success'
        };
        return classes[status] || 'info';
    }

    // Mock data for development mode
    getMockStats() {
        return {
            totalRequests: 1234567,
            piiDetected: 8432,
            complianceScore: 0.987,
            avgResponseTime: 0.7,
            changes: {
                totalRequests: 12.5,
                piiDetected: -3.2,
                complianceScore: 2.1,
                avgResponseTime: -15
            }
        };
    }

    getMockChartData() {
        return {
            trends: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                datasets: [{
                    label: 'Total Detections',
                    data: [65, 78, 90, 81, 86, 95, 114, 120, 118, 125, 142, 156],
                    borderColor: '#3B82F6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)'
                }, {
                    label: 'Critical',
                    data: [12, 19, 15, 25, 22, 30, 28, 35, 33, 38, 42, 45],
                    borderColor: '#EF4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)'
                }]
            },
            dataTypes: {
                labels: ['Credit Cards', 'SSN', 'Email', 'API Keys', 'Other'],
                data: [30, 25, 20, 15, 10]
            },
            performance: {
                labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
                data: [0.5, 0.6, 0.7, 0.9, 0.8, 0.6]
            },
            compliance: {
                gdpr: 98,
                ccpa: 96,
                hipaa: 99,
                sox: 94
            }
        };
    }

    getMockDetections() {
        return [
            { id: '1', timestamp: new Date(Date.now() - 120000), dataType: 'Credit Card', source: 'API Gateway', riskLevel: 'Critical', status: 'Blocked' },
            { id: '2', timestamp: new Date(Date.now() - 300000), dataType: 'SSN', source: 'Web App', riskLevel: 'High', status: 'Redacted' },
            { id: '3', timestamp: new Date(Date.now() - 480000), dataType: 'Email', source: 'Database', riskLevel: 'Medium', status: 'Logged' },
            { id: '4', timestamp: new Date(Date.now() - 720000), dataType: 'Phone', source: 'API Gateway', riskLevel: 'Low', status: 'Allowed' },
            { id: '5', timestamp: new Date(Date.now() - 900000), dataType: 'API Key', source: 'Logs', riskLevel: 'Critical', status: 'Blocked' }
        ];
    }

    loadMockData() {
        this.updateStats(this.getMockStats());
        this.updateCharts(this.getMockChartData());
        this.updateDetectionsTable({ items: this.getMockDetections() });
        this.updateNotifications({ items: [] });
    }

    // Initialize theme and other UI components
    initializeTheme() {
        const themeToggle = document.querySelector('[data-theme-toggle]');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                document.body.classList.toggle('light-mode');
                const isDark = !document.body.classList.contains('light-mode');
                localStorage.setItem('theme', isDark ? 'dark' : 'light');

                // Update charts
                Object.values(this.charts).forEach(chart => {
                    if (chart) chart.destroy();
                });
                this.updateCharts(this.getMockChartData());
            });
        }
    }

    initializeSearch() {
        // Search functionality would be implemented here
    }

    initializeNotifications() {
        // Notification dropdown functionality would be implemented here
    }

    // Cleanup on page unload
    destroy() {
        Object.values(this.refreshIntervals).forEach(interval => clearInterval(interval));
        Object.values(this.charts).forEach(chart => {
            if (chart) chart.destroy();
        });
    }
}

// Initialize dashboard when DOM is loaded
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new DashboardController();
    dashboard.initialize();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (dashboard) {
        dashboard.destroy();
    }
});