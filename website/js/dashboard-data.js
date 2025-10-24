// Dashboard Data Management
// This file manages all dynamic data for the Aegis dashboard

// Initialize user data from localStorage or set defaults
function initializeUserData() {
    // Check if user data exists
    if (!localStorage.getItem('safekey_user_initialized')) {
        // Set default user data
        const userData = {
            name: 'Alex Johnson',
            email: 'alex.johnson@techcorp.io',
            company: 'TechCorp Industries',
            role: 'Security Administrator',
            apiKey: 'sk_live_' + generateRandomKey(),
            avatar: 'AJ',
            plan: 'Professional',
            signupDate: new Date().toISOString()
        };

        // Store in localStorage
        localStorage.setItem('safekey_user_name', userData.name);
        localStorage.setItem('safekey_user_email', userData.email);
        localStorage.setItem('safekey_user_company', userData.company);
        localStorage.setItem('safekey_user_role', userData.role);
        localStorage.setItem('safekey_user_apikey', userData.apiKey);
        localStorage.setItem('safekey_user_avatar', userData.avatar);
        localStorage.setItem('safekey_user_plan', userData.plan);
        localStorage.setItem('safekey_user_signup', userData.signupDate);
        localStorage.setItem('safekey_user_initialized', 'true');

        // Initialize stats
        initializeStats();
    }

    updateUserProfile();
}

// Generate random key
function generateRandomKey() {
    return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}

// Initialize statistics
function initializeStats() {
    const stats = {
        totalRequests: 1234567,
        piiDetected: 8432,
        complianceScore: 98.7,
        responseTime: 0.7,
        monthlyUsage: 523489,
        remainingCredits: 476511,
        detectionAccuracy: 99.2,
        falsePositiveRate: 0.3,
        lastUpdated: new Date().toISOString()
    };

    localStorage.setItem('safekey_stats', JSON.stringify(stats));
}

// Update user profile in UI
function updateUserProfile() {
    const name = localStorage.getItem('safekey_user_name') || 'Alex Johnson';
    const role = localStorage.getItem('safekey_user_role') || 'Administrator';
    const avatar = localStorage.getItem('safekey_user_avatar') || 'AJ';
    const email = localStorage.getItem('safekey_user_email') || 'user@company.com';

    // Update profile elements
    const userNameEl = document.getElementById('userName');
    const userRoleEl = document.getElementById('userRole');
    const userAvatarEl = document.getElementById('userAvatar');

    if (userNameEl) userNameEl.textContent = name;
    if (userRoleEl) userRoleEl.textContent = role;
    if (userAvatarEl) userAvatarEl.textContent = avatar;

    // Update any email references
    document.querySelectorAll('[data-user-email]').forEach(el => {
        el.textContent = email;
    });
}

// Update statistics dynamically
function updateStats() {
    const statsStr = localStorage.getItem('safekey_stats');
    if (!statsStr) {
        initializeStats();
        return;
    }

    const stats = JSON.parse(statsStr);

    // Add some randomness to make it look live
    stats.totalRequests += Math.floor(Math.random() * 100);
    stats.piiDetected += Math.floor(Math.random() * 10);
    stats.responseTime = (0.5 + Math.random() * 0.5).toFixed(1);

    // Update UI elements
    updateStatCard('total-requests', formatNumber(stats.totalRequests));
    updateStatCard('pii-detected', formatNumber(stats.piiDetected));
    updateStatCard('compliance-score', stats.complianceScore + '%');
    updateStatCard('response-time', stats.responseTime + 'ms');

    // Save updated stats
    localStorage.setItem('safekey_stats', JSON.stringify(stats));
}

// Update a stat card
function updateStatCard(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
        element.removeAttribute('data-loading');
    }
}

// Format large numbers
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toLocaleString();
}

// Generate sample detection data
function generateDetectionData() {
    const types = ['SSN', 'Credit Card', 'Email', 'Phone', 'Address', 'Medical Record', 'API Key', 'Password'];
    const sources = ['API Gateway', 'Web Form', 'Database Scan', 'Log Analysis', 'File Upload', 'Chat System'];
    const statuses = ['Blocked', 'Redacted', 'Flagged', 'Allowed'];

    const detections = [];
    const now = new Date();

    for (let i = 0; i < 20; i++) {
        const time = new Date(now - Math.random() * 86400000); // Random time in last 24 hours
        detections.push({
            id: 'DET-' + Math.random().toString(36).substring(2, 8).toUpperCase(),
            type: types[Math.floor(Math.random() * types.length)],
            source: sources[Math.floor(Math.random() * sources.length)],
            status: statuses[Math.floor(Math.random() * statuses.length)],
            confidence: (85 + Math.random() * 15).toFixed(1) + '%',
            time: formatTime(time),
            details: generateSampleDetail(types[Math.floor(Math.random() * types.length)])
        });
    }

    return detections;
}

// Generate sample detail based on type
function generateSampleDetail(type) {
    const samples = {
        'SSN': 'XXX-XX-6789',
        'Credit Card': 'XXXX-XXXX-XXXX-1234',
        'Email': 'XXXXX@example.com',
        'Phone': 'XXX-XXX-4567',
        'Address': 'XXX Main Street, City, ST',
        'Medical Record': 'MRN: XXXXX789',
        'API Key': 'sk_XXXXX...abc123',
        'Password': '********'
    };
    return samples[type] || 'Sensitive data redacted';
}

// Format time ago
function formatTime(date) {
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return minutes + ' min ago';
    if (hours < 24) return hours + ' hour' + (hours > 1 ? 's' : '') + ' ago';
    return date.toLocaleDateString();
}

// Update detection table
function updateDetectionTable() {
    const tableBody = document.getElementById('detectionsTable');
    if (!tableBody) return;

    const detections = generateDetectionData();

    tableBody.innerHTML = detections.map(d => `
        <tr>
            <td>${d.time}</td>
            <td><span class="badge badge-${d.type.toLowerCase().replace(' ', '-')}">${d.type}</span></td>
            <td>${d.source}</td>
            <td><span class="status-badge ${d.status.toLowerCase()}">${d.status}</span></td>
            <td>${d.confidence}</td>
            <td>
                <button class="action-btn" onclick="viewDetection('${d.id}')">View</button>
            </td>
        </tr>
    `).join('');
}

// View detection details
function viewDetection(id) {
    alert(`Detection Details\n\nID: ${id}\n\nIn a production environment, this would show:\n- Full detection context\n- Remediation options\n- Audit trail\n- Export capabilities`);
}

// Update charts with real-looking data
function updateCharts() {
    // Update any chart elements if they exist
    const chartElements = document.querySelectorAll('[data-chart]');
    chartElements.forEach(el => {
        // This would integrate with Chart.js or similar
        console.log('Chart update needed for:', el.dataset.chart);
    });
}

// Simulate real-time updates
function startRealTimeUpdates() {
    // Update stats every 5 seconds
    setInterval(() => {
        updateStats();
    }, 5000);

    // Update detection table every 10 seconds
    setInterval(() => {
        updateDetectionTable();
    }, 10000);

    // Add new notifications periodically
    setInterval(() => {
        addNewNotification();
    }, 30000);
}

// Add new notification
function addNewNotification() {
    const types = ['info', 'success', 'warning', 'danger'];
    const messages = [
        { title: 'New PII Detection', message: 'Credit card number detected in logs', type: 'danger' },
        { title: 'Compliance Update', message: 'GDPR audit completed successfully', type: 'success' },
        { title: 'System Update', message: 'New PII patterns added to detection engine', type: 'info' },
        { title: 'Usage Alert', message: 'API usage at 75% of monthly limit', type: 'warning' }
    ];

    const notification = messages[Math.floor(Math.random() * messages.length)];

    // Update notification count
    const badge = document.querySelector('.notification-badge');
    if (badge) {
        const count = parseInt(badge.textContent) || 0;
        badge.textContent = count + 1;
        badge.style.display = 'flex';
    }
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize user data
    initializeUserData();

    // Update all dynamic content
    updateStats();
    updateDetectionTable();
    updateCharts();

    // Start real-time updates
    startRealTimeUpdates();

    // Set current date/time
    const dateElement = document.querySelector('.current-date');
    if (dateElement) {
        dateElement.textContent = new Date().toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    }
});

// Export functions for use in other scripts
window.dashboardData = {
    initializeUserData,
    updateStats,
    updateDetectionTable,
    viewDetection,
    formatNumber,
    generateDetectionData
};