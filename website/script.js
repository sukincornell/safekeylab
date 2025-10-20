// Aegis Website Interactive Features

// PII Detection Patterns - Accurate and specific
const piiPatterns = {
    ssn: /\b\d{3}-\d{2}-\d{4}\b/g,
    creditCard: /\b(?:\d{4}[\s-]?){3}\d{4}\b/g,
    cvv: /\bCVV:?\s*\d{3,4}\b/gi,
    email: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,
    phone: /(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/g,
    apiKey: /\b(sk|pk|api)[_-](live|test)[_-][A-Za-z0-9]{10,}\b/gi,
    ipAddress: /\b(?:\d{1,3}\.){3}\d{1,3}\b/g,
    name: /\b(?:Patient|Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+/g,
    medicalCondition: /\bdiagnosed with [\w\s]+/gi
};

// Protect Data Function
function protectData() {
    const input = document.getElementById('demo-input').value;
    const startTime = performance.now();

    if (!input.trim()) {
        alert('Please enter some text with PII to protect');
        return;
    }

    let protectedText = input;
    let piiCount = 0;

    // Detect and replace PII - Process in specific order to avoid conflicts
    const replacements = [];
    const processedRanges = [];

    // Define specific replacement text for each type
    const replacementText = {
        ssn: '[SSN_REDACTED]',
        creditCard: '[CREDIT_CARD]',
        cvv: '[CVV]',
        email: '[EMAIL]',
        phone: '[PHONE]',
        apiKey: '[API_KEY]',
        ipAddress: '[IP_ADDRESS]',
        name: '[NAME]',
        medicalCondition: '[MEDICAL_CONDITION]'
    };

    // Process patterns in order of specificity (longer patterns first)
    const orderedPatterns = ['medicalCondition', 'apiKey', 'email', 'ssn', 'creditCard', 'cvv', 'phone', 'ipAddress', 'name'];

    for (const type of orderedPatterns) {
        const pattern = piiPatterns[type];
        if (!pattern) continue;

        const matches = [...input.matchAll(pattern)];
        if (matches.length > 0) {
            matches.forEach(match => {
                const replacement = replacementText[type] || `[${type.toUpperCase()}]`;
                protectedText = protectedText.replace(match[0], replacement);
                piiCount++;
                replacements.push({ original: match[0], type, replacement });
            });
        }
    }

    // Calculate processing time
    const endTime = performance.now();
    const processTime = (endTime - startTime).toFixed(2);

    // Update output
    const outputElement = document.getElementById('demo-output');

    // Create highlighted output
    let highlightedOutput = protectedText;
    replacements.forEach(rep => {
        highlightedOutput = highlightedOutput.replace(
            rep.replacement,
            `<span class="protected">${rep.replacement}</span>`
        );
    });

    outputElement.innerHTML = highlightedOutput || protectedText;

    // Update stats
    document.getElementById('pii-count').textContent = piiCount;
    document.getElementById('process-time').textContent = `${processTime}ms`;

    // Add animation effect
    outputElement.style.animation = 'fadeIn 0.5s';
    setTimeout(() => {
        outputElement.style.animation = '';
    }, 500);

    // Show notification
    showNotification(`✅ Protected ${piiCount} PII elements in ${processTime}ms`);
}

// ROI Calculator
function calculateROI() {
    const requests = parseInt(document.getElementById('requests').value) || 0;
    const exposure = parseFloat(document.getElementById('exposure').value) || 0;
    const fineAmount = parseInt(document.getElementById('industry').value) || 0;

    // Calculate annual risk
    const annualRequests = requests * 365;
    const exposureRate = exposure / 100;
    const annualRisk = (annualRequests * exposureRate * 0.00001 * fineAmount); // Simplified calculation

    // Aegis cost
    const aegisCost = 200000; // $200K/year for enterprise

    // Calculate ROI
    const roi = Math.round(annualRisk / aegisCost);

    // Update display
    document.getElementById('risk-amount').textContent = `$${annualRisk.toLocaleString()}`;
    document.getElementById('roi-value').textContent = `${roi}x`;

    // Add pulse effect to result
    const riskDisplay = document.querySelector('.risk-amount');
    riskDisplay.style.animation = 'pulse 0.5s';
    setTimeout(() => {
        riskDisplay.style.animation = '';
    }, 500);
}

// Copy Install Command
function copyInstallCommand() {
    const command = 'pip install aegis';
    navigator.clipboard.writeText(command).then(() => {
        showNotification('✅ Copied to clipboard!');

        // Change icon temporarily
        const copyIcon = document.querySelector('.copy-icon');
        copyIcon.textContent = '✅';
        setTimeout(() => {
            copyIcon.textContent = '📋';
        }, 2000);
    });
}

// Show Tab
function showTab(tabName) {
    // Hide all code blocks
    document.querySelectorAll('.code-block').forEach(block => {
        block.classList.add('hidden');
    });

    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });

    // Show selected code block
    document.getElementById(`${tabName}-code`).classList.remove('hidden');

    // Add active class to clicked tab
    event.target.classList.add('active');
}

// Scroll to Demo
function scrollToDemo() {
    document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
}

// Show Notification
function showNotification(message) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: #ffffff;
        color: #000;
        padding: 15px 25px;
        border-radius: 8px;
        font-weight: 600;
        z-index: 9999;
        animation: slideIn 0.3s ease-out;
        box-shadow: 0 10px 40px rgba(255, 255, 255, 0.2);
    `;

    document.body.appendChild(notification);

    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }

    .notification {
        transition: all 0.3s ease;
    }
`;
document.head.appendChild(style);

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling to all links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });

    // Add typing effect to hero title
    const heroTitle = document.querySelector('.hero-title');
    if (heroTitle) {
        heroTitle.style.animation = 'fadeIn 1s ease-out';
    }

    // Initialize ROI calculator with default values
    calculateROI();

    // Add particle effect to security grid
    createSecurityParticles();

    // Handle contact form submission
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            showNotification('✅ Thank you! Our sales team will contact you within 2 hours.');
            contactForm.reset();
        });
    }

    // Add real-time validation to demo input
    const demoInput = document.getElementById('demo-input');
    if (demoInput) {
        demoInput.addEventListener('input', function() {
            // Highlight PII in real-time
            const piiFound = Object.values(piiPatterns).some(pattern =>
                pattern.test(this.value)
            );

            if (piiFound) {
                this.style.borderColor = '#ffffff';
                this.style.boxShadow = '0 0 10px rgba(255, 255, 255, 0.3)';
            } else {
                this.style.borderColor = '#333333';
                this.style.boxShadow = 'none';
            }
        });

        // Add Enter key support
        demoInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                protectData();
            }
        });
    }
});

// Create Security Particles Effect
function createSecurityParticles() {
    const canvas = document.createElement('canvas');
    canvas.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
        opacity: 0.3;
    `;
    document.body.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles = [];
    const particleCount = 50;

    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.vx = (Math.random() - 0.5) * 0.5;
            this.vy = (Math.random() - 0.5) * 0.5;
            this.radius = Math.random() * 2;
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;

            if (this.x < 0 || this.x > canvas.width) this.vx = -this.vx;
            if (this.y < 0 || this.y > canvas.height) this.vy = -this.vy;
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = '#ffffff';
            ctx.fill();
        }
    }

    // Create particles
    for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
    }

    // Animation loop
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        particles.forEach(particle => {
            particle.update();
            particle.draw();
        });

        // Draw connections
        particles.forEach((p1, i) => {
            particles.slice(i + 1).forEach(p2 => {
                const distance = Math.hypot(p1.x - p2.x, p1.y - p2.y);
                if (distance < 100) {
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(255, 255, 255, ${1 - distance / 100})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            });
        });

        requestAnimationFrame(animate);
    }

    animate();

    // Handle resize
    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
}

// Add hover effects to metric cards
document.addEventListener('DOMContentLoaded', function() {
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px) scale(1.05)';
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
});

// Auto-update copyright year
document.addEventListener('DOMContentLoaded', function() {
    const year = new Date().getFullYear();
    const copyrightElements = document.querySelectorAll('.footer-bottom p');
    copyrightElements.forEach(el => {
        el.innerHTML = el.innerHTML.replace('2024', year);
    });
});

// Add loading animation
window.addEventListener('load', function() {
    document.body.style.animation = 'fadeIn 0.5s ease-out';
});

// Track demo usage (analytics placeholder)
function trackEvent(category, action, label) {
    console.log(`Analytics: ${category} - ${action} - ${label}`);
    // In production, replace with actual analytics tracking
    // gtag('event', action, { event_category: category, event_label: label });
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + D: Jump to demo
    if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
        e.preventDefault();
        scrollToDemo();
    }

    // Ctrl/Cmd + I: Copy install command
    if ((e.ctrlKey || e.metaKey) && e.key === 'i') {
        e.preventDefault();
        copyInstallCommand();
    }
});