// Initialize OpenStreetMap with Leaflet
const map = L.map('map').setView([10.8505, 76.2711], 7); // Kerala coordinates

// Add OpenStreetMap tiles with a modern style
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: ' OpenStreetMap contributors,  CARTO'
}).addTo(map);

// Sample alert data - In production, this would come from your backend
const alerts = [
    {
        type: 'danger',
        title: 'Severe Weather Alert',
        message: 'Heavy rainfall expected in Wayanad district',
        icon: 'bxs-cloud-lightning'
    },
    {
        type: 'warning',
        title: 'Flood Warning',
        message: 'Rising water levels in Periyar river',
        icon: 'bxs-water'
    },
    {
        type: 'info',
        title: 'Relief Camp Update',
        message: 'New relief camp opened in Thrissur',
        icon: 'bxs-home'
    }
];

// Function to create and animate number counters
function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const current = Math.floor(progress * (end - start) + start);
        obj.textContent = current.toLocaleString();
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Function to update alerts with animation
function updateAlerts() {
    const alertsContainer = document.querySelector('.alerts-container');
    alertsContainer.innerHTML = '';
    
    alerts.forEach((alert, index) => {
        const alertElement = document.createElement('div');
        alertElement.className = `alert alert-${alert.type} d-flex align-items-center`;
        alertElement.style.animationDelay = `${index * 0.2}s`;
        alertElement.innerHTML = `
            <i class="bx ${alert.icon} fs-4 me-2"></i>
            <div>
                <strong>${alert.title}:</strong> ${alert.message}
            </div>
        `;
        alertsContainer.appendChild(alertElement);
    });
}

// Function to update stock levels with smooth animation
function updateStockLevels() {
    const progressBars = document.querySelectorAll('.progress-bar');
    progressBars.forEach(bar => {
        const target = parseInt(bar.style.width);
        let current = 0;
        const increment = target / 100;
        
        const animate = () => {
            if (current < target) {
                current += increment;
                bar.style.width = `${Math.min(current, target)}%`;
                bar.textContent = `${Math.round(Math.min(current, target))}%`;
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    });
}

// Handle notification form submission with modern validation
document.querySelector('form').addEventListener('submit', function(e) {
    e.preventDefault();
    const email = this.querySelector('input[type="email"]');
    const phone = this.querySelector('input[type="tel"]');
    
    // Modern form validation
    let isValid = true;
    
    if (!email.value.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/)) {
        email.classList.add('is-invalid');
        isValid = false;
    } else {
        email.classList.remove('is-invalid');
        email.classList.add('is-valid');
    }
    
    if (!phone.value.match(/^\+?[\d\s-]{10,}$/)) {
        phone.classList.add('is-invalid');
        isValid = false;
    } else {
        phone.classList.remove('is-invalid');
        phone.classList.add('is-valid');
    }
    
    if (isValid) {
        // Show success message with animation
        const alert = document.createElement('div');
        alert.className = 'alert alert-success mt-3 d-flex align-items-center';
        alert.innerHTML = `
            <i class="bx bxs-check-circle fs-4 me-2"></i>
            <div>You have successfully subscribed to alerts!</div>
        `;
        this.appendChild(alert);
        
        // Clear form
        this.reset();
        setTimeout(() => {
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 300);
        }, 3000);
    }
});

// Real-time updates simulation with smooth transitions
function updateRandomStats() {
    // Update random stock level
    const progressBars = document.querySelectorAll('.progress-bar');
    const randomBar = progressBars[Math.floor(Math.random() * progressBars.length)];
    const newValue = Math.floor(Math.random() * 100);
    
    randomBar.style.transition = 'width 1s ease-in-out';
    randomBar.style.width = `${newValue}%`;
    randomBar.textContent = `${newValue}%`;
    
    // Update status badges with smooth transition
    const badges = document.querySelectorAll('.badge');
    const randomBadge = badges[Math.floor(Math.random() * badges.length)];
    const states = [
        { class: 'bg-success', text: 'Operational', icon: 'bxs-check-circle' },
        { class: 'bg-warning', text: 'Limited', icon: 'bxs-error' },
        { class: 'bg-danger', text: 'Critical', icon: 'bxs-x-circle' }
    ];
    const newState = states[Math.floor(Math.random() * states.length)];
    
    randomBadge.style.transition = 'all 0.3s ease';
    randomBadge.className = `badge ${newState.class} d-inline-flex align-items-center`;
    randomBadge.innerHTML = `<i class="bx ${newState.icon} me-1"></i>${newState.text}`;
}

// Add sample map markers with custom icons
const reliefCamps = [
    { coordinates: [10.8505, 76.2711], name: 'Kochi Relief Camp', type: 'camp' },
    { coordinates: [8.5241, 76.9366], name: 'Thiruvananthapuram Camp', type: 'camp' },
    { coordinates: [11.2588, 75.7804], name: 'Kozhikode Camp', type: 'camp' }
];

// Custom marker icons
const campIcon = L.divIcon({
    html: '<i class="bx bxs-home text-primary fs-3"></i>',
    className: 'custom-marker',
    iconSize: [30, 30]
});

// Add markers with animation
reliefCamps.forEach((camp, index) => {
    setTimeout(() => {
        const marker = L.marker(camp.coordinates, { icon: campIcon })
            .bindPopup(`
                <div class="popup-content">
                    <h6 class="mb-1">${camp.name}</h6>
                    <span class="badge bg-success">Active</span>
                </div>
            `)
            .addTo(map);
        
        // Add marker animation
        marker.on('mouseover', function(e) {
            this.openPopup();
        });
    }, index * 200);
});

// Initialize components with staggered animations
document.addEventListener('DOMContentLoaded', () => {
    // Animate stat cards
    document.querySelectorAll('.stat-card').forEach((card, index) => {
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
    
    updateAlerts();
    updateStockLevels();
    setInterval(updateRandomStats, 5000);
});
