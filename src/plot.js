// Store filter states for each tab
const filterStates = {};

function filterBenchmarks(categoryId) {
    const searchTerm = document.getElementById('searchBox_' + categoryId).value.toLowerCase();
    const sections = document.querySelectorAll(`[data-category="${categoryId}"]`);
    const currentFilter = filterStates[categoryId] || 'all';

    sections.forEach(section => {
        const name = section.getAttribute('data-name');
        const status = section.getAttribute('data-status');

        const matchesSearch = name.includes(searchTerm);
        const matchesFilter = currentFilter === 'all' || status === currentFilter;

        if (matchesSearch && matchesFilter) {
            section.classList.remove('hidden');
        } else {
            section.classList.add('hidden');
        }
    });
}

function filterByStatus(categoryId, status, button) {
    filterStates[categoryId] = status;

    // Update button states within the same category
    button.parentElement.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    button.classList.add('active');

    // Apply filter
    filterBenchmarks(categoryId);
}

// Initialize
document.getElementById('benchmark-count').textContent =
    `${GLOBAL_TOTAL_BENCHMARKS} benchmarks across ${GLOBAL_TOTAL_CATEGORIES} categories`;

// Keyboard shortcuts
document.addEventListener('keydown', function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault();
        // Focus on the search box of the active tab
        const activeTab = document.querySelector('.tab-content.active');
        const searchBox = activeTab.querySelector('.search-box');
        if (searchBox) searchBox.focus();
    }
});

// Tab navigation with arrow keys
document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        const tabs = Array.from(document.querySelectorAll('.tab'));
        const activeIndex = tabs.findIndex(tab => tab.classList.contains('active'));
        let newIndex = activeIndex;

        if (e.key === 'ArrowLeft' && activeIndex > 0) {
            newIndex = activeIndex - 1;
        } else if (e.key === 'ArrowRight' && activeIndex < tabs.length - 1) {
            newIndex = activeIndex + 1;
        }

        if (newIndex !== activeIndex) {
            tabs[newIndex].click();
        }
    }
});


// Make Plotly plots responsive on window resize
window.addEventListener('resize', function () {
    const plots = document.querySelectorAll('[id^="plot_"]');
    plots.forEach(function (plot) {
        Plotly.Plots.resize(plot);
    });
});

// Initial resize to ensure plots fit containers
setTimeout(function () {
    const plots = document.querySelectorAll('[id^="plot_"]');
    plots.forEach(function (plot) {
        Plotly.Plots.resize(plot);
    });
}, 100);
