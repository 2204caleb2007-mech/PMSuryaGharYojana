/**
 * Main application JavaScript
 * Handles API calls with authentication
 */

// Override fetch to include auth headers
const originalFetch = window.fetch;
window.fetch = function(url, options = {}) {
    if (typeof url === 'string' && url.startsWith('/api/')) {
        const token = authManager.getToken();
        if (token) {
            options.headers = {
                ...options.headers,
                'Authorization': `Bearer ${token}`
            };
        }
    }
    return originalFetch.call(this, url, options);
};

// Handle analyze location with auth
async function analyzeLocationWithAuth(latitude, longitude, sampleId) {
    try {
        // Check authentication first
        if (!authManager.isAuthenticated()) {
            authManager.showLoginModal();
            return null;
        }

        // Create form data for upload
        const formData = new FormData();
        formData.append('latitude', latitude);
        formData.append('longitude', longitude);
        if (sampleId) {
            formData.append('sample_id', sampleId);
        }

        // Upload analysis
        const uploadResponse = await fetch('/api/analysis/upload', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authManager.getToken()}`
            },
            body: formData
        });

        if (uploadResponse.status === 401) {
            authManager.showLoginModal();
            return null;
        }

        if (!uploadResponse.ok) {
            const error = await uploadResponse.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const analysis = await uploadResponse.json();

        // Run analysis
        const runResponse = await fetch(`/api/analysis/run/${analysis.id}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authManager.getToken()}`
            }
        });

        if (!runResponse.ok) {
            const error = await runResponse.json();
            throw new Error(error.detail || 'Analysis failed');
        }

        const result = await runResponse.json();

        // Get full analysis
        const fullResponse = await fetch(`/api/analysis/${analysis.id}`, {
            headers: {
                'Authorization': `Bearer ${authManager.getToken()}`
            }
        });

        if (!fullResponse.ok) {
            throw new Error('Failed to fetch analysis');
        }

        const fullAnalysis = await fullResponse.json();

        return {
            ...fullAnalysis.result,
            lat: analysis.latitude,
            lon: analysis.longitude,
            sample_id: analysis.sample_id
        };
    } catch (error) {
        console.error('Analysis error:', error);
        throw error;
    }
}

// Export for use in templates
window.analyzeLocationWithAuth = analyzeLocationWithAuth;

