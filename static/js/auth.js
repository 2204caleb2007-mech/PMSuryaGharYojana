/**
 * Authentication and Guest Mode Handler
 * Handles JWT tokens, user state, and guest mode behavior
 */

class AuthManager {
    constructor() {
        this.tokenKey = 'solar_analyzer_token';
        this.userKey = 'solar_analyzer_user';
        this.apiBase = '/api';
    }

    // Get stored token
    getToken() {
        return localStorage.getItem(this.tokenKey);
    }

    // Store token
    setToken(token) {
        localStorage.setItem(this.tokenKey, token);
    }

    // Remove token (logout)
    removeToken() {
        localStorage.removeItem(this.tokenKey);
        localStorage.removeItem(this.userKey);
    }

    // Get current user info
    getCurrentUser() {
        const userStr = localStorage.getItem(this.userKey);
        return userStr ? JSON.parse(userStr) : null;
    }

    // Set current user info
    setCurrentUser(user) {
        localStorage.setItem(this.userKey, JSON.stringify(user));
    }

    // Check if user is authenticated
    isAuthenticated() {
        return !!this.getToken();
    }

    // Get auth headers for fetch requests
    getAuthHeaders() {
        const token = this.getToken();
        return token ? {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        } : {};
    }

    // Register new user
    async register(email, username, password, fullName = null) {
        try {
            const response = await fetch(`${this.apiBase}/auth/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    email,
                    username,
                    password,
                    full_name: fullName
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Registration failed');
            }

            const user = await response.json();
            this.setCurrentUser(user);
            
            // Auto-login after registration
            return await this.login(email, password);
        } catch (error) {
            throw error;
        }
    }

    // Login
    async login(email, password) {
        try {
            const response = await fetch(`${this.apiBase}/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    email,
                    password
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Login failed');
            }

            const data = await response.json();
            this.setToken(data.access_token);
            
            // Fetch user info
            await this.fetchUserInfo();
            
            return data;
        } catch (error) {
            throw error;
        }
    }

    // Fetch current user info
    async fetchUserInfo() {
        try {
            const response = await fetch(`${this.apiBase}/auth/me`, {
                headers: this.getAuthHeaders()
            });

            if (response.ok) {
                const user = await response.json();
                this.setCurrentUser(user);
                return user;
            }
            return null;
        } catch (error) {
            console.error('Failed to fetch user info:', error);
            return null;
        }
    }

    // Logout
    async logout() {
        try {
            await fetch(`${this.apiBase}/auth/logout`, {
                method: 'POST',
                headers: this.getAuthHeaders()
            });
        } catch (error) {
            console.error('Logout error:', error);
        } finally {
            this.removeToken();
            window.location.reload();
        }
    }

    // Show login modal for restricted actions
    showLoginModal() {
        const modal = document.getElementById('loginModal');
        if (modal) {
            modal.classList.remove('hidden');
        } else {
            // Create modal if it doesn't exist
            this.createLoginModal();
            document.getElementById('loginModal').classList.remove('hidden');
        }
    }

    // Create login modal
    createLoginModal() {
        const modalHTML = `
            <div id="loginModal" class="fixed inset-0 bg-black bg-opacity-50 z-50 hidden flex items-center justify-center">
                <div class="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4">
                    <h2 class="text-2xl font-bold mb-4 text-gray-900 dark:text-white">Authentication Required</h2>
                    <p class="text-gray-600 dark:text-gray-300 mb-6">
                        Please login or register to run solar analysis.
                    </p>
                    <div class="flex gap-4">
                        <button onclick="window.location.href='/?login=true'" class="flex-1 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
                            Login
                        </button>
                        <button onclick="window.location.href='/?register=true'" class="flex-1 bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700">
                            Register
                        </button>
                        <button onclick="document.getElementById('loginModal').classList.add('hidden')" class="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700">
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHTML);
    }

    // Handle API request with auth
    async apiRequest(url, options = {}) {
        const headers = {
            ...this.getAuthHeaders(),
            ...options.headers
        };

        const response = await fetch(url, {
            ...options,
            headers
        });

        if (response.status === 401) {
            // Not authenticated
            this.removeToken();
            if (options.requireAuth !== false) {
                this.showLoginModal();
            }
            throw new Error('Authentication required');
        }

        return response;
    }
}

// Global auth manager instance
const authManager = new AuthManager();

// Initialize auth on page load
document.addEventListener('DOMContentLoaded', () => {
    // Check if user is authenticated and fetch user info
    if (authManager.isAuthenticated()) {
        authManager.fetchUserInfo();
    }
    
    // Update UI based on auth state
    updateAuthUI();
});

// Update UI based on authentication state
function updateAuthUI() {
    const isAuthenticated = authManager.isAuthenticated();
    const user = authManager.getCurrentUser();

    // Update login/logout buttons
    const loginBtn = document.getElementById('loginBtn');
    const registerBtn = document.getElementById('registerBtn');
    const logoutBtn = document.getElementById('logoutBtn');
    const userInfo = document.getElementById('userInfo');

    if (isAuthenticated && user) {
        if (loginBtn) loginBtn.classList.add('hidden');
        if (registerBtn) registerBtn.classList.add('hidden');
        if (logoutBtn) logoutBtn.classList.remove('hidden');
        if (userInfo) {
            userInfo.textContent = user.username || user.email;
            userInfo.classList.remove('hidden');
        }
    } else {
        if (loginBtn) loginBtn.classList.remove('hidden');
        if (registerBtn) registerBtn.classList.remove('hidden');
        if (logoutBtn) logoutBtn.classList.add('hidden');
        if (userInfo) userInfo.classList.add('hidden');
    }
}

// Export for use in other scripts
window.authManager = authManager;
window.updateAuthUI = updateAuthUI;

