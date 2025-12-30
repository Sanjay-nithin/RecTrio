// Authentication utilities

// Check if user is authenticated
function isAuthenticated() {
    return localStorage.getItem('access_token') !== null;
}

// Get access token
function getAccessToken() {
    return localStorage.getItem('access_token');
}

// Get user info
function getUserInfo() {
    const userStr = localStorage.getItem('user');
    return userStr ? JSON.parse(userStr) : null;
}

// Logout function
function logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
    window.location.href = '/login';
}

// Redirect to login if not authenticated
function requireAuth() {
    if (!isAuthenticated()) {
        window.location.href = '/login';
        return false;
    }
    return true;
}

// Make authenticated API call
async function authenticatedFetch(url, options = {}) {
    const token = getAccessToken();
    
    if (!token) {
        throw new Error('Not authenticated');
    }
    
    const headers = options.headers || {};
    
    // Don't set Content-Type for FormData - let browser handle it
    if (!(options.body instanceof FormData)) {
        headers['Content-Type'] = headers['Content-Type'] || 'application/json';
    }
    
    headers['Authorization'] = `Bearer ${token}`;
    
    const response = await fetch(url, {
        ...options,
        headers
    });
    
    // If unauthorized, logout and redirect
    if (response.status === 401) {
        logout();
        throw new Error('Session expired');
    }
    
    return response;
}
