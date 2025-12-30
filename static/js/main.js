// Main application JavaScript

// Global state
let currentInputType = 'image'; // 'image' or 'text'
let selectedFile = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Check authentication
    if (!requireAuth()) return;
    
    // Display user info
    const user = getUserInfo();
    if (user) {
        document.getElementById('userName').textContent = `Welcome, ${user.username}`;
    }
    
    // Load recent searches
    loadSearchHistory();
    
    // Setup event listeners
    setupEventListeners();
});

// Setup all event listeners
function setupEventListeners() {
    // Image upload area
    const uploadArea = document.getElementById('uploadArea');
    const imageFile = document.getElementById('imageFile');
    
    // Click to upload - make sure this works properly
    uploadArea.addEventListener('click', (e) => {
        e.stopPropagation();
        imageFile.click();
    });
    
    imageFile.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#2563eb';
        uploadArea.style.background = '#eff6ff';
    });
    
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#d0d0d0';
        uploadArea.style.background = '';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#d0d0d0';
        uploadArea.style.background = '';
        
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    // Search form
    document.getElementById('searchForm').addEventListener('submit', handleSearch);
}

// Handle file selection
function handleFileSelect(file) {
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImg').src = e.target.result;
        document.querySelector('.upload-placeholder').style.display = 'none';
        document.getElementById('imagePreview').style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// Clear image
function clearImage() {
    selectedFile = null;
    document.getElementById('imageFile').value = '';
    document.querySelector('.upload-placeholder').style.display = 'block';
    document.getElementById('imagePreview').style.display = 'none';
}

// Switch input type (image/text)
function switchInputType(type) {
    currentInputType = type;
    
    // Update button states
    document.getElementById('imageTabBtn').classList.toggle('active', type === 'image');
    document.getElementById('textTabBtn').classList.toggle('active', type === 'text');
    
    // Toggle sections
    document.getElementById('imageInput').style.display = type === 'image' ? 'block' : 'none';
    document.getElementById('textInput').style.display = type === 'text' ? 'block' : 'none';
}

// Handle search submission
async function handleSearch(e) {
    e.preventDefault();
    
    const btn = document.getElementById('searchBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');
    
    try {
        // Validate input
        if (currentInputType === 'image' && !selectedFile) {
            showError('Please select an image');
            return;
        }
        
        if (currentInputType === 'text' && !document.getElementById('textQuery').value.trim()) {
            showError('Please enter a text query');
            return;
        }
        
        // Show loading state
        btn.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-block';
        
        // Prepare form data
        const formData = new FormData();
        formData.append('top_k', document.getElementById('topK').value);
        
        if (currentInputType === 'image') {
            formData.append('image', selectedFile);
        } else {
            formData.append('text', document.getElementById('textQuery').value);
        }
        
        // Make similarity search API call
        const response = await authenticatedFetch('/api/similarity-search', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displaySimilarityResults(data);
            loadSearchHistory(); // Refresh history
            
            // Automatically load recommendations after similarity search
            await loadAutoRecommendations();
        } else {
            // Show detailed error
            console.error('API Error:', data);
            const errorMsg = data.error || 'Search failed';
            const detailMsg = data.detail ? `\n\nDetails: ${data.detail}` : '';
            showError(errorMsg + detailMsg);
        }
        
    } catch (error) {
        console.error('Search error:', error);
        showError(`Network error: ${error.message}. Please try again.`);
    } finally {
        // Reset button state
        btn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

// Display similarity search results
function displaySimilarityResults(data) {
    const similaritySection = document.getElementById('similaritySection');
    const similarityGrid = document.getElementById('similarityGrid');
    const similarityInfo = document.getElementById('similarityInfo');
    
    similaritySection.style.display = 'block';
    similarityGrid.innerHTML = '';
    
    const results = data.results || [];
    const queryEntity = data.query_entity || 'unknown';
    
    let infoText = `Found ${results.length} similar images`;
    if (queryEntity && queryEntity !== 'unknown') {
        infoText += ` (Query: ${queryEntity})`;
    }
    
    similarityInfo.innerHTML = infoText;
    
    // Render result cards
    results.forEach(result => {
        const card = createResultCard(result, false);
        similarityGrid.appendChild(card);
    });
    
    // Scroll to results
    similaritySection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display recommendations results
function displayRecommendations(data) {
    const recommendationsSection = document.getElementById('recommendationsSection');
    const recommendationsGrid = document.getElementById('recommendationsGrid');
    const recommendationsInfo = document.getElementById('recommendationsInfo');
    
    const results = data.recommendations || [];
    
    if (results.length === 0) {
        recommendationsSection.style.display = 'none';
        return;
    }
    
    recommendationsSection.style.display = 'block';
    recommendationsGrid.innerHTML = '';
    
    const entity = data.query_entity || 'unknown';
    const related = data.related_entities || [];
    const distribution = data.strength_distribution || {};
    
    let infoText = `Based on "${entity}" - Related: ${related.join(', ')}<br>`;
    infoText += `Strong: ${distribution.strong || 0} | Moderate: ${distribution.moderate || 0} | Weak: ${distribution.weak || 0}`;
    
    recommendationsInfo.innerHTML = infoText;
    
    // Render result cards
    results.forEach(result => {
        const card = createResultCard(result, true);
        recommendationsGrid.appendChild(card);
    });
}

// Create result card element
function createResultCard(result, isRecommendation) {
    const card = document.createElement('div');
    card.className = 'result-card';
    
    // Use label field directly from result
    const label = result.label || result.matched_label || result.result_entity || 'unknown';
    const similarity = result.similarity;
    const strengthCategory = result.strength_category;
    const relationshipStrength = result.relationship_strength;
    
    let strengthBadge = '';
    if (isRecommendation && strengthCategory) {
        strengthBadge = `<span class="strength-badge strength-${strengthCategory}">${strengthCategory}</span>`;
    }
    
    // Check if this is the search image (100% match)
    const isSearchImage = similarity >= 0.999;
    
    let detailsHTML = '';
    if (isSearchImage) {
        detailsHTML = `
            <div class="search-image-badge">
                <span>Search Image</span>
            </div>
        `;
    }
    
    if (relationshipStrength && !isSearchImage && isRecommendation) {
        detailsHTML += `
            <div class="similarity-score">
                <span>Rel: ${(relationshipStrength * 100).toFixed(0)}%</span>
            </div>
        `;
    }
    
    card.innerHTML = `
        <img class="result-image" src="${result.path}" alt="${label}" 
             onerror="this.src='/static/images/placeholder.png'">
        <div class="result-info">
            <div class="result-label">${label} ${strengthBadge}</div>
            <div class="result-details">
                ${detailsHTML}
            </div>
        </div>
    `;
    
    return card;
}

// Load search history
async function loadSearchHistory() {
    try {
        const response = await authenticatedFetch('/api/history');
        const data = await response.json();
        
        if (response.ok && data.history && data.history.length > 0) {
            document.getElementById('recentSearches').style.display = 'block';
            const historyDiv = document.getElementById('searchHistory');
            historyDiv.innerHTML = '';
            
            data.history.forEach(search => {
                const item = document.createElement('div');
                item.className = 'history-item';
                
                const searchType = search.search_type === 'similarity' ? 'Similarity' : 'Recommendation';
                const queryType = search.query_type === 'image' ? 'Image' : 'Text';
                const query = search.query_text || search.query_entity || 'Image query';
                
                item.textContent = `${searchType} | ${queryType} | ${query}`;
                historyDiv.appendChild(item);
            });
        }
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

// Load automatic recommendations based on last search
async function loadAutoRecommendations() {
    try {
        const topK = document.getElementById('topK').value;
        const response = await authenticatedFetch(`/api/auto-recommendations?top_k=${topK}`);
        const data = await response.json();
        
        console.log('Auto-recommendations response:', data);
        console.log('Number of recommendations:', data.recommendations?.length || 0);
        
        if (response.ok) {
            if (data.recommendations && data.recommendations.length > 0) {
                console.log('Auto-loading recommendations for:', data.query_entity);
                displayRecommendations(data);
            } else {
                // Hide recommendations section if no recommendations
                document.getElementById('recommendationsSection').style.display = 'none';
                console.log('No recommendations available. Message:', data.message);
                console.log('Query entity:', data.query_entity);
                console.log('Related entities:', data.related_entities);
            }
        } else {
            console.error('Failed to load auto recommendations:', data.error);
        }
    } catch (error) {
        console.error('Error loading auto recommendations:', error);
    }
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}
