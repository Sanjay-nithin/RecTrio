// Knowledge Graph Visualizer JavaScript

let graphData = null;
let svg = null;
let simulation = null;
let showLabels = true;

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    loadUserInfo();
    await loadKnowledgeGraph();
});

// Load user information
async function loadUserInfo() {
    try {
        const token = localStorage.getItem('access_token');
        const response = await fetch('/api/auth/me', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            document.getElementById('userName').textContent = data.name || data.email;
        }
    } catch (error) {
        console.error('Error loading user info:', error);
    }
}

// Logout function
function logout() {
    localStorage.removeItem('access_token');
    window.location.href = '/login';
}

// Load knowledge graph data
async function loadKnowledgeGraph() {
    try {
        const token = localStorage.getItem('access_token');
        const response = await fetch('/api/knowledge-graph', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            throw new Error('Failed to load knowledge graph');
        }

        const data = await response.json();
        graphData = data;

        // Update statistics
        document.getElementById('totalEntities').textContent = data.statistics.total_entities;
        document.getElementById('totalRelationships').textContent = data.statistics.total_relationships;
        document.getElementById('avgStrength').textContent = data.statistics.average_strength.toFixed(2);

        // Hide loading state
        document.getElementById('loading').style.display = 'none';

        // Render graph
        renderGraph(data.graph);

        // Setup controls
        setupControls();

    } catch (error) {
        console.error('Error loading knowledge graph:', error);
        document.getElementById('loading').style.display = 'none';
        showError('Failed to load knowledge graph: ' + error.message);
    }
}

// Render the graph using D3.js
function renderGraph(kgData) {
    const entities = kgData.entities;

    // Prepare nodes
    const nodes = Object.keys(entities).map(name => ({
        id: name,
        name: name,
        connections: Object.keys(entities[name].related_entities || {}).length
    }));

    // Prepare links - FILTER to reduce clutter
    // Only show relationships above a threshold and limit per node
    const links = [];
    const minStrength = 0.70; // Only show strong relationships (0.70+)
    const maxLinksPerNode = 5; // Limit to top 5 connections per node
    
    Object.entries(entities).forEach(([source, data]) => {
        // Get all relationships for this source
        const relationships = Object.entries(data.related_entities || {})
            .map(([target, strength]) => ({ target, strength }))
            .filter(rel => rel.strength >= minStrength) // Filter by strength
            .sort((a, b) => b.strength - a.strength) // Sort by strength descending
            .slice(0, maxLinksPerNode); // Take top N
        
        // Add filtered relationships
        relationships.forEach(rel => {
            // Avoid duplicate links (A-B and B-A)
            const existingLink = links.find(l => 
                (l.source === source && l.target === rel.target) ||
                (l.source === rel.target && l.target === source)
            );
            
            if (!existingLink) {
                links.push({
                    source: source,
                    target: rel.target,
                    strength: rel.strength
                });
            }
        });
    });

    console.log(`Rendering ${nodes.length} nodes and ${links.length} links (filtered from ${Object.keys(entities).length * 10} potential)`);

    // Set up SVG
    const container = document.getElementById('graphSvg');
    const width = container.clientWidth;
    const height = container.clientHeight;

    svg = d3.select('#graphSvg')
        .attr('width', width)
        .attr('height', height);

    // Clear any existing content
    svg.selectAll('*').remove();

    // Create group for zoom
    const g = svg.append('g');

    // Setup zoom
    const zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);

    // Store zoom for reset functionality
    svg.zoom = zoom;

    // Create force simulation with better spacing
    simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(200)) // Increased from 150
        .force('charge', d3.forceManyBody().strength(-1200)) // Increased repulsion from -800
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(70)); // Increased from 50;

    // Create links
    const link = g.append('g')
        .selectAll('line')
        .data(links)
        .enter()
        .append('line')
        .attr('class', 'link')
        .attr('stroke', d => getStrengthColor(d.strength))
        .attr('stroke-width', d => Math.max(2, d.strength * 6))
        .on('mouseover', function(event, d) {
            showTooltip(event, `${d.source.id} ↔ ${d.target.id}<br/>Strength: ${d.strength.toFixed(2)}`);
        })
        .on('mouseout', hideTooltip);

    // Create nodes
    const node = g.append('g')
        .selectAll('g')
        .data(nodes)
        .enter()
        .append('g')
        .attr('class', 'node')
        .call(d3.drag()
            .on('start', dragStarted)
            .on('drag', dragged)
            .on('end', dragEnded))
        .on('click', (event, d) => showEntityDetails(d.id));

    // Add circles to nodes - reduced size for better visibility
    node.append('circle')
        .attr('class', 'node-circle')
        .attr('r', d => 12 + Math.min(d.connections, 5) * 2) // Reduced size and capped growth
        .attr('fill', d => getNodeColor(d.connections))
        .on('mouseover', function(event, d) {
            d3.select(this).attr('r', 12 + Math.min(d.connections, 5) * 2 + 5);
            showTooltip(event, `${d.name}<br/>Connections: ${d.connections}`);
        })
        .on('mouseout', function(event, d) {
            d3.select(this).attr('r', 12 + Math.min(d.connections, 5) * 2);
            hideTooltip();
        });

    // Add labels to nodes
    const labels = node.append('text')
        .attr('class', 'node-label')
        .attr('dy', d => 12 + Math.min(d.connections, 5) * 2 + 12) // Adjusted position
        .attr('font-size', '11px') // Slightly smaller text
        .text(d => d.name.replace('_', ' ')) // Replace underscores with spaces
        .style('display', showLabels ? 'block' : 'none');

    // Update positions on simulation tick
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
}

// Get color based on strength
function getStrengthColor(strength) {
    if (strength >= 0.9) return '#dc2626'; // Very strong (0.9+) - red
    if (strength >= 0.75) return '#f59e0b'; // Strong (0.75-0.89) - orange
    if (strength >= 0.6) return '#3b82f6'; // Moderate (0.6-0.74) - blue
    return '#9ca3af'; // Weak - gray (won't be shown due to filter)
}

// Get node color based on connections
function getNodeColor(connections) {
    if (connections >= 5) return '#7c3aed'; // Purple for highly connected
    if (connections >= 3) return '#2563eb'; // Blue for moderately connected
    return '#10b981'; // Green for less connected
}

// Drag functions
function dragStarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragEnded(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// Show tooltip
function showTooltip(event, html) {
    const tooltip = document.getElementById('tooltip');
    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    tooltip.style.left = (event.pageX + 10) + 'px';
    tooltip.style.top = (event.pageY + 10) + 'px';
}

// Hide tooltip
function hideTooltip() {
    document.getElementById('tooltip').style.display = 'none';
}

// Show entity details panel
function showEntityDetails(entityName) {
    const entities = graphData.graph.entities;
    const entity = entities[entityName];

    if (!entity) return;

    document.getElementById('entityName').textContent = entityName.charAt(0).toUpperCase() + entityName.slice(1);

    const relatedEntitiesDiv = document.getElementById('relatedEntities');
    relatedEntitiesDiv.innerHTML = '';

    const relatedEntities = entity.related_entities || {};
    const sortedEntities = Object.entries(relatedEntities).sort((a, b) => b[1] - a[1]);

    sortedEntities.forEach(([name, strength]) => {
        const strengthClass = getStrengthClass(strength);
        const item = document.createElement('div');
        item.className = `related-item ${strengthClass}`;
        item.innerHTML = `
            <span class="related-name">${name.charAt(0).toUpperCase() + name.slice(1)}</span>
            <span class="related-strength">${strength.toFixed(2)}</span>
        `;
        relatedEntitiesDiv.appendChild(item);
    });

    document.getElementById('entityDetails').style.display = 'block';
}

// Close entity details panel
function closeEntityDetails() {
    document.getElementById('entityDetails').style.display = 'none';
}

// Get strength class for styling
function getStrengthClass(strength) {
    if (strength >= 0.8) return 'very-strong';
    if (strength >= 0.6) return 'strong';
    if (strength >= 0.4) return 'moderate';
    return 'weak';
}

// Setup control buttons
function setupControls() {
    document.getElementById('resetZoom').addEventListener('click', () => {
        if (svg && svg.zoom) {
            svg.transition().duration(750).call(svg.zoom.transform, d3.zoomIdentity);
        }
    });

    document.getElementById('toggleLabels').addEventListener('click', () => {
        showLabels = !showLabels;
        d3.selectAll('.node-label').style('display', showLabels ? 'block' : 'none');
    });
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

// Handle window resize
window.addEventListener('resize', () => {
    if (graphData) {
        renderGraph(graphData.graph);
    }
});
