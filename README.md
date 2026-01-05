# VisionRec - Image Similarity Search and Recommendation System

VisionRec is an intelligent image similarity search and recommendation system that combines CLIP-based visual understanding with knowledge graph-based recommendations. The system uses OpenVINO-optimized CLIP models for efficient image embeddings and FAISS for vector similarity search. For testing purposes, this implementation uses a fashion dataset to demonstrate the capabilities of visual search and intelligent recommendations.

## Features

- **Dual Search Modes**: Upload images or use text queries to find similar content
- **Out-of-Distribution (OOD) Detection**: Semantic domain filtering rejects non-fashion items (e.g., bicycle pumps, tools, electronics)
- **Knowledge Graph Integration**: Semantic relationships between 30 fashion entities for intelligent recommendations
- **Smart Recommendations Fallback**: Tries last 3 searches for recommendations, only shows results when available
- **Curated Fashion Dataset**: 32,356 images across 29 fashion categories pruned based on knowledge graph
- **Interactive Knowledge Graph Visualizer**: Explore entity relationships visually using D3.js
- **JWT Authentication**: Secure user authentication with Supabase PostgreSQL
- **Vector Database**: FAISS-based similarity search with L2-normalized CLIP embeddings
- **Real-time Results**: Fast inference using OpenVINO CPU optimization with embedding cache

## Technology Stack

- **Backend**: Flask 3.0.0, Flask-JWT-Extended, SQLAlchemy
- **Database**: Supabase PostgreSQL
- **ML/AI**: OpenVINO 2024.6.0, CLIP ViT-B/32, FAISS IndexFlatIP (cosine similarity)
- **Frontend**: Vanilla JavaScript, D3.js for visualization
- **Models**: Pre-trained CLIP models converted to OpenVINO IR format
- **OOD Detection**: CLIP text-image alignment with 17 fashion + 17 non-fashion terms, 5% margin

## Project Structure

```
VisionRec/
├── app.py                          # Main Flask application
├── models.py                       # Database models (User, SearchHistory)
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (not in repo)
├── datasets/
│   ├── main.ipynb                 # Dataset preparation notebook (MUST RUN FIRST)
│   ├── styles.csv                 # Fashion dataset metadata
│   └── fashion/
│       └── (29 category folders)  # Organized after running main.ipynb
├── services/
│   ├── __init__.py
│   └── recommendation_service.py   # Core ML inference, OOD detection, recommendations
├── static/
│   ├── css/
│   │   └── main.css               # Includes confidence indicators, no-results styling
│   └── js/
│       └── main.js                # Frontend logic with smart recommendations fallback
├── templates/                     # HTML templates
├── uploads/                       # User uploaded images
└── V1/
    ├── models/                    # OpenVINO model files (.xml, .bin)
    ├── notebooks/
    │   ├── build_embeddings.ipynb # Step 2: Build vector database (29 categories)
    │   └── inference.ipynb        # Step 3: Test inference
    └── vector_db/
        ├── embeddings.npy         # 32,356 image embeddings (L2-normalized)
        ├── faiss_index.bin        # FAISS IndexFlatIP for cosine similarity
        ├── metadata.pkl           # Image paths metadata
        └── fashion_knowledge_graph.json  # 30 entities with relationships
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher (Python 3.11.7 recommended)
- Virtual environment tool (venv)
- Git
- Supabase account (for PostgreSQL database)
- Jupyter Notebook or VS Code with Jupyter extension

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sanjay-nithin/RecTrio.git
cd RecTrio
```

### Step 2: Download Fashion Dataset

Download the Fashion Product Images (Small) dataset from Kaggle:

**Dataset URL**: [Fashion-Product-images-Small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

1. Visit the Kaggle link above
2. Click "Download" (requires Kaggle account)
3. Extract the downloaded ZIP file
4. You should have a folder containing:
   - `images/` folder (44,441 fashion product images)
   - `styles.csv` file (metadata with product info)

### Step 3: Prepare Dataset Structure

**IMPORTANT**: Before building embeddings, you must run the dataset preparation notebook.

1. Copy the downloaded files to the `datasets/` directory:

   ```
   RecTrio/datasets/
   ├── images/          # From Kaggle download
   └── styles.csv       # From Kaggle download
   ```

2. Open and run `datasets/main.ipynb`:

   ```bash
   # If using Jupyter Notebook
   cd datasets
   jupyter notebook main.ipynb

   # If using VS Code, open main.ipynb and run all cells
   ```

3. The notebook will:

   - Read `styles.csv` metadata
   - Filter to 29 fashion categories based on knowledge graph
   - Organize images into category folders: `datasets/fashion/<category>/`
   - Process 32,356 images from original 44,441
   - Create organized structure for embedding generation

4. Verify the output structure:
   ```
   datasets/fashion/
   ├── Shirts/
   ├── Tshirts/
   ├── Casual Shoes/
   ├── Watches/
   ├── Sports Shoes/
   ├── Handbags/
   ├── Heels/
   └── ... (29 categories total)
   ```

### Step 4: Create and Activate Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:

- Flask 3.0.0
- OpenVINO 2024.6.0
- FAISS 1.13.2
- PyTorch 2.9.1
- SQLAlchemy 2.0.45
- And other dependencies

### Step 6: Build Vector Database

Navigate to the notebooks directory and run them in order:

```bash
cd V1/notebooks
```

**Important:** Open and run the notebooks in Jupyter or VS Code in the following order:

1. **build_embeddings.ipynb**

   - Loads OpenVINO CLIP model
   - Processes 32,356 images from 29 fashion categories
   - Generates L2-normalized embeddings (512-dimensional)
   - Builds FAISS IndexFlatIP for cosine similarity
   - Creates metadata files
   - **Time**: ~15-30 minutes depending on CPU (Intel i5/i7 recommended)
   - **Output**:
     - `V1/vector_db/embeddings.npy` (~160 MB)
     - `V1/vector_db/faiss_index.bin`
     - `V1/vector_db/metadata.pkl`

2. **inference.ipynb**
   - Tests the inference pipeline
   - Validates embeddings and similarity search
   - Tests OOD detection with non-fashion images
   - Ensures models are loaded properly
   - Verifies recommendation system

Return to project root after completion:

```bash
cd ../..
```

### Step 7: Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Flask Configuration
FLASK_DEBUG=True
PORT=5000

# JWT Secret Key (generate a secure random string)
JWT_SECRET_KEY=your-secret-key-here

# Supabase Database URL
SUPABASE_DB_URL=postgresql://user:password@host:port/database
```

To generate a secure JWT secret key:

```python
python -c "import secrets; print(secrets.token_hex(32))"
```

### Step 8: Initialize Database

The application will automatically create database tables on first run. Ensure your Supabase database is accessible.

### Step 9: Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000` by default.

## Using the Application

### First Time Access

1. Navigate to `http://localhost:5000`
2. You will be redirected to the signup page
3. Create an account with username, email, and password
4. Login with your credentials

### Similarity Search

1. **Upload Image**: Click the upload area or drag and drop a fashion image
2. **Or Use Text**: Switch to "Text Query" tab and enter a description (e.g., "red dress", "blue jeans", "leather shoes")
3. **Set Results**: Choose number of results (5, 10, 15, or 20)
4. **Click Search**: View similar images immediately below
5. **Smart Recommendations**: System tries last 3 searches and shows first successful recommendations

### Out-of-Distribution (OOD) Handling

The system intelligently filters non-fashion items:

**Accepted Fashion Items:**

- Clothing: Shirts, T-shirts, Dresses, Jeans, Shorts, Kurtas, Kurtis
- Footwear: Shoes, Heels, Sandals, Flip Flops, Flats
- Accessories: Watches, Sunglasses, Handbags, Wallets, Belts, Jewelry, Caps

**Rejected Non-Fashion Items:**

- Tools, Electronics, Vehicles, Sports Equipment
- Animals, Nature, Food
- Office Supplies, Furniture, Appliances

**When Non-Fashion Item is Uploaded:**

- Shows friendly error message: "No similar images found in the database"
- Explains the item is outside the fashion domain
- Automatically attempts to load recommendations from search history
- If no recommendations available, section remains hidden

### Search Flow

1. User uploads image or enters text query
2. **Domain Check (OOD Detection)**: System compares query against fashion and non-fashion embeddings
   - Calculates average similarity with 17 fashion terms (fashion_score)
   - Calculates average similarity with 17 non-fashion terms (non_fashion_score)
   - Applies threshold: if fashion_score \* 1.05 < non_fashion_score, query is rejected
3. **If Threshold Passed (Fashion Item)**: Proceeds to similarity search and shows similar images + auto-loads recommendations
4. **If Threshold Failed (Non-Fashion Item)**: Rejects query, shows error message + tries recommendations from history
5. **Recommendations Logic**:
   - Tries last search entity from localStorage
   - If no recommendations, tries second-to-last search
   - If still none, tries third-to-last search
   - Only displays recommendations section when successful

### Knowledge Graph Visualizer

1. Click "Knowledge Graph" in the navigation menu
2. Explore interactive visualization of 30 fashion entity relationships
3. **Hover** over nodes/edges to see relationship details
4. **Click** nodes to view related entities
5. **Drag** nodes to reposition
6. **Zoom/Pan** to navigate the graph

### Understanding Results

- **Search Image Badge**: Green badge marks the original query image (100% match)
- **Category Labels**: Fashion category names (Shirts, Dresses, Heels, etc.)
- **Confidence Indicators**: Color-coded bars showing match quality
  - **Excellent** (Green): 85%+ relationship strength
  - **High** (Blue): 70-85% relationship strength
  - **Medium** (Yellow): 50-70% relationship strength
  - **Low** (Orange): 30-50% relationship strength
  - **Very Low** (Red): Below 30% relationship strength
- **No Percentages**: Clean UI with quality labels only (no cluttering numbers)
- **Strength Badges**: For recommendations, shows strong/moderate/weak relationship category

## Dataset Structure

The Fashion Product Images (Small) dataset from Kaggle contains 44,441 images across 143 categories. RecTrio uses a **curated subset of 29 fashion categories** aligned with the knowledge graph:

```
datasets/fashion/
├── Shirts/              # 2,098 images
├── Tshirts/             # 1,965 images
├── Casual Shoes/        # 1,876 images
├── Watches/             # 1,687 images
├── Sports Shoes/        # 1,543 images
├── Handbags/            # 1,432 images
├── Heels/               # 1,287 images
├── Sunglasses/          # 1,165 images
├── Wallets/             # 1,043 images
├── Tops/                # 987 images
├── Flip Flops/          # 876 images
├── Belts/               # 765 images
├── Sandals/             # 654 images
├── Backpacks/           # 543 images
└── ... (29 categories, 32,356 total images)
```

The `datasets/main.ipynb` notebook automatically:

- Filters the original 44,441 images to 32,356 fashion items
- Organizes by category based on `styles.csv` metadata
- Creates folder structure for embedding generation
- Excludes 99 non-fashion or low-quality categories

## Knowledge Graph

The fashion knowledge graph defines semantic relationships between 30 fashion entities. The graph is located at `V1/vector_db/fashion_knowledge_graph.json`:

**Example Structure:**

```json
{
  "entities": {
    "Handbags": {
      "related_entities": {
        "Heels": 0.9,
        "Dresses": 0.85,
        "Tops": 0.8,
        "Sunglasses": 0.75,
        "Wallets": 0.7
      }
    },
    "Heels": {
      "related_entities": {
        "Dresses": 0.95,
        "Handbags": 0.9,
        "Clutches": 0.85,
        "Jewellery Set": 0.8
      }
    },
    "Tshirts": {
      "related_entities": {
        "Jeans": 0.95,
        "Shorts": 0.9,
        "Casual Shoes": 0.85,
        "Caps": 0.8,
        "Backpacks": 0.75
      }
    }
  }
}
```

**Relationship Strength Scale:**

- **0.90-1.00**: Very Strong (e.g., Heels + Dresses, Jeans + T-shirts)
- **0.75-0.89**: Strong (e.g., Handbags + Heels, Shirts + Trousers)
- **0.60-0.74**: Moderate (e.g., Watches + Casual Shoes)
- **0.40-0.59**: Weak (e.g., Caps + Formal Shoes)
- **0.00-0.39**: Very Weak (minimal relationship)

**30 Fashion Entities in Knowledge Graph:**
Shirts, Tshirts, Casual Shoes, Watches, Sports Shoes, Kurtas, Tops, Handbags, Heels, Sunglasses, Wallets, Flip Flops, Belts, Sandals, Shoe Accessories, Backpacks, Jeans, Jewellery Set, Flats, Shorts, Trousers, Kurtis, Formal Shoes, Dresses, Socks, Caps, Clutches, Mufflers, Innerwear, Track Pants

**How It's Used:**

1. User searches for "Handbags"
2. System finds similar handbag images
3. System loads recommendations from related entities (Heels: 0.90, Dresses: 0.85)
4. UI displays relationship strength as confidence indicators

## API Endpoints

### Authentication

- `POST /api/auth/signup` - Create new user account
- `POST /api/auth/login` - Login and receive JWT token
- `GET /api/auth/me` - Get current user information

### Search

- `POST /api/similarity-search` - Find similar images (image or text input)
- `POST /api/recommendations` - Get KG-based recommendations
- `GET /api/auto-recommendations` - Automatic recommendations from history

### Utility

- `GET /api/history` - Get user's recent searches
- `GET /api/knowledge-graph` - Get KG data for visualization

## Development

### Adding New Fashion Categories

1. Add images to `datasets/fashion/new_category/`
2. Update knowledge graph in `V1/vector_db/fashion_knowledge_graph.json`:
   ```json
   "New Category": {
     "related_entities": {
       "Existing Category": 0.85,
       "Another Category": 0.75
     }
   }
   ```
3. Re-run `V1/notebooks/build_embeddings.ipynb` to rebuild index
4. Restart application to load new knowledge graph

### Customizing OOD Detection

Edit `services/recommendation_service.py`:

**Add Fashion Terms:**

```python
fashion_terms = [
    "fashion", "clothing", "apparel", "shoes", ...,
    "your_new_term"  # Add here
]
```

**Adjust Margin:**

```python
def is_fashion_related(self, query_embedding, margin=1.05):  # Change margin here
    # 1.05 = 5% margin (default)
    # 1.10 = 10% margin (more lenient)
    # 1.03 = 3% margin (stricter)
```

### Customizing Recommendations

Edit `services/recommendation_service.py`:

**Modify Strength Thresholds:**

```python
# In get_recommendations_by_entity method
if strength >= 0.7:
    strength_category = "strong"
elif strength >= 0.5:
    strength_category = "moderate"
else:
    strength_category = "weak"
```

**Adjust Mixing Ratios:**

```python
# In get_recommendations_by_entity method
strong_count = min(len(strong_recommendations), max(1, int(top_k * 0.6)))  # 60% strong
moderate_count = min(len(moderate_recommendations), max(1, int(top_k * 0.3)))  # 30% moderate
weak_count = max(0, top_k - strong_count - moderate_count)  # Rest weak
```

**Change Similarity Thresholds:**

```python
# In search_similar_images method
def search_similar_images(self, query_embedding, top_k=10,
                         min_threshold=0.25,  # Minimum similarity (adjust here)
                         adaptive=True,
                         adaptive_ratio=0.70):  # Adaptive threshold (adjust here)
```

## Technical Details

### CLIP Model Architecture

- **Model**: OpenAI CLIP ViT-B/32
- **Embedding Dimension**: 512
- **Optimization**: OpenVINO IR format for CPU inference
- **Normalization**: L2-normalized embeddings for cosine similarity
- **Preprocessing**: 224x224 resize, RGB normalization

### FAISS Index Configuration

- **Index Type**: IndexFlatIP (Inner Product for cosine similarity)
- **Metric**: Cosine similarity via L2-normalized vectors
- **Size**: 32,356 vectors × 512 dimensions = ~160 MB
- **Search**: Exact nearest neighbor (no approximation)

### Recommendation Algorithm

1. **Input**: Query entity (e.g., "Handbags")
2. **Knowledge Graph Lookup**: Get related entities with strengths
3. **Categorization**:
   - Strong: strength ≥ 0.7 (e.g., Handbags → Heels: 0.90)
   - Moderate: 0.5 ≤ strength < 0.7
   - Weak: strength < 0.5
4. **Sampling**: 60% strong, 30% moderate, 10% weak
5. **FAISS Search**: Find similar images for each related entity
6. **Weighting**: `final_similarity = similarity × relationship_strength`
7. **Display**: Show relationship strength in UI confidence indicators

## Acknowledgments

- **OpenVINO Toolkit** by Intel for optimized CPU inference
- **CLIP Model** by OpenAI for vision-language understanding
- **FAISS** by Facebook AI Research for efficient similarity search
- **Supabase** for PostgreSQL database infrastructure
- **Kaggle** and dataset contributors for fashion product images
- **Flask** community for web framework
- **D3.js** for interactive knowledge graph visualization
