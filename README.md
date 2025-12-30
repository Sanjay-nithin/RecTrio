# VisionRec - AI-Powered Image Recommendation System

VisionRec is an intelligent image recommendation system that combines similarity search with knowledge graph-based recommendations. The system uses OpenVINO-optimized CLIP models for image understanding and FAISS for efficient vector search, providing both visually similar images and semantically related recommendations.

## Features

- **Dual Search Modes**: Upload images or use text queries to find similar content
- **Knowledge Graph Integration**: Semantic relationships between entities for intelligent recommendations
- **Automatic Recommendations**: Get related suggestions based on your search history
- **Interactive Knowledge Graph Visualizer**: Explore entity relationships visually using D3.js
- **JWT Authentication**: Secure user authentication with Supabase PostgreSQL
- **Vector Database**: FAISS-based similarity search with 26,179+ image embeddings
- **Real-time Results**: Fast inference using OpenVINO CPU optimization

## Technology Stack

- **Backend**: Flask 3.0.0, Flask-JWT-Extended, SQLAlchemy
- **Database**: Supabase PostgreSQL
- **ML/AI**: OpenVINO 2023+, CLIP ViT-B/32, FAISS IndexFlatIP
- **Frontend**: Vanilla JavaScript, D3.js for visualization
- **Models**: Pre-trained CLIP models converted to OpenVINO IR format

## Project Structure

```
RecTrio/
├── app.py                          # Main Flask application
├── models.py                       # Database models (User, SearchHistory)
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (not in repo)
├── datasets/
│   └── animals/
│       └── raw-img/               # Dataset images organized by category
├── services/
│   └── recommendation_service.py   # Core ML inference and recommendation logic
├── static/
│   ├── css/                       # Stylesheets
│   └── js/                        # Frontend JavaScript
├── templates/                     # HTML templates
├── uploads/                       # User uploaded images
└── V1/
    ├── models/                    # OpenVINO model files (.xml, .bin)
    ├── notebooks/
    │   ├── build_embeddings.ipynb # Step 1: Build vector database
    │   └── inference.ipynb        # Step 2: Test inference
    └── vector_db/
        ├── embeddings.npy         # Image embeddings
        ├── faiss_index.bin        # FAISS index
        ├── metadata.pkl           # Image paths metadata
        └── animal_knowledge_graph.json  # Entity relationships
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Virtual environment tool (venv)
- Git
- Supabase account (for PostgreSQL database)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sanjay-nithin/RecTrio.git
cd RecTrio
```

### Step 2: Create and Activate Virtual Environment

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

### Step 3: Run Notebooks (First Time Setup)

Navigate to the notebooks directory and run them in order:

```bash
cd V1/notebooks
```

**Important:** Open and run the notebooks in Jupyter or VS Code in the following order:

1. **build_embeddings.ipynb**
   - Builds the FAISS vector database from your image dataset
   - Generates embeddings for all images
   - Creates metadata files
   - This may take several minutes depending on dataset size

2. **inference.ipynb**
   - Tests the inference pipeline
   - Validates that embeddings are working correctly
   - Ensures models are loaded properly

Return to project root after completion:
```bash
cd ../..
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables

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

### Step 6: Initialize Database

The application will automatically create database tables on first run. Ensure your Supabase database is accessible.

### Step 7: Run the Application

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

1. **Upload Image**: Click the upload area or drag and drop an image
2. **Or Use Text**: Switch to "Text Query" tab and enter a description (e.g., "cat", "butterfly")
3. **Set Results**: Choose number of results (5, 10, 15, or 20)
4. **Click Search**: View similar images immediately below
5. **Automatic Recommendations**: Related recommendations appear below similarity results

### Search Flow

1. User provides input (image or text)
2. System shows visually similar images
3. System automatically displays recommendations based on knowledge graph
4. Both results visible simultaneously

### Knowledge Graph Visualizer

1. Click "Knowledge Graph" in the navigation menu
2. Explore interactive visualization of entity relationships
3. **Hover** over nodes/edges to see details
4. **Click** nodes to view related entities
5. **Drag** nodes to reposition
6. **Zoom/Pan** to navigate the graph

### Understanding Results

- **Search Image Badge**: Green badge marks the original query image (100% match)
- **Labels**: Category names extracted from image paths
- **Strength Badges**: Shows relationship strength (strong, moderate, weak)
- **Relationship Score**: Percentage indicating semantic connection strength

## Dataset Structure

Organize your images in the following structure for automatic label extraction:

```
datasets/
└── your_dataset/
    ├── category1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── category2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
```

The system automatically extracts labels from folder names, making it scalable to any dataset.

## Knowledge Graph

The knowledge graph defines semantic relationships between entities. Edit `V1/vector_db/animal_knowledge_graph.json` to customize:

```json
{
  "entities": {
    "cat": {
      "related_entities": {
        "dog": 0.9,
        "lion": 0.7,
        "tiger": 0.8
      }
    }
  }
}
```

Relationship strengths range from 0.0 (weak) to 1.0 (strong).

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

## Troubleshooting

### Database Connection Issues
- Verify Supabase credentials in `.env`
- Check database URL format: `postgresql://user:password@host:port/database`
- Ensure database is accessible from your network

### Model Loading Errors
- Ensure notebooks were run successfully
- Check that model files exist in `V1/models/`
- Verify OpenVINO installation: `pip show openvino`

### Empty Results
- Confirm FAISS index exists: `V1/vector_db/faiss_index.bin`
- Check embeddings file: `V1/vector_db/embeddings.npy`
- Re-run `build_embeddings.ipynb` if files are missing

### JWT Authentication Errors
- Clear browser localStorage: Open console, run `localStorage.clear()`
- Generate new JWT secret key
- Restart Flask application

### Performance Issues
- Reduce number of results (5 instead of 20)
- Optimize FAISS index (use IVF index for large datasets)
- Enable CPU optimization flags in OpenVINO

## Development

### Adding New Entities

1. Add images to `datasets/your_dataset/new_entity/`
2. Update knowledge graph in `animal_knowledge_graph.json`
3. Re-run `build_embeddings.ipynb` to rebuild index
4. Restart application

### Customizing Recommendations

Edit `services/recommendation_service.py`:
- Modify relationship strength thresholds
- Adjust mixing ratios (strong/moderate/weak)
- Change similarity scoring algorithms

## Security Notes

- Never commit `.env` file to version control
- Use strong JWT secret keys in production
- Enable HTTPS in production environments
- Regularly update dependencies for security patches

## Performance Optimization

- For datasets over 100k images, use FAISS IVF index
- Enable OpenVINO threading: `ie.set_config({"CPU_THREADS_NUM": "4"})`
- Use image preprocessing caching
- Implement Redis for search history caching

## License

This project is for educational and research purposes.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

## Acknowledgments

- OpenVINO toolkit for optimized inference
- CLIP model by OpenAI
- FAISS by Facebook AI Research
- Supabase for database infrastructure
