"""
Flask Application for RecTrio - Image Recommendation System
With JWT Authentication and Supabase PostgreSQL Integration
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, 
    get_jwt_identity, get_jwt
)
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
import traceback
from pathlib import Path
from dotenv import load_dotenv

from models import db, User, SearchHistory, init_db
# Note: cleanup_old_searches deprecated - search history now in localStorage
from services.recommendation_service import RecommendationService

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SUPABASE_DB_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}

# JWT Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'dev-secret-key')
app.config['JWT_ALGORITHM'] = os.getenv('JWT_ALGORITHM', 'HS256')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(
    hours=int(os.getenv('JWT_EXPIRATION_HOURS', 24))
)

# File upload configuration
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16777216))
ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'png,jpg,jpeg,gif').split(','))

# Initialize extensions
CORS(app)
jwt = JWTManager(app)

# Initialize database only if explicitly requested
# Set INIT_DB=true in environment to create tables
if os.getenv('INIT_DB', 'false').lower() == 'true':
    init_db(app)
    print("✓ Database initialization enabled (INIT_DB=true)")
else:
    db.init_app(app)
    print("✓ Database connected (tables not created - set INIT_DB=true to initialize)")

# Initialize recommendation service
MODEL_DIR = Path(os.getenv('MODEL_DIR', 'V1/models'))
VECTOR_DB_DIR = Path(os.getenv('VECTOR_DB_DIR', 'V1/vector_db'))
rec_service = RecommendationService(MODEL_DIR, VECTOR_DB_DIR)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==================== Authentication Endpoints ====================

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """User registration endpoint"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if user exists
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 409
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 409
        
        # Create new user
        user = User(
            username=data['username'],
            email=data['email']
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        # Create access token - JWT identity must be a string
        access_token = create_access_token(identity=str(user.id))
        
        return jsonify({
            'message': 'User created successfully',
            'access_token': access_token,
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        
        if not data or not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Missing username or password'}), 400
        
        # Find user
        user = User.query.filter_by(username=data['username']).first()
        
        if not user or not user.check_password(data['password']):
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Create access token - JWT identity must be a string
        access_token = create_access_token(identity=str(user.id))
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get current user info"""
    try:
        user_id = int(get_jwt_identity())  # Convert string back to int
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({'user': user.to_dict()}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== Search History Endpoints (DEPRECATED - Now using localStorage) ====================

# @app.route('/api/history', methods=['GET'])
# @jwt_required()
# def get_search_history():
#     """Get user's search history (last 3 searches) - DEPRECATED: Now using browser localStorage"""
#     try:
#         user_id = int(get_jwt_identity())  # Convert string back to int
#         user = User.query.get(user_id)
#         
#         if not user:
#             return jsonify({'error': 'User not found'}), 404
#         
#         recent_searches = user.get_recent_searches(limit=3)
#         
#         return jsonify({
#             'history': [search.to_dict() for search in recent_searches]
#         }), 200
#         
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# DEPRECATED: Search history now stored in browser localStorage
# def save_search_history(user_id, search_type, query_type, query_text=None, 
#                        query_image_path=None, query_entity=None, 
#                        results_count=0, search_metadata=None):
#     """Save search to history and cleanup old searches - DEPRECATED"""
#     try:
#         print(f"Saving search history:")
#         print(f"  - user_id: {user_id}")
#         print(f"  - search_type: {search_type}")
#         print(f"  - query_type: {query_type}")
#         print(f"  - query_entity: {query_entity}")
#         print(f"  - query_text: {query_text}")
#         print(f"  - query_image_path: {query_image_path}")
#         
#         search = SearchHistory(
#             user_id=user_id,
#             search_type=search_type,
#             query_type=query_type,
#             query_text=query_text,
#             query_image_path=query_image_path,
#             query_entity=query_entity,
#             results_count=results_count,
#             search_metadata=search_metadata
#         )
#         
#         db.session.add(search)
#         db.session.commit()
#         
#         print(f"Search history saved successfully with query_entity: {query_entity}")
#         
#         # Keep only last 3 searches
#         cleanup_old_searches(user_id, keep_count=3)
#         
#     except Exception as e:
#         print(f"Error saving search history: {e}")
#         db.session.rollback()


# ==================== Recommendation Endpoints ====================

@app.route('/api/similarity-search', methods=['POST'])
@jwt_required()
def similarity_search():
    """Similarity search endpoint (image or text)"""
    try:
        user_id = int(get_jwt_identity())  # Convert string back to int
        
        # Debug logging
        print(f"Request form keys: {list(request.form.keys())}")
        print(f"Request files keys: {list(request.files.keys())}")
        print(f"Content-Type: {request.content_type}")
        
        # Get top_k parameter
        try:
            top_k = int(request.form.get('top_k', 10))
        except (ValueError, TypeError):
            top_k = 10
        
        # Check if image or text query
        if 'image' in request.files:
            file = request.files['image']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print(f"Processing image: {filepath}")
            
            # Perform similarity search
            search_result = rec_service.similarity_search(filepath, input_type='image', top_k=top_k)
            results = search_result['results']
            query_entity = search_result.get('query_entity')
            
            # If entity not found, try extracting from uploaded filename
            if not query_entity and file and file.filename:
                filename_lower = file.filename.lower()
                for entity in rec_service.knowledge_graph['entities'].keys():
                    if entity in filename_lower:
                        query_entity = entity
                        print(f"Entity extracted from filename: {query_entity}")
                        break
            
            # If still no entity, extract from first result's label
            if not query_entity and results and len(results) > 0:
                query_entity = results[0].get('label')
                if query_entity and query_entity != 'unknown':
                    print(f"Entity extracted from first result: {query_entity}")
            
            print(f"Similarity search completed - query_entity: {query_entity}, results: {len(results)}")
            
            # Note: Search history is now stored in browser localStorage for fast retrieval
            
        elif 'text' in request.form and request.form['text'].strip():
            text_query = request.form['text'].strip()
            
            print(f"Processing text query: {text_query}")
            
            # Perform similarity search
            search_result = rec_service.similarity_search(text_query, input_type='text', top_k=top_k)
            results = search_result['results']
            query_entity = search_result.get('query_entity')
            
            # If still no entity, extract from first result's label
            if not query_entity and results and len(results) > 0:
                query_entity = results[0].get('label')
                if query_entity and query_entity != 'unknown':
                    print(f"Entity extracted from first result: {query_entity}")
            
            print(f"Similarity search completed - query_entity: {query_entity}, results: {len(results)}")
            
            # Note: Search history is now stored in browser localStorage for fast retrieval
        else:
            error_msg = 'No image or text query provided. '
            error_msg += f'Form keys: {list(request.form.keys())}, '
            error_msg += f'File keys: {list(request.files.keys())}'
            print(f"ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 422
        
        return jsonify({
            'results': results,
            'query_entity': query_entity,
            'count': len(results)
        }), 200
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"ERROR in similarity_search: {error_detail}")
        return jsonify({'error': str(e), 'detail': error_detail}), 500


@app.route('/api/recommendations', methods=['POST'])
@jwt_required()
def get_recommendations():
    """Knowledge graph-based recommendations (mixes moderate and strong)"""
    try:
        user_id = int(get_jwt_identity())  # Convert string back to int
        
        # Debug logging
        print(f"Recommendations - Form keys: {list(request.form.keys())}")
        print(f"Recommendations - Files keys: {list(request.files.keys())}")
        
        # Get top_k parameter
        try:
            top_k = int(request.form.get('top_k', 10))
        except (ValueError, TypeError):
            top_k = 10
        
        # Check if image or text query
        if 'image' in request.files:
            file = request.files['image']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print(f"Processing image for recommendations: {filepath}")
            
            # Get recommendations
            results = rec_service.kg_based_recommendation(
                filepath, 
                input_type='image', 
                top_k=top_k,
                mix_all_strengths=True  # Mix moderate and strong
            )
            
            # Note: Search history is now stored in browser localStorage for fast retrieval
            
        elif 'text' in request.form and request.form['text'].strip():
            text_query = request.form['text'].strip()
            
            print(f"Processing text for recommendations: {text_query}")
            
            # Get recommendations
            results = rec_service.kg_based_recommendation(
                text_query, 
                input_type='text', 
                top_k=top_k,
                mix_all_strengths=True  # Mix moderate and strong
            )
            
            # Note: Search history is now stored in browser localStorage for fast retrieval
        else:
            error_msg = 'No image or text query provided. '
            error_msg += f'Form keys: {list(request.form.keys())}, '
            error_msg += f'File keys: {list(request.files.keys())}'
            print(f"ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 422
        
        return jsonify(results), 200
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"ERROR in get_recommendations: {error_detail}")
        return jsonify({'error': str(e), 'detail': error_detail}), 500


@app.route('/api/auto-recommendations', methods=['GET'])
@jwt_required()
def get_auto_recommendations():
    """
    Get automatic recommendations based on entity name
    Uses knowledge graph to find related entities from vector DB
    """
    try:
        # Get entity from query params (from localStorage in frontend)
        entity_name = request.args.get('entity')
        
        if not entity_name:
            return jsonify({
                'message': 'No entity provided',
                'recommendations': [],
                'query_entity': None,
                'related_entities': []
            }), 200
        
        print(f"Auto-recommendations for entity: {entity_name}")
        
        # Get top_k from query params
        top_k = int(request.args.get('top_k', 10))
        
        # Get automatic recommendations
        results = rec_service.get_auto_recommendations(
            entity_name=entity_name,
            top_k=top_k,
            mix_all_strengths=True
        )
        
        print(f"Auto-recommendations results: {len(results.get('recommendations', []))} items")
        print(f"Query entity: {results.get('query_entity')}")
        print(f"Related entities: {results.get('related_entities')}")
        
        return jsonify(results), 200
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"ERROR in get_auto_recommendations: {error_detail}")
        return jsonify({'error': str(e), 'detail': error_detail}), 500


# ==================== Knowledge Graph ====================

@app.route('/api/knowledge-graph', methods=['GET'])
@jwt_required()
def get_knowledge_graph():
    """
    Get the knowledge graph data for visualization
    Returns the complete KG JSON with entities and relationships
    """
    try:
        import json
        
        # Load knowledge graph from file
        kg_path = os.path.join('V1', 'vector_db', 'fashion_knowledge_graph.json')
        
        if not os.path.exists(kg_path):
            return jsonify({'error': 'Knowledge graph file not found'}), 404
        
        with open(kg_path, 'r') as f:
            kg_data = json.load(f)
        
        # Calculate some statistics
        entities = kg_data.get('entities', {})
        total_entities = len(entities)
        
        # Count total relationships
        total_relationships = sum(len(entity.get('related_entities', {})) for entity in entities.values())
        
        # Calculate average strength
        all_strengths = []
        for entity in entities.values():
            all_strengths.extend(entity.get('related_entities', {}).values())
        
        avg_strength = sum(all_strengths) / len(all_strengths) if all_strengths else 0.0
        
        # Add statistics to response
        response_data = {
            'graph': kg_data,
            'statistics': {
                'total_entities': total_entities,
                'total_relationships': total_relationships,
                'average_strength': round(avg_strength, 2)
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"ERROR in get_knowledge_graph: {error_detail}")
        return jsonify({'error': str(e), 'detail': error_detail}), 500


# ==================== Static Files ====================

@app.route('/')
def index():
    """Serve main page (protected, will redirect to login via JS if not authenticated)"""
    return send_from_directory('templates', 'index.html')


@app.route('/login')
def login_page():
    """Serve login page"""
    return send_from_directory('templates', 'login.html')


@app.route('/signup')
def signup_page():
    """Serve signup page (default entry point)"""
    return send_from_directory('templates', 'signup.html')


@app.route('/kg-visualizer')
def kg_visualizer_page():
    """Serve knowledge graph visualizer page"""
    return send_from_directory('templates', 'kg-visualizer.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/datasets/<path:filepath>')
def serve_dataset(filepath):
    """Serve dataset images"""
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    return send_from_directory(datasets_dir, filepath)


# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"Starting RecTrio server on port {port}")
    print(f"Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
