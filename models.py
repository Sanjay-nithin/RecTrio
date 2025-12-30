"""
Database models for RecTrio application
Uses SQLAlchemy with Supabase PostgreSQL
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import json

db = SQLAlchemy()


class User(db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    search_history = db.relationship('SearchHistory', backref='user', lazy='dynamic', 
                                    cascade='all, delete-orphan',
                                    order_by='SearchHistory.created_at.desc()')
    
    def set_password(self, password):
        """Hash and set user password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def get_recent_searches(self, limit=3):
        """Get the last N searches for this user (default 3)"""
        return self.search_history.limit(limit).all()
    
    def to_dict(self):
        """Convert user to dictionary (exclude password)"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def __repr__(self):
        return f'<User {self.username}>'


class SearchHistory(db.Model):
    """Search history model to store user queries"""
    __tablename__ = 'search_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    search_type = db.Column(db.String(20), nullable=False)  # 'similarity' or 'recommendation'
    query_type = db.Column(db.String(20), nullable=False)   # 'image' or 'text'
    query_text = db.Column(db.Text)                         # Text query if applicable
    query_image_path = db.Column(db.String(500))            # Image path if applicable
    query_entity = db.Column(db.String(100))                # Detected entity from query
    results_count = db.Column(db.Integer)                   # Number of results returned
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Store related entities and search metadata as JSON (renamed from 'metadata' to avoid SQLAlchemy conflict)
    search_metadata = db.Column(db.JSON)  # Can store related_entities, strengths, etc.
    
    def to_dict(self):
        """Convert search history to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'search_type': self.search_type,
            'query_type': self.query_type,
            'query_text': self.query_text,
            'query_image_path': self.query_image_path,
            'query_entity': self.query_entity,
            'results_count': self.results_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'search_metadata': self.search_metadata
        }
    
    def __repr__(self):
        return f'<SearchHistory {self.id} - {self.search_type} by User {self.user_id}>'


def init_db(app):
    """Initialize database with app context"""
    db.init_app(app)
    with app.app_context():
        # Create all tables
        db.create_all()
        print("✓ Database tables created successfully")


def cleanup_old_searches(user_id, keep_count=3):
    """
    Keep only the last N searches for a user, delete older ones
    Called after adding a new search (default keeps 3)
    """
    user = User.query.get(user_id)
    if not user:
        return
    
    # Get all searches ordered by date
    all_searches = user.search_history.all()
    
    # Delete searches beyond the keep_count
    if len(all_searches) > keep_count:
        searches_to_delete = all_searches[keep_count:]
        for search in searches_to_delete:
            db.session.delete(search)
        db.session.commit()
        print(f"✓ Cleaned up {len(searches_to_delete)} old searches for user {user_id}")
