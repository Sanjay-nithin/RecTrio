"""
Recommendation Service for RecTrio
Handles both similarity search and knowledge graph-based recommendations
UPDATED: Mixes moderate and strong relationship matches
"""
import numpy as np
import faiss
import pickle
from pathlib import Path
import json
from PIL import Image
import torch
from torchvision import transforms
import clip
from openvino.runtime import Core
import os


class RecommendationService:
    def __init__(self, model_dir, vector_db_dir):
        """Initialize the recommendation service with models and vector database"""
        self.model_dir = Path(model_dir)
        self.vector_db_dir = Path(vector_db_dir)
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load OpenVINO models
        core = Core()
        self.text_model = core.read_model(str(self.model_dir / "clip_text_model.xml"))
        self.vision_model = core.read_model(str(self.model_dir / "clip_vision_model.xml"))
        
        self.text_compiled = core.compile_model(self.text_model, "CPU")
        self.vision_compiled = core.compile_model(self.vision_model, "CPU")
        
        # Get layer names
        self.text_output_layer = self.text_compiled.output(0)
        self.vision_input_layer = self.vision_compiled.input(0)
        self.vision_output_layer = self.vision_compiled.output(0)
        
        # Load FAISS index
        self.index = faiss.read_index(str(self.vector_db_dir / "faiss_index.bin"))
        
        # Load metadata
        with open(self.vector_db_dir / "metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.image_paths = self.metadata['image_paths']
        
        # Load knowledge graph
        kg_path = self.vector_db_dir / "animal_knowledge_graph.json"
        with open(kg_path, 'r') as f:
            self.knowledge_graph = json.load(f)
        
        print(f"✓ Loaded {len(self.image_paths)} images in vector database")
        print(f"✓ Loaded {len(self.knowledge_graph['entities'])} entities in knowledge graph")
    
    def get_text_embedding(self, text):
        """Get CLIP text embedding using OpenVINO"""
        try:
            print(f"Processing text query: {text}")
            text_tokens = clip.tokenize([text])
            
            print(f"Text tokens shape: {text_tokens.shape}")
            
            text_embedding = self.text_compiled([text_tokens.numpy()])[self.text_output_layer]
            text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=-1, keepdims=True)
            
            print(f"Text embedding shape: {text_embedding.shape}")
            
            return text_embedding.astype('float32')
        except Exception as e:
            print(f"Error in get_text_embedding: {e}")
            raise
    
    def get_image_embedding(self, image_path):
        """Get CLIP image embedding using OpenVINO"""
        try:
            print(f"Loading image from: {image_path}")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).numpy()
            
            print(f"Image tensor shape: {image_tensor.shape}")
            
            image_embedding = self.vision_compiled([image_tensor])[self.vision_output_layer]
            image_embedding = image_embedding / np.linalg.norm(image_embedding, axis=-1, keepdims=True)
            
            print(f"Image embedding shape: {image_embedding.shape}")
            
            return image_embedding.astype('float32')
        except Exception as e:
            print(f"Error in get_image_embedding: {e}")
            raise
    
    def search_similar_images(self, query_embedding, top_k=10):
        """Search for similar images using FAISS"""
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.image_paths):
                # Convert absolute path to web-accessible path
                image_path = self.image_paths[idx]
                # Extract the relative path from datasets onwards
                if 'datasets' in image_path:
                    # Convert to forward slashes and get path from datasets
                    web_path = image_path.replace('\\', '/')
                    datasets_idx = web_path.find('datasets')
                    if datasets_idx != -1:
                        web_path = '/' + web_path[datasets_idx:]
                else:
                    web_path = image_path
                
                # Extract entity/label from path
                entity = self.find_entity_from_path(image_path)
                
                results.append({
                    'path': web_path,
                    'similarity': float(distance),
                    'label': entity if entity else 'unknown'
                })
        
        return results
    
    def find_entity_from_path(self, image_path):
        """
        Extract entity/label name from image path
        Scalable approach: checks folder names in path against known entities
        Falls back to parent folder name if not in knowledge graph
        
        Args:
            image_path: Full or relative path to image
            
        Returns:
            Entity name (lowercase) or None if not found
        """
        if not image_path:
            return None
            
        # Convert to Path object for easier manipulation
        path_obj = Path(image_path)
        
        # Get all parent directories
        path_parts = list(path_obj.parts)
        
        # First pass: Check if any directory matches known entities in KG
        for part in reversed(path_parts):  # Check from deepest to shallowest
            part_lower = part.lower()
            if part_lower in self.knowledge_graph['entities']:
                return part_lower
        
        # Second pass: Check filename itself for entity names
        filename = path_obj.stem.lower()  # Get filename without extension
        for entity in self.knowledge_graph['entities'].keys():
            if entity in filename:
                return entity
        
        # Third pass: For generic datasets, return the immediate parent folder
        # This assumes structure like: .../category_name/image.jpg
        if len(path_parts) >= 2:
            # Get parent directory name (the folder containing the image)
            parent_folder = path_parts[-2].lower()
            
            # Skip common non-entity folder names
            skip_folders = {'images', 'data', 'train', 'test', 'val', 'validation', 
                          'raw-img', 'processed', 'uploads', 'datasets', 'static'}
            
            if parent_folder not in skip_folders:
                return parent_folder
        
        return None
    
    def get_related_entities(self, entity_name, max_related=10):
        """
        Get related entities with strength scores
        Returns list of tuples: [(entity, strength), ...]
        """
        if entity_name not in self.knowledge_graph['entities']:
            return []
        
        entity_info = self.knowledge_graph['entities'][entity_name]
        related = entity_info.get('related_entities', {})
        
        # Convert to list and sort by strength
        related_list = [(entity, strength) for entity, strength in related.items()]
        related_list.sort(key=lambda x: -x[1])
        
        return related_list[:max_related]
    
    def kg_based_recommendation(self, query_input, input_type='image', top_k=10, 
                               min_strength=0.4, mix_all_strengths=True):
        """
        Get recommendations using knowledge graph
        UPDATED: Now mixes moderate (0.4-0.7) and strong (0.8-1.0) matches
        
        Args:
            query_input: Image path or text query
            input_type: 'image' or 'text'
            top_k: Number of recommendations
            min_strength: Minimum relationship strength to include (default 0.4)
            mix_all_strengths: Mix all strength levels in results (True by default)
        
        Returns:
            Dictionary with recommendations and metadata
        """
        results = {
            'query_entity': None,
            'related_entities': [],
            'recommendations': [],
            'strength_distribution': {'strong': 0, 'moderate': 0, 'weak': 0}
        }
        
        # Get query embedding and entity
        if input_type == 'image':
            query_embedding = self.get_image_embedding(query_input)
            query_entity = self.find_entity_from_path(query_input)
        else:
            query_embedding = self.get_text_embedding(query_input)
            query_entity = None
            query_lower = query_input.lower()
            for entity in self.knowledge_graph['entities'].keys():
                if entity in query_lower:
                    query_entity = entity
                    break
        
        if not query_entity:
            # Fall back to direct similarity search
            initial_results = self.search_similar_images(query_embedding, top_k)
            results['recommendations'] = initial_results
            return results
        
        results['query_entity'] = query_entity
        
        # Get related entities with strengths
        related_with_strength = self.get_related_entities(query_entity, max_related=10)
        
        # Filter by minimum strength
        related_with_strength = [
            (entity, strength) for entity, strength in related_with_strength 
            if strength >= min_strength
        ]
        
        related_entities = [entity for entity, _ in related_with_strength]
        results['related_entities'] = related_entities
        
        if not related_with_strength:
            return results
        
        # Search for images from related entities
        all_recommendations = []
        seen_paths = set()
        
        for label, relationship_strength in related_with_strength:
            # Categorize strength
            if relationship_strength >= 0.8:
                strength_category = 'strong'
            elif relationship_strength >= 0.6:
                strength_category = 'moderate'
            else:
                strength_category = 'weak'
            
            # Create text query
            text_query = f"a photo of a {label}"
            label_embedding = self.get_text_embedding(text_query)
            
            # Search for similar images (get more to ensure variety)
            label_results = self.search_similar_images(label_embedding, top_k=20)
            
            for result in label_results:
                result_entity = self.find_entity_from_path(result['path'])
                
                # Skip if same class as query
                if result_entity == query_entity:
                    continue
                
                # Add if not seen
                if result['path'] not in seen_paths:
                    # Calculate weighted similarity
                    weighted_similarity = result['similarity'] * relationship_strength
                    
                    result['matched_label'] = label
                    result['result_entity'] = result_entity
                    result['relationship_strength'] = relationship_strength
                    result['strength_category'] = strength_category
                    result['original_similarity'] = result['similarity']
                    result['weighted_similarity'] = weighted_similarity
                    result['similarity'] = weighted_similarity
                    
                    all_recommendations.append(result)
                    seen_paths.add(result['path'])
                    results['strength_distribution'][strength_category] += 1
        
        # Sort by weighted similarity
        all_recommendations.sort(key=lambda x: -x['weighted_similarity'])
        
        # MIX MODERATE AND STRONG: Ensure diversity in results
        if mix_all_strengths and len(all_recommendations) > top_k:
            # Separate by strength
            strong = [r for r in all_recommendations if r['strength_category'] == 'strong']
            moderate = [r for r in all_recommendations if r['strength_category'] == 'moderate']
            weak = [r for r in all_recommendations if r['strength_category'] == 'weak']
            
            # Calculate proportional distribution
            strong_count = max(1, int(top_k * 0.5))  # 50% strong
            moderate_count = max(1, int(top_k * 0.35))  # 35% moderate
            weak_count = top_k - strong_count - moderate_count  # 15% weak
            
            # Adjust if categories don't have enough
            if len(strong) < strong_count:
                extra = strong_count - len(strong)
                moderate_count += extra
                strong_count = len(strong)
            
            if len(moderate) < moderate_count:
                extra = moderate_count - len(moderate)
                weak_count += extra
                moderate_count = len(moderate)
            
            # Mix results
            mixed_results = (
                strong[:strong_count] + 
                moderate[:moderate_count] + 
                weak[:weak_count]
            )
            
            # Sort mixed results by weighted similarity
            mixed_results.sort(key=lambda x: -x['weighted_similarity'])
            results['recommendations'] = mixed_results
        else:
            results['recommendations'] = all_recommendations[:top_k]
        
        return results
    
    def similarity_search(self, query_input, input_type='image', top_k=10):
        """
        Simple similarity search without knowledge graph
        
        Args:
            query_input: Image path or text query
            input_type: 'image' or 'text'
            top_k: Number of results
        
        Returns:
            Dictionary with results and query_entity
        """
        try:
            print(f"similarity_search called with input_type={input_type}, top_k={top_k}")
            
            query_entity = None
            
            if input_type == 'image':
                query_embedding = self.get_image_embedding(query_input)
                # Extract entity from uploaded image path
                query_entity = self.find_entity_from_path(query_input)
            else:
                query_embedding = self.get_text_embedding(query_input)
                # Extract entity from text query
                query_lower = query_input.lower()
                for entity in self.knowledge_graph['entities'].keys():
                    if entity in query_lower:
                        query_entity = entity
                        break
            
            results = self.search_similar_images(query_embedding, top_k)
            print(f"Found {len(results)} results, query_entity: {query_entity}")
            
            return {
                'results': results,
                'query_entity': query_entity
            }
        except Exception as e:
            print(f"Error in similarity_search: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_auto_recommendations(self, entity_name, top_k=10, min_strength=0.4, mix_all_strengths=True):
        """
        Get automatic recommendations based on entity name from search history
        Uses knowledge graph to find related entities and retrieves from vector DB
        
        Args:
            entity_name: Entity name (e.g., 'cat', 'dog', 'butterfly')
            top_k: Number of recommendations
            min_strength: Minimum relationship strength
            mix_all_strengths: Mix all strength levels
        
        Returns:
            Dictionary with recommendations and metadata
        """
        try:
            print(f"get_auto_recommendations called for entity: {entity_name}")
            
            results = {
                'query_entity': entity_name,
                'related_entities': [],
                'recommendations': [],
                'strength_distribution': {'strong': 0, 'moderate': 0, 'weak': 0}
            }
            
            # Check if entity exists in knowledge graph
            if entity_name not in self.knowledge_graph['entities']:
                print(f"Entity '{entity_name}' not found in knowledge graph")
                return results
            
            # Get related entities with strengths
            related_with_strength = self.get_related_entities(entity_name, max_related=10)
            
            # Filter by minimum strength
            related_with_strength = [
                (entity, strength) for entity, strength in related_with_strength 
                if strength >= min_strength
            ]
            
            related_entities = [entity for entity, _ in related_with_strength]
            results['related_entities'] = related_entities
            
            if not related_with_strength:
                print(f"No related entities found for '{entity_name}'")
                return results
            
            # Search for images from related entities using vector DB
            all_recommendations = []
            seen_paths = set()
            
            for label, relationship_strength in related_with_strength:
                # Categorize strength
                if relationship_strength >= 0.8:
                    strength_category = 'strong'
                elif relationship_strength >= 0.6:
                    strength_category = 'moderate'
                else:
                    strength_category = 'weak'
                
                # Create text query for the related entity
                text_query = f"a photo of a {label}"
                label_embedding = self.get_text_embedding(text_query)
                
                # Search for similar images in vector DB
                label_results = self.search_similar_images(label_embedding, top_k=20)
                
                for result in label_results:
                    result_entity = self.find_entity_from_path(result['path'])
                    
                    # Skip if same class as query entity
                    if result_entity == entity_name:
                        continue
                    
                    # Add if not seen
                    if result['path'] not in seen_paths:
                        # Calculate weighted similarity
                        weighted_similarity = result['similarity'] * relationship_strength
                        
                        result['matched_label'] = label
                        result['result_entity'] = result_entity
                        result['relationship_strength'] = relationship_strength
                        result['strength_category'] = strength_category
                        result['original_similarity'] = result['similarity']
                        result['weighted_similarity'] = weighted_similarity
                        result['similarity'] = weighted_similarity
                        
                        all_recommendations.append(result)
                        seen_paths.add(result['path'])
                        results['strength_distribution'][strength_category] += 1
            
            # Sort by weighted similarity
            all_recommendations.sort(key=lambda x: -x['weighted_similarity'])
            
            # Mix strengths if requested
            if mix_all_strengths and len(all_recommendations) > top_k:
                strong = [r for r in all_recommendations if r['strength_category'] == 'strong']
                moderate = [r for r in all_recommendations if r['strength_category'] == 'moderate']
                weak = [r for r in all_recommendations if r['strength_category'] == 'weak']
                
                strong_count = max(1, int(top_k * 0.5))
                moderate_count = max(1, int(top_k * 0.35))
                weak_count = top_k - strong_count - moderate_count
                
                if len(strong) < strong_count:
                    extra = strong_count - len(strong)
                    moderate_count += extra
                    strong_count = len(strong)
                
                if len(moderate) < moderate_count:
                    extra = moderate_count - len(moderate)
                    weak_count += extra
                    moderate_count = len(moderate)
                
                mixed_results = (
                    strong[:strong_count] + 
                    moderate[:moderate_count] + 
                    weak[:weak_count]
                )
                
                mixed_results.sort(key=lambda x: -x['weighted_similarity'])
                results['recommendations'] = mixed_results
            else:
                results['recommendations'] = all_recommendations[:top_k]
            
            print(f"Generated {len(results['recommendations'])} recommendations for '{entity_name}'")
            return results
            
        except Exception as e:
            print(f"Error in get_auto_recommendations: {e}")
            import traceback
            traceback.print_exc()
            raise
