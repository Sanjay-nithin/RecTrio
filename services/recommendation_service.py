"""
Recommendation Service for RecTrio
Handles both similarity search and knowledge graph-based     def search_similar_images(self, query_embedding, top_k=10, min_threshold=0.25, 
                             adaptive=True, adaptive_ratio=0.70):commendations
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
        kg_path = self.vector_db_dir / "fashion_knowledge_graph.json"
        with open(kg_path, 'r') as f:
            self.knowledge_graph = json.load(f)
        
        # Cache for domain filtering text embeddings (for performance)
        self._domain_embeddings_cache = None
        
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
    
    def is_fashion_related(self, query_embedding, margin=1.05):
        """
        Check if query embedding is more related to fashion than non-fashion items
        Uses CLIP text embeddings to compare with fashion vs non-fashion concepts
        Uses caching for performance (only computes text embeddings once)
        
        Args:
            query_embedding: Image embedding to check (normalized)
            margin: Fashion score must be this much higher than non-fashion (default 1.05 = 5%)
        
        Returns:
            tuple: (is_fashion: bool, fashion_score: float, non_fashion_score: float)
        """
        try:
            # Initialize cache on first call
            if self._domain_embeddings_cache is None:
                print("Building domain embeddings cache (one-time setup)...")
                
                # Fashion-related terms
                fashion_terms = [
                    "clothing", "shoes", "accessories", "fashion item", "apparel",
                    "footwear", "garment", "dress", "shirt", "pants", "jacket",
                    "bag", "watch", "jewelry", "sunglasses", "hat", "belt"
                ]
                
                # Non-fashion terms (tools, machines, nature, etc.)
                non_fashion_terms = [
                    "tool", "machine", "vehicle", "building", "equipment", "appliance",
                    "electronics", "furniture", "nature", "animal", "plant", "food",
                    "bicycle pump", "windmill", "ladder", "wrench", "screwdriver"
                ]
                
                # Pre-compute all text embeddings
                fashion_embeddings = []
                for term in fashion_terms:
                    text_emb = self.get_text_embedding(f"a photo of {term}")
                    fashion_embeddings.append(text_emb.flatten())
                
                non_fashion_embeddings = []
                for term in non_fashion_terms:
                    text_emb = self.get_text_embedding(f"a photo of {term}")
                    non_fashion_embeddings.append(text_emb.flatten())
                
                self._domain_embeddings_cache = {
                    'fashion': np.array(fashion_embeddings),
                    'non_fashion': np.array(non_fashion_embeddings)
                }
                print(f"✓ Cached {len(fashion_embeddings)} fashion + {len(non_fashion_embeddings)} non-fashion embeddings")
            
            # Use cached embeddings for fast comparison
            query_flat = query_embedding.flatten()
            
            # Compute similarities with all fashion terms at once (vectorized)
            fashion_similarities = np.dot(self._domain_embeddings_cache['fashion'], query_flat)
            avg_fashion = float(np.mean(fashion_similarities))
            
            # Compute similarities with all non-fashion terms at once (vectorized)
            non_fashion_similarities = np.dot(self._domain_embeddings_cache['non_fashion'], query_flat)
            avg_non_fashion = float(np.mean(non_fashion_similarities))
            
            is_fashion = avg_fashion > (avg_non_fashion * margin)
            
            print(f"Fashion domain check: fashion={avg_fashion:.3f}, non-fashion={avg_non_fashion:.3f}, is_fashion={is_fashion}")
            
            return is_fashion, avg_fashion, avg_non_fashion
        except Exception as e:
            print(f"Warning: Fashion domain check failed: {e}")
            # If check fails, assume it's fashion to avoid false rejections
            return True, 0.0, 0.0

    
    def search_similar_images(self, query_embedding, top_k=10, min_threshold=0.25, 
                             adaptive=True, adaptive_ratio=0.70, check_domain=True):
        # Domain check: Filter out non-fashion items
        if check_domain:
            is_fashion, fashion_score, non_fashion_score = self.is_fashion_related(query_embedding)
            if not is_fashion:
                print(f"⚠ Query rejected: Not fashion-related (fashion={fashion_score:.3f}, non-fashion={non_fashion_score:.3f})")
                return []  # Return empty results for non-fashion items
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get top similarity for adaptive thresholding
        top_similarity = float(distances[0][0]) if len(distances[0]) > 0 else 0.0
        
        # Calculate effective threshold
        effective_threshold = min_threshold
        if adaptive and top_similarity > min_threshold:
            # Adaptive threshold: results must be within adaptive_ratio of top result
            adaptive_threshold_val = top_similarity * adaptive_ratio
            effective_threshold = max(min_threshold, adaptive_threshold_val)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            similarity = float(distance)
            
            # Apply threshold filter
            if similarity < effective_threshold:
                continue
                
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
                    'similarity': similarity,
                    'label': entity if entity else 'unknown'
                })
        
        # Log threshold info
        if len(results) == 0 and top_similarity > 0:
            print(f"⚠ No results above threshold (top similarity: {top_similarity:.3f}, threshold: {effective_threshold:.3f})")
            print(f"   Query appears to be out-of-domain (not fashion-related)")
        elif len(results) < top_k and top_similarity > 0:
            print(f"ℹ Filtered {top_k - len(results)} low-similarity results (threshold: {effective_threshold:.3f})")
        
        return results
    
    def find_entity_from_path(self, image_path):
        """
        Extract entity/label name from image path
        Scalable approach: checks folder names in path against known entities
        Falls back to parent folder name if not in knowledge graph
        
        Args:
            image_path: Full or relative path to image
            
        Returns:
            Entity name (matching knowledge graph case) or None if not found
        """
        if not image_path:
            return None
            
        # Convert to Path object for easier manipulation
        path_obj = Path(image_path)
        
        # Get all parent directories
        path_parts = list(path_obj.parts)
        
        # First pass: Check if any directory matches known entities in KG (case-insensitive)
        for part in reversed(path_parts):  # Check from deepest to shallowest
            part_lower = part.lower()
            # Match against knowledge graph entities (case-insensitive)
            for kg_entity in self.knowledge_graph['entities'].keys():
                if kg_entity.lower() == part_lower or kg_entity.lower().replace('_', ' ') == part_lower:
                    return kg_entity  # Return with proper capitalization
        
        # Second pass: Check filename itself for entity names
        filename = path_obj.stem.lower()  # Get filename without extension
        for kg_entity in self.knowledge_graph['entities'].keys():
            if kg_entity.lower() in filename:
                return kg_entity  # Return with proper capitalization
        
        # Third pass: For generic datasets, return the immediate parent folder
        # This assumes structure like: .../category_name/image.jpg
        if len(path_parts) >= 2:
            # Get parent directory name (the folder containing the image)
            parent_folder = path_parts[-2]
            
            # Skip common non-entity folder names
            skip_folders = {'images', 'data', 'train', 'test', 'val', 'validation', 
                          'raw-img', 'processed', 'uploads', 'datasets', 'static'}
            
            if parent_folder.lower() not in skip_folders:
                # Try to match with knowledge graph entities (case-insensitive)
                for kg_entity in self.knowledge_graph['entities'].keys():
                    if kg_entity.lower() == parent_folder.lower():
                        return kg_entity  # Return with proper capitalization
                # If not in KG, return as-is (for compatibility)
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
            query_lower = query_input.lower().strip()
            
            # Level 1: Try EXACT match (case-insensitive, full entity name)
            for entity in self.knowledge_graph['entities'].keys():
                entity_lower = entity.lower()
                entity_with_space = entity.replace('_', ' ').lower()
                
                # Exact match with or without underscores
                if query_lower == entity_lower or query_lower == entity_with_space:
                    query_entity = entity
                    print(f"Level 1 - Exact match: '{query_input}' -> '{entity}'")
                    break
            
            # Level 2: Try matching with "a photo of" prefix removed
            if not query_entity:
                query_cleaned = query_lower.replace('a photo of', '').replace('an image of', '').strip()
                for entity in self.knowledge_graph['entities'].keys():
                    entity_lower = entity.lower()
                    entity_with_space = entity.replace('_', ' ').lower()
                    
                    if query_cleaned == entity_lower or query_cleaned == entity_with_space:
                        query_entity = entity
                        print(f"Level 2 - Prefix removed: '{query_input}' -> '{entity}'")
                        break
            
            # Level 3: Try singular/plural variations (exact word match)
            if not query_entity:
                query_singular = query_lower.rstrip('s')
                for entity in self.knowledge_graph['entities'].keys():
                    entity_lower = entity.lower()
                    entity_singular = entity_lower.rstrip('s')
                    entity_with_space = entity.replace('_', ' ').lower()
                    entity_with_space_singular = entity_with_space.rstrip('s')
                    
                    # Check if singular forms match
                    if (query_lower == entity_lower or 
                        query_singular == entity_singular or
                        query_lower == entity_with_space or
                        query_singular == entity_with_space_singular):
                        query_entity = entity
                        print(f"Level 3 - Singular/plural: '{query_input}' -> '{entity}'")
                        break
            
            # Level 4: Try word boundary matching (prevent "shirts" matching "tshirts")
            if not query_entity:
                import re
                query_words = set(re.findall(r'\b\w+\b', query_lower))
                
                best_match = None
                best_score = 0
                
                for entity in self.knowledge_graph['entities'].keys():
                    entity_lower = entity.lower()
                    entity_with_space = entity.replace('_', ' ').lower()
                    entity_words = set(re.findall(r'\b\w+\b', entity_with_space))
                    
                    # Calculate word overlap
                    common_words = query_words & entity_words
                    if common_words:
                        # Prefer exact word matches over substrings
                        score = len(common_words) / max(len(query_words), len(entity_words))
                        if score > best_score:
                            best_score = score
                            best_match = entity
                
                if best_match and best_score >= 0.5:  # At least 50% word overlap
                    query_entity = best_match
                    print(f"Level 4 - Word boundary: '{query_input}' -> '{best_match}' (score: {best_score:.2f})")
            
            # Level 5: Fallback to similarity search result
            if not query_entity:
                print(f"Level 5 - No entity match for '{query_input}', will use fallback")
        
        if not query_entity:
            # Fall back to direct similarity search
            initial_results = self.search_similar_images(query_embedding, top_k)
            # Try to detect entity from top result
            if initial_results:
                query_entity = self.find_entity_from_path(initial_results[0]['path'])
                if query_entity and query_entity in self.knowledge_graph['entities']:
                    results['query_entity'] = query_entity
                    # Continue with KG recommendations
                else:
                    results['recommendations'] = initial_results
                    return results
            else:
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
                
                # Skip if not from the target label category (ensure correct category)
                if result_entity != label:
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
        # NEW: 5 very strong (0.9+), 3 strong (0.75-0.89), 2 moderate (0.6-0.74)
        if mix_all_strengths and len(all_recommendations) > top_k:
            # Separate by strength with more granular categories
            very_strong = [r for r in all_recommendations if r['relationship_strength'] >= 0.9]
            strong = [r for r in all_recommendations if 0.75 <= r['relationship_strength'] < 0.9]
            moderate = [r for r in all_recommendations if 0.6 <= r['relationship_strength'] < 0.75]
            weak = [r for r in all_recommendations if r['relationship_strength'] < 0.6]
            
            # Calculate proportional distribution: 5 very strong, 3 strong, 2 moderate
            very_strong_count = min(5, len(very_strong))
            strong_count = min(3, len(strong))
            moderate_count = min(2, len(moderate))
            
            # Fill remaining slots if categories don't have enough
            total_assigned = very_strong_count + strong_count + moderate_count
            remaining = top_k - total_assigned
            
            # Distribute remaining slots
            if remaining > 0:
                if len(very_strong) > very_strong_count:
                    extra = min(remaining, len(very_strong) - very_strong_count)
                    very_strong_count += extra
                    remaining -= extra
                
                if remaining > 0 and len(strong) > strong_count:
                    extra = min(remaining, len(strong) - strong_count)
                    strong_count += extra
                    remaining -= extra
                
                if remaining > 0 and len(moderate) > moderate_count:
                    extra = min(remaining, len(moderate) - moderate_count)
                    moderate_count += extra
                    remaining -= extra
                
                if remaining > 0 and len(weak) > 0:
                    weak_count = min(remaining, len(weak))
                else:
                    weak_count = 0
            else:
                weak_count = 0
            
            # Mix results
            mixed_results = (
                very_strong[:very_strong_count] + 
                strong[:strong_count] + 
                moderate[:moderate_count] +
                (weak[:weak_count] if weak_count > 0 else [])
            )
            
            # Sort mixed results by weighted similarity
            mixed_results.sort(key=lambda x: -x['weighted_similarity'])
            results['recommendations'] = mixed_results
        else:
            results['recommendations'] = all_recommendations[:top_k]
        
        return results
    
    def similarity_search(self, query_input, input_type='image', top_k=10, 
                         min_threshold=0.25, adaptive_ratio=0.70):
        """
        Simple similarity search without knowledge graph, with dynamic thresholding
        
        Args:
            query_input: Image path or text query
            input_type: 'image' or 'text'
            top_k: Number of results
            min_threshold: Minimum absolute similarity threshold (default 0.20)
            adaptive_ratio: Adaptive threshold ratio (default 0.65)
        
        Returns:
            Dictionary with results and query_entity
        """
        try:
            print(f"similarity_search called with input_type={input_type}, top_k={top_k}")
            print(f"  Threshold config: min={min_threshold}, adaptive_ratio={adaptive_ratio}")
            
            query_entity = None
            
            if input_type == 'image':
                query_embedding = self.get_image_embedding(query_input)
                # Extract entity from uploaded image path
                query_entity = self.find_entity_from_path(query_input)
            else:
                query_embedding = self.get_text_embedding(query_input)
                # Extract entity from text query using improved matching
                query_entity = None
                query_lower = query_input.lower().strip()
                
                # Try exact match first
                for entity in self.knowledge_graph['entities'].keys():
                    entity_lower = entity.lower()
                    entity_with_space = entity.replace('_', ' ').lower()
                    
                    if query_lower == entity_lower or query_lower == entity_with_space:
                        query_entity = entity
                        break
                
                # If not found, try word boundary matching to prevent "shirts" matching "tshirts"
                if not query_entity:
                    import re
                    query_words = set(re.findall(r'\b\w+\b', query_lower))
                    
                    for entity in self.knowledge_graph['entities'].keys():
                        entity_with_space = entity.replace('_', ' ').lower()
                        entity_words = set(re.findall(r'\b\w+\b', entity_with_space))
                        
                        # Check for word overlap
                        if query_words & entity_words:
                            query_entity = entity
                            break
            
            # Call search with threshold parameters and domain checking
            results = self.search_similar_images(
                query_embedding, top_k=top_k,
                min_threshold=min_threshold, adaptive=True, adaptive_ratio=adaptive_ratio,
                check_domain=True  # Enable fashion domain filtering
            )
            print(f"Found {len(results)} results (after threshold filtering), query_entity: {query_entity}")
            
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
            
            # Normalize entity name - try case-insensitive matching
            normalized_entity = None
            if entity_name in self.knowledge_graph['entities']:
                normalized_entity = entity_name
            else:
                # Try case-insensitive match
                entity_lower = entity_name.lower()
                for kg_entity in self.knowledge_graph['entities'].keys():
                    if kg_entity.lower() == entity_lower:
                        normalized_entity = kg_entity
                        print(f"Normalized '{entity_name}' to '{normalized_entity}'")
                        break
            
            results = {
                'query_entity': normalized_entity or entity_name,
                'related_entities': [],
                'recommendations': [],
                'strength_distribution': {'strong': 0, 'moderate': 0, 'weak': 0}
            }
            
            # Check if entity exists in knowledge graph
            if normalized_entity is None:
                print(f"Entity '{entity_name}' not found in knowledge graph")
                available_entities = list(self.knowledge_graph['entities'].keys())
                print(f"Available entities: {available_entities[:10]}...")
                return results
            
            # Use normalized entity name
            entity_name = normalized_entity
            
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
                    
                    # Skip if not from target label category (ensure correct category)
                    if result_entity != label:
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
            
            # Mix strengths: 5 very strong (0.9+), 3 strong (0.75-0.89), 2 moderate (0.6-0.74)
            if mix_all_strengths and len(all_recommendations) > top_k:
                # Separate by more granular strength levels
                very_strong = [r for r in all_recommendations if r['relationship_strength'] >= 0.9]
                strong = [r for r in all_recommendations if 0.75 <= r['relationship_strength'] < 0.9]
                moderate = [r for r in all_recommendations if 0.6 <= r['relationship_strength'] < 0.75]
                weak = [r for r in all_recommendations if r['relationship_strength'] < 0.6]
                
                # Target distribution: 5 very strong, 3 strong, 2 moderate
                very_strong_count = min(5, len(very_strong))
                strong_count = min(3, len(strong))
                moderate_count = min(2, len(moderate))
                
                # Fill remaining slots
                total_assigned = very_strong_count + strong_count + moderate_count
                remaining = top_k - total_assigned
                
                if remaining > 0:
                    if len(very_strong) > very_strong_count:
                        extra = min(remaining, len(very_strong) - very_strong_count)
                        very_strong_count += extra
                        remaining -= extra
                    
                    if remaining > 0 and len(strong) > strong_count:
                        extra = min(remaining, len(strong) - strong_count)
                        strong_count += extra
                        remaining -= extra
                    
                    if remaining > 0 and len(moderate) > moderate_count:
                        extra = min(remaining, len(moderate) - moderate_count)
                        moderate_count += extra
                        remaining -= extra
                    
                    if remaining > 0 and len(weak) > 0:
                        weak_count = min(remaining, len(weak))
                    else:
                        weak_count = 0
                else:
                    weak_count = 0
                
                mixed_results = (
                    very_strong[:very_strong_count] + 
                    strong[:strong_count] + 
                    moderate[:moderate_count] +
                    (weak[:weak_count] if weak_count > 0 else [])
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
