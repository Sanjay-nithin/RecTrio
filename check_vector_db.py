import pickle
from pathlib import Path

metadata_path = Path(r"e:\Projects\AI Based\RecTrio\V1\vector_db\metadata.pkl")

if metadata_path.exists():
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Total images in vector DB: {metadata['total_images']}")
    print(f"Embedding dimension: {metadata['embedding_dim']}")
    
    # Check a few sample paths to see if they're from pruned dataset
    print(f"\nSample image paths (first 5):")
    for i, path in enumerate(metadata['image_paths'][:5]):
        print(f"  {i+1}. {path}")
    
    # Count images per category
    from collections import Counter
    categories = []
    for path in metadata['image_paths']:
        path_parts = Path(path).parts
        if 'fashion' in path_parts:
            idx = path_parts.index('fashion')
            if idx + 1 < len(path_parts):
                categories.append(path_parts[idx + 1])
    
    category_counts = Counter(categories)
    print(f"\nCategories in vector DB: {len(category_counts)}")
    print(f"Category distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} images")
else:
    print("Metadata file not found!")
