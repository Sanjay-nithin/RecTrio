# CLIP-based Image Similarity and Recommendation System

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended, 8GB+ VRAM) or CPU
- 16GB+ RAM recommended for FAISS indexing

### Required Python Packages
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install faiss-gpu  # or faiss-cpu if no GPU
pip install pillow numpy matplotlib
```

### 2. Prepare Your Dataset
Organize your images in folders by class name:
```
Animals/
└── archive (1)/
    └── raw-img/
        ├── cat/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── dog/
        │   ├── image1.jpg
        │   └── ...
        ├── butterfly/
        └── ... (10 classes total)
```

### 3. Run the Complete Pipeline
```bash
# Option 1: Run everything at once
python main.py --mode all

# Option 2: Run steps individually
python main.py --mode train      # Train the triplet network
python main.py --mode index      # Build FAISS index
python main.py --mode search     # Search for similar images
```

### 4. Search for Similar Images
```bash
# Interactive search
python search.py --query_image path/to/your/image.jpg --top_k 10

# Or use the main script
python main.py --mode search --query_image path/to/image.jpg
```

## 📁 Project Structure

```
Animals/
├── config.py              # All configuration and hyperparameters
├── dataset.py             # Triplet dataset with folder-based labels
├── model.py               # CLIP + projection head architecture
├── train.py               # Training pipeline with staged learning
├── build_index.py         # FAISS index construction
├── search.py              # Similarity search and retrieval
├── utils.py               # Helper functions
├── visualize_results.py   # Result visualization
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
└── outputs/
    ├── checkpoints/       # Trained model weights
    │   ├── best_triplet_model.pth
    │   └── last_triplet_model.pth
    └── faiss_index/       # FAISS index files
        ├── image_embeddings.index
        └── image_paths.pkl
```

## 🔧 Configuration

Edit `config.py` to customize:

- **Model Settings**: CLIP variant, embedding dimensions
- **Training Settings**: Epochs, batch size, learning rate
- **Triplet Loss**: Margin, mining strategy
- **FAISS Settings**: Index type (HNSW, IVF, Flat), search parameters
- **Paths**: Data directory, output locations

## 📚 System Architecture

### Phase 1: Training (Offline)
1. **Triplet Dataset Creation**
   - Scans folders to discover classes automatically
   - Generates (anchor, positive, negative) triplets
   - Anchor & Positive: same class folder
   - Negative: different class folder

2. **Model Fine-tuning**
   - Stage 1: Freeze CLIP, train projection head only
   - Stage 2: Unfreeze last CLIP layers, fine-tune end-to-end
   - Uses Triplet Margin Loss with L2-normalized embeddings

3. **Embedding Generation**
   - Process all images through trained model
   - Generate 128-D embeddings (512-D CLIP → projection head)
   - L2 normalize for cosine similarity

4. **FAISS Index Building**
   - Store embeddings in HNSW index for sub-linear search
   - Maintain mapping: FAISS index → image file paths
   - Enables O(log n) retrieval vs O(n) brute-force

### Phase 2: Inference (Online)
1. Query image → CLIP embedding
2. FAISS search → Top-K nearest neighbors (fast!)
3. Return image paths + similarity scores
4. Display recommendations

## 🎯 Key Design Decisions

### Why Folder-based Labels?
- **Simplicity**: Folder name = class label, no manual annotation
- **Triplet Learning**: We only need same/different class for triplet selection
- **Scalability**: Easy to add new classes by adding folders

### Why FAISS?
- **Speed**: HNSW provides O(log n) search vs O(n) brute-force
- **Scalability**: Can handle millions of images efficiently
- **Memory**: Optimized storage and retrieval

### Why Triplet Loss?
- **Metric Learning**: Learns discriminative embeddings
- **Pull**: Same-class images closer in embedding space
- **Push**: Different-class images further apart

### Why Precompute Embeddings?
- **Offline Cost**: Generate once, search many times
- **Online Speed**: Search only requires query encoding + FAISS lookup
- **Scalability**: Adding new images just requires incremental indexing

## 💡 Usage Examples

### Train from Scratch
```python
from train import train_model

# Train with custom settings
train_model(
    data_dir="path/to/raw-img",
    epochs=15,
    batch_size=64,
    learning_rate=1e-4
)
```

### Build FAISS Index
```python
from build_index import build_faiss_index

# Build index for all images
build_faiss_index(
    data_dir="path/to/raw-img",
    model_path="outputs/checkpoints/best_triplet_model.pth",
    index_type="HNSW"
)
```

### Search Similar Images
```python
from search import search_similar_images

# Find top-10 similar images
results = search_similar_images(
    query_image_path="test_image.jpg",
    top_k=10
)

for img_path, score in results:
    print(f"{img_path}: {score:.4f}")
```

## 🔬 Performance Tips

1. **GPU Acceleration**: Use CUDA for training and embedding generation
2. **Batch Processing**: Use DataLoader with num_workers > 0
3. **FAISS GPU**: Use `faiss-gpu` for faster search on large datasets
4. **HNSW Tuning**: Increase `ef_search` for better accuracy, decrease for speed
5. **Mixed Precision**: Enable AMP (Automatic Mixed Precision) for faster training

## 📊 Expected Results

- **Training**: 15 epochs should converge triplet loss
- **Validation**: Triplet accuracy > 80% indicates good separation
- **Search**: Top-K results should be visually similar to query
- **Speed**: HNSW search < 10ms for 10K images on GPU

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `config.py`
- Use smaller CLIP model: "ViT-B/32" instead of "ViT-B/16"
- Enable gradient checkpointing

### Slow Training
- Increase `NUM_WORKERS` in config
- Use mixed precision training
- Ensure GPU is being used (check with `nvidia-smi`)

### Poor Search Results
- Train for more epochs
- Increase triplet margin
- Use more triplets per image
- Try different CLIP model variant

### FAISS Installation Issues
```bash
# CPU version (if no GPU)
pip install faiss-cpu

# GPU version
conda install -c pytorch faiss-gpu
```

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{clip-faiss-similarity,
  title={CLIP-based Image Similarity System with FAISS},
  author={Your Name},
  year={2024}
}
```

## 📄 License

MIT License - Feel free to use for commercial and non-commercial projects.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Hard negative mining
- Online triplet mining
- Multi-modal search (text + image)
- Distributed training
- Production deployment (FastAPI/Flask)

## 📧 Support

For issues, questions, or suggestions, please open a GitHub issue.
