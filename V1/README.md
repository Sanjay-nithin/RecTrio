# Custom Multimodal CNN for Image-Text Retrieval

## Overview
This is a **custom-built dual-encoder model** trained from scratch for multimodal image-text retrieval. The model learns to embed both images and text into a shared embedding space, enabling:
- 🖼️ **Image-to-Image search**: Find similar images
- 📝 **Text-to-Image search**: Find images matching a description
- 🔄 **Cross-modal retrieval**: Images and text in the same space

## Architecture

### Image Encoder (Custom CNN)
- **Input**: 224×224 RGB images
- **Architecture**: 4-block CNN (64→128→256→512 channels)
- **Features**:
  - Batch normalization for stable training
  - ReLU activations
  - Max pooling for downsampling
  - Global average pooling
- **Output**: 256-dimensional L2-normalized embedding

### Text Encoder (BiLSTM)
- **Input**: Tokenized text descriptions (max 20 words)
- **Architecture**: Embedding layer → BiLSTM → Projection
- **Features**:
  - 128-dimensional word embeddings
  - Bidirectional LSTM (256 hidden units per direction)
  - Dropout for regularization
- **Output**: 256-dimensional L2-normalized embedding

### Training Objective
- **Loss**: Contrastive loss (CLIP-style InfoNCE)
- **Goal**: Maximize similarity between matching image-text pairs, minimize for non-matching pairs
- **Temperature**: 0.07 (controls sharpness of distribution)

## Dataset
- **Classes**: 10 animals (butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel)
- **Text Descriptions**: 10 diverse descriptions per class (100 total)
- **Augmentation**: Random flips, rotations, color jitter

## Intel CPU Optimization

### OpenVINO Integration
The trained PyTorch model is converted to **OpenVINO IR format** for Intel CPU optimization:
- ✅ Optimized graph execution
- ✅ INT8 quantization support
- ✅ Intel DNNL (Deep Neural Network Library) kernels
- ✅ Fast inference on CPU

### Files Generated
```
V1/models/custom_cnn/
├── best_multimodal_model.pth      # PyTorch checkpoint
├── vocabulary.pkl                  # Text vocabulary
├── image_encoder.xml/.bin         # OpenVINO image encoder
├── text_encoder.xml/.bin          # OpenVINO text encoder
└── training_history.png           # Training curves
```

## How to Use

### 1. Training
```bash
# Open the training notebook
jupyter notebook V1/training/custom_cnn/train_multimodal.ipynb

# Follow the cells sequentially:
# 1. Install dependencies
# 2. Load data and build vocabulary
# 3. Train dual-encoder model (30 epochs)
# 4. Convert to OpenVINO
```

**Expected Training Time**: 
- CPU: ~2-3 hours (depending on CPU)
- GPU: ~30-45 minutes (if available)

### 2. Inference

```bash
# Open the inference notebook
jupyter notebook V1/inference/custom_cnn/multimodal_inference.ipynb

# Features:
# - Automatically builds/loads embedding database
# - Image-based search
# - Text-based search
# - Interactive search interface
```

### 3. Example Queries

#### Image Query:
```python
query_image_path = "path/to/cat.jpg"
query_embedding = get_image_embedding(query_image_path)
results = search_similar_images(query_embedding, top_k=10)
display_results(results)
```

#### Text Query:
```python
query_text = "a fluffy cat with whiskers"
query_embedding = get_text_embedding(query_text)
results = search_similar_images(query_embedding, top_k=10)
display_results(results)
```

## Text Descriptions

The model is trained with **10 descriptions per animal class**:

**Example (Cat):**
- "a domestic cat sitting peacefully"
- "a cute cat with whiskers and fur"
- "a feline pet resting indoors"
- "a cat with green eyes looking at camera"
- ...and 6 more variations

See `datasets/animals/text_descriptions.py` for full list.

## Performance Metrics

### Training Metrics
- **Loss**: Contrastive loss (lower is better)
- **Accuracy**: % of correct image-text matches in batch

### Inference Performance (Intel CPU)
- **Image encoding**: ~10-20ms per image
- **Text encoding**: ~5-10ms per query
- **FAISS search**: <1ms for 10K images

## Vector Database

### FAISS Index
- **Type**: IndexFlatIP (Inner Product / Cosine Similarity)
- **Metric**: Cosine similarity (due to L2 normalization)
- **Storage**: Efficient binary format

### Files:
```
V1/models/custom_cnn/vector_db/
├── embeddings.npy        # All image embeddings
├── metadata.pkl          # Image paths and metadata
└── faiss_index.bin       # FAISS search index
```

## Advantages Over Pre-trained Models

### Custom CNN Benefits:
1. ✅ **Smaller model** (~5-10M parameters vs CLIP's 100M+)
2. ✅ **Faster inference** on CPU
3. ✅ **Domain-specific** training on your data
4. ✅ **Full control** over architecture
5. ✅ **No licensing issues**

### Trade-offs:
- ❌ Requires training time
- ❌ Limited to trained classes (can be fine-tuned)
- ❌ Less general than CLIP

## Extending the Model

### Add More Classes:
1. Add images to `datasets/animals/raw-img/<new_class>/`
2. Add descriptions to `text_descriptions.py`
3. Retrain the model

### Fine-tune for New Domain:
1. Load checkpoint: `torch.load('best_multimodal_model.pth')`
2. Add new data
3. Train with lower learning rate

### Improve Performance:
- Increase model capacity (more CNN layers)
- Use Transformer for text encoder
- Add more data augmentation
- Train longer with learning rate scheduling

## Dependencies

```txt
torch>=2.0.0
torchvision>=0.15.0
openvino>=2024.0.0
faiss-cpu>=1.7.0
numpy
pillow
matplotlib
seaborn
tqdm
```

## Intel Technologies Used

1. **OpenVINO**: Model optimization and inference
2. **Intel DNNL**: Optimized deep learning kernels
3. **FAISS-CPU**: Fast similarity search (Intel MKL optimized)
4. **NumPy**: Intel MKL-accelerated linear algebra

## Project Structure

```
RecTrio/
├── datasets/
│   └── animals/
│       ├── raw-img/          # Images organized by class
│       └── text_descriptions.py  # Text descriptions
├── V1/
│   ├── training/
│   │   └── custom_cnn/
│   │       └── train_multimodal.ipynb
│   ├── inference/
│   │   └── custom_cnn/
│   │       └── multimodal_inference.ipynb
│   └── models/
│       └── custom_cnn/
│           ├── *.pth         # PyTorch models
│           ├── *.xml/.bin    # OpenVINO models
│           └── vector_db/    # Embeddings database
```

## Troubleshooting

### Issue: Slow training
**Solution**: 
- Reduce batch size
- Use GPU if available
- Reduce image resolution (but keep aspect ratio)

### Issue: Poor text matching
**Solution**:
- Add more diverse text descriptions
- Increase text encoder capacity
- Train longer with lower temperature

### Issue: OpenVINO conversion fails
**Solution**:
- Ensure OpenVINO version compatibility
- Check PyTorch model is in eval mode
- Verify input shapes match

## License
This is a custom implementation for educational purposes.

## Author
Built for RecTrio project - Intel CPU optimized multimodal retrieval system
