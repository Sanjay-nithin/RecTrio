# RecTrio - Image Similarity & Recommendation System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive image similarity and recommendation system using **Triplet Networks**. This repository contains two distinct implementations demonstrating different approaches to metric learning:

1. **Fashion MNIST**: Training a triplet network **from scratch**
2. **Animals Dataset**: Fine-tuning a **pretrained CLIP model** using triplet networks

---

## 📚 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Technical Approach](#technical-approach)
- [Full-Stack Application Roadmap](#full-stack-application-roadmap)
- [Free Infrastructure Options](#free-infrastructure-options)
- [Quick Start](#quick-start)
- [Training on Google Colab](#training-on-google-colab)
- [Deployment Guide](#deployment-guide)

---

## 🎯 Overview

### What is RecTrio?

RecTrio is an AI-powered image similarity search and recommendation system that learns to understand visual similarity using triplet networks. It can find visually similar images, recommend products, and power reverse image search applications.

### Two Implementations Explained

#### 1️⃣ **Fashion MNIST - Training from Scratch**

**Dataset**: Fashion MNIST (70,000 grayscale images of 10 fashion categories)

**Approach**:
- ✅ Custom CNN architecture built from scratch
- ✅ Trained on triplets: (anchor, positive, negative)
- ✅ Learns embeddings in a 128-dimensional space
- ✅ Uses triplet margin loss for metric learning
- ✅ No pretrained models - learns everything from the dataset

**Use Case**: Demonstrates fundamental triplet learning concepts on a simple dataset

**Files**: `FashionMNIST/main.ipynb`, `FashionMNIST/fashion_mnist_visualization.ipynb`

**Key Features**:
- Simple CNN: Conv2D → BatchNorm → ReLU → MaxPool → FC layers
- Embedding dimension: 128
- Training time: ~15-20 minutes on GPU
- Perfect for learning and experimentation

---

#### 2️⃣ **Animals Dataset - Fine-tuning CLIP**

**Dataset**: 10 animal classes with RGB images (variable sizes)

**Approach**:
- ✅ Leverages **CLIP** (Contrastive Language-Image Pre-training) from OpenAI
- ✅ CLIP is pretrained on 400M image-text pairs from the internet
- ✅ Two-phase fine-tuning strategy:
  - **Phase 1**: Train projection head only (CLIP frozen)
  - **Phase 2**: Fine-tune last transformer layers + projection head
- ✅ Adapts powerful visual representations to our specific task
- ✅ Uses FAISS for efficient similarity search at scale

**Use Case**: Production-ready system for real-world image search applications

**Files**: `Animals/*.py`, `Animals/train_colab.ipynb`

**Key Features**:
- CLIP ViT-B/32 encoder (87M parameters)
- Projection head reduces 512D → 128D
- FAISS HNSW index for fast nearest neighbor search
- Handles high-resolution images
- Scalable to millions of images

---

## 📂 Project Structure

```
RecTrio/
│
├── FashionMNIST/                      # Project 1: From-scratch training
│   ├── main.ipynb                     # Complete training notebook
│   ├── fashion_mnist_visualization.ipynb
│   ├── best_triplet_model.pth         # Trained model weights
│   └── archive (1)/                   # Dataset (auto-downloaded)
│       └── fashion-mnist_*.csv
│
├── Animals/                           # Project 2: CLIP fine-tuning
│   ├── config.py                      # Configuration & hyperparameters
│   ├── dataset.py                     # Triplet dataset loader
│   ├── model.py                       # CLIP + projection head
│   ├── train.py                       # Two-phase training pipeline
│   ├── build_index.py                 # FAISS index builder
│   ├── search.py                      # Similarity search engine
│   ├── visualize_results.py           # Result visualization
│   ├── main.py                        # CLI interface
│   ├── train_colab.ipynb              # Google Colab training notebook
│   ├── utils.py                       # Helper functions
│   ├── requirements.txt               # Python dependencies
│   └── outputs/
│       ├── checkpoints/               # Model checkpoints
│       │   ├── best_triplet_model.pth
│       │   └── last_triplet_model.pth
│       └── faiss_index/               # Vector database
│           ├── image_embeddings.index
│           └── image_paths.pkl
│
└── requirements.txt                   # Global dependencies
```

---

## 🔬 Technical Approach

### Triplet Learning Fundamentals

Both projects use **triplet loss** for metric learning:

```
Loss = max(0, ||anchor - positive||² - ||anchor - negative||² + margin)
```

**Goal**: 
- Minimize distance between similar images (anchor & positive)
- Maximize distance between dissimilar images (anchor & negative)
- Maintain a safety margin between the two distances

### Architecture Comparison

| Aspect | Fashion MNIST | Animals (CLIP) |
|--------|---------------|----------------|
| **Input** | 28×28 grayscale | 224×224 RGB |
| **Base Model** | Custom CNN | Pretrained CLIP ViT-B/32 |
| **Encoder Params** | ~500K | 87M (pretrained) |
| **Training Strategy** | End-to-end from scratch | Two-phase fine-tuning |
| **Embedding Dim** | 128 | 512 → 128 (projection) |
| **Training Time** | 15-20 min | 2-3 hours |
| **Scalability** | Limited (simple dataset) | Production-ready |
| **Search Method** | Brute force | FAISS HNSW index |

### Why CLIP for Animals Dataset?

1. **Transfer Learning**: CLIP has seen millions of diverse images during pretraining
2. **Rich Features**: Understands complex visual concepts (textures, shapes, contexts)
3. **Faster Convergence**: Pretrained weights mean less training time
4. **Better Generalization**: Works well even with limited training data
5. **Multimodal**: Can be extended to text-image search

---

## 🏗️ Full-Stack Application Roadmap

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  Web App (React/Next.js) │ Mobile App (React Native)        │
│  - Image Upload UI        │ - Camera Integration             │
│  - Search Results Grid    │ - Similar Product Feed           │
│  - Filters & Sorting      │ - Saved Searches                 │
└─────────────────────────────────────────────────────────────┘
                              ↓ HTTPS/REST API
┌─────────────────────────────────────────────────────────────┐
│                    API GATEWAY LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  FastAPI / Flask Backend (Python)                           │
│  - /upload - Image upload endpoint                          │
│  - /search - Similarity search endpoint                     │
│  - /recommend - Recommendation endpoint                     │
│  - Authentication & Rate Limiting                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   ML INFERENCE LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  Model Server (TorchServe / Custom)                         │
│  - Load trained CLIP model                                  │
│  - Generate 128D embeddings                                 │
│  - Batch processing support                                 │
│  - GPU acceleration                                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   VECTOR DATABASE LAYER                      │
├─────────────────────────────────────────────────────────────┤
│  FAISS / Pinecone / Qdrant / Weaviate                       │
│  - Store image embeddings (128D vectors)                    │
│  - Fast nearest neighbor search                             │
│  - Filter by metadata (category, price, etc.)               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   STORAGE & DATABASE LAYER                   │
├─────────────────────────────────────────────────────────────┤
│  Object Storage          │  Metadata Database               │
│  - Cloudinary (Free)     │  - PostgreSQL / MongoDB          │
│  - AWS S3 (Free tier)    │  - Supabase (Free)               │
│  - Imgur API             │  - Store image URLs, tags        │
│  - ImageKit.io           │  - User data, search history     │
└─────────────────────────────────────────────────────────────┘
```

---

## 💰 Free Infrastructure Options

### 1. **Image Storage (Free Tier)**

#### Option A: **Cloudinary** ⭐ Recommended
- **Free Tier**: 25GB storage, 25GB bandwidth/month
- **Features**: Auto-optimization, CDN, transformations
- **Pros**: Easy to use, excellent free tier, image transformations
- **Capacity**: ~25,000-50,000 images
- **API**: Simple REST API
```python
# Example
import cloudinary.uploader
result = cloudinary.uploader.upload("image.jpg")
url = result['secure_url']
```

### 2. **Vector Database for Embeddings**

#### Option A: **Pinecone** ⭐ Best for Production
- **Free Tier**: 
  - 1 pod (1M vectors, 100k queries/month)
  - 128-dimensional embeddings
  - Perfect for this project!
- **Features**: Managed service, auto-scaling, metadata filtering
- **Capacity**: ~1,000,000 images
```python
import pinecone
pinecone.init(api_key="YOUR_KEY")
index = pinecone.Index("image-search")
index.upsert(vectors=[("id1", embedding, {"url": "..."})])
```

#### Option B: **Qdrant Cloud**
- **Free Tier**: 1GB cluster (256 dimensional vectors)
- **Features**: Open source, good documentation
- **Capacity**: ~8,000,000 vectors (128D)

#### Option C: **Weaviate Cloud**
- **Free Tier**: 30-day trial, then limited free tier
- **Features**: GraphQL API, hybrid search
- **Capacity**: Varies

#### Option D: **Local FAISS** ⭐ Best for Development
- **Cost**: FREE (runs on your server)
- **Features**: Lightning fast, no API limits
- **Cons**: You manage infrastructure
- **Capacity**: Limited only by RAM/disk
- **Best for**: MVP, testing, small-scale deployments

### 3. **Compute & Hosting**

#### Model Inference:
- **Hugging Face Spaces** (Free): Host Gradio/Streamlit apps with GPU
- **Google Colab** (Free): Training and experimentation
- **Railway.app** (Free tier): Deploy Python backends
- **Render.com** (Free tier): 750 hours/month
- **Fly.io** (Free tier): Small containers

#### Backend API:
- **Vercel** (Free): Serverless functions (Node.js)
- **Railway.app** (Free): $5 credit/month
- **Render.com** (Free): Web services
- **Heroku** (Limited free): Dynos available

#### Frontend:
- **Vercel** (Free): Next.js hosting ⭐
- **Netlify** (Free): Static sites
- **GitHub Pages** (Free): Static sites

### 4. **Database for Metadata**

#### Option A: **Supabase** ⭐ Recommended
- **Free Tier**: 500MB database, 1GB file storage
- **Features**: PostgreSQL, real-time, auth, storage
- **Capacity**: 100,000+ records
```sql
CREATE TABLE images (
    id UUID PRIMARY KEY,
    url TEXT,
    embedding_id TEXT,
    category TEXT,
    tags TEXT[],
    created_at TIMESTAMP
);
```

#### Option B: **MongoDB Atlas**
- **Free Tier**: 512MB storage
- **Features**: NoSQL, flexible schema
- **Capacity**: 100,000+ documents

#### Option C: **PlanetScale**
- **Free Tier**: 5GB storage, 1 billion reads/month
- **Features**: MySQL-compatible, branching

---

## 🚀 Recommended Free Stack (0 Cost MVP)

### For ~100,000 Images:

| Component | Service | Free Tier |
|-----------|---------|-----------|
| **Image Storage** | Cloudinary | 25GB (~25k images) |
| **Vector DB** | Pinecone | 1M vectors |
| **Metadata DB** | Supabase | 500MB PostgreSQL |
| **Backend API** | Railway/Render | Free tier |
| **Model Hosting** | Hugging Face Spaces | GPU inference |
| **Frontend** | Vercel | Unlimited |
| **CDN** | Cloudinary/Cloudflare | Included |

**Total Monthly Cost**: $0 (within free tiers)

**Capacity**: 
- 25,000 images (Cloudinary limit)
- 1,000,000 searchable vectors (Pinecone limit)
- Unlimited searches (with reasonable rate limits)

---

## 💡 Scaling Strategy

### For 1M+ Images:

| Users/Month | Images | Storage | Vector DB | Cost Estimate |
|-------------|--------|---------|-----------|---------------|
| <1K | <25K | Cloudinary Free | Pinecone Free | **$0** |
| 1K-10K | 25K-100K | Cloudinary $99 | Pinecone Free | **$99/mo** |
| 10K-100K | 100K-500K | S3 ~$10 | Pinecone $70 | **$80-150/mo** |
| 100K+ | 1M+ | S3 ~$50 | Qdrant/Custom | **$200-500/mo** |

---

## 🛠️ Quick Start

### Prerequisites

```bash
# Python 3.8+
python --version

python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 1. Fashion MNIST (From Scratch)

```bash
cd FashionMNIST
jupyter notebook main.ipynb
# Run all cells to train and evaluate
```

---
