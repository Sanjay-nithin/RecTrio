"""
Image Upscaling Script for Fashion Dataset

This script upscales small images in the fashion dataset using high-quality
LANCZOS resampling while maintaining aspect ratio and preserving quality.

Usage:
    python upscale_images.py --input datasets/fashion --output datasets/fashion_upscaled --min-size 300
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def upscale_image(image, min_size=300, max_size=800):
    """
    Upscale small images while maintaining quality using LANCZOS resampling.
    
    Args:
        image: PIL Image object
        min_size: Minimum dimension size (will upscale if smaller)
        max_size: Maximum dimension size (won't upscale beyond this)
        
    Returns:
        Upscaled PIL Image
    """
    width, height = image.size
    min_dim = min(width, height)
    max_dim = max(width, height)
    
    # If image is already large enough, return as-is
    if min_dim >= min_size:
        return image
    
    # Calculate scale factor to reach minimum size
    scale_factor = min_size / min_dim
    
    # Don't upscale beyond max_size
    if max_dim * scale_factor > max_size:
        scale_factor = max_size / max_dim
    
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Use LANCZOS for high-quality upscaling
    upscaled = image.resize((new_width, new_height), Image.LANCZOS)
    
    return upscaled


def process_dataset(input_dir, output_dir, min_size=300, max_size=800, extensions=None):
    """
    Process all images in the dataset directory structure.
    
    Args:
        input_dir: Path to input dataset directory
        output_dir: Path to output directory for upscaled images
        min_size: Minimum dimension size for upscaling
        max_size: Maximum dimension size limit
        extensions: List of image file extensions to process
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all image files
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.rglob(f"*{ext}"))
        image_files.extend(input_path.rglob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(image_files)} images")
    print(f"Upscaling settings: min={min_size}px, max={max_size}px")
    print(f"Interpolation: LANCZOS (high quality)")
    print()
    
    stats = {
        'total': len(image_files),
        'upscaled': 0,
        'unchanged': 0,
        'errors': 0
    }
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images", unit="img"):
        try:
            # Calculate relative path to maintain directory structure
            rel_path = img_path.relative_to(input_path)
            out_path = output_path / rel_path
            
            # Create output subdirectories
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if output file already exists and is newer
            if out_path.exists() and out_path.stat().st_mtime > img_path.stat().st_mtime:
                stats['unchanged'] += 1
                continue
            
            # Load and process image
            with Image.open(img_path) as img:
                original_size = img.size
                
                # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                
                # Upscale if needed
                upscaled_img = upscale_image(img, min_size=min_size, max_size=max_size)
                new_size = upscaled_img.size
                
                # Save with high quality
                upscaled_img.save(out_path, quality=95, optimize=True)
                
                if new_size != original_size:
                    stats['upscaled'] += 1
                else:
                    stats['unchanged'] += 1
                    
        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {str(e)}")
            stats['errors'] += 1
    
    # Print summary
    print()
    print("=" * 60)
    print("📊 Processing Summary")
    print("=" * 60)
    print(f"Total images processed: {stats['total']}")
    print(f"Upscaled: {stats['upscaled']} ({stats['upscaled']/stats['total']*100:.1f}%)")
    print(f" Unchanged (already large): {stats['unchanged']} ({stats['unchanged']/stats['total']*100:.1f}%)")
    if stats['errors'] > 0:
        print(f"❌ Errors: {stats['errors']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Upscale small images in fashion dataset using high-quality LANCZOS resampling"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='datasets/fashion',
        help='Input directory containing images (default: datasets/fashion)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='datasets/fashion_upscaled',
        help='Output directory for upscaled images (default: datasets/fashion_upscaled)'
    )
    parser.add_argument(
        '--min-size', '-m',
        type=int,
        default=300,
        help='Minimum dimension size for upscaling in pixels (default: 300)'
    )
    parser.add_argument(
        '--max-size', '-M',
        type=int,
        default=800,
        help='Maximum dimension size limit in pixels (default: 800)'
    )
    parser.add_argument(
        '--extensions', '-e',
        nargs='+',
        default=['.jpg', '.jpeg', '.png'],
        help='Image file extensions to process (default: .jpg .jpeg .png)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        input_path = Path(args.input)
        image_files = []
        for ext in args.extensions:
            image_files.extend(input_path.rglob(f"*{ext}"))
        
        print(f"Dry run mode")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f" Would process {len(image_files)} images")
        print(f"Min size: {args.min_size}px")
        print(f"Max size: {args.max_size}px")
        return
    
    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        min_size=args.min_size,
        max_size=args.max_size,
        extensions=args.extensions
    )


if __name__ == '__main__':
    main()
