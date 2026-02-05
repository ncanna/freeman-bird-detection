#!/usr/bin/env python
"""
Copied from old github repo
Prepare COCO annotations for MMDetection
Ensures category IDs start from 0 and checks data integrity
"""

import json
import os
import shutil
from pathlib import Path


def check_and_fix_annotations(ann_file, output_file,split):
    """Check and fix COCO annotations for MMDetection"""
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nProcessing {ann_file}")
    print(f"Original categories: {data['categories']}")
    
    # Check current category IDs
    category_ids = set()
    for ann in data['annotations']:
        category_ids.add(ann['category_id'])
    
    print(f"Unique category IDs found: {sorted(category_ids)}")
    
    # Fix categories to ensure single class with ID 0
    data['categories'] = [
        {
            "id": 0,
            "name": "bird",
            "supercategory": "animal"
        }
    ]
    
    # Map old category IDs to 0
    category_mapping = {old_id: 0 for old_id in category_ids}
    
    # Update annotations
    for ann in data['annotations']:
        ann['category_id'] = category_mapping[ann['category_id']]
    
    # Verify all images have corresponding files and add dimensions if missing
    img_dir = Path(ann_file).parent.parent / 'images' / split
    valid_images = []
    
    for img in data['images']:
        img_path = img_dir / img['file_name']
        if img_path.exists():
            # Add width/height if missing
            if 'width' not in img or 'height' not in img or img['width'] is None or img['height'] is None:
                from PIL import Image
                pil_img = Image.open(str(img_path))
                img['width'] = pil_img.width
                img['height'] = pil_img.height
                print(f"Added dimensions for {img['file_name']}: {img['width']}x{img['height']}")
            valid_images.append(img)
        else:
            print(f"Warning: Image not found: {img_path}")
    
    data['images'] = valid_images
    
    # Save fixed annotations
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Fixed annotations saved to: {output_file}")
    print(f"Total images: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")
    
    return data


def create_mmdet_data_structure():
    """Create proper directory structure for MMDetection"""
    
    # Create directories if needed
    os.makedirs('data/mmdet_format/annotations', exist_ok=True)
    
    # Process each split
    splits = ['train', 'val', 'test']
    
    for split in splits:
        # Check original annotation file
        orig_ann = f'data/annotations/instances_{split}.json'
        if os.path.exists(orig_ann):
            # Fix and save annotations
            fixed_ann = f'data/mmdet_format/annotations/instances_{split}.json'
            check_and_fix_annotations(orig_ann, fixed_ann,split)
            
            # Create symbolic link for images (to avoid copying)
            img_source = os.path.abspath(f'data/images/{split}')
            img_target = f'data/mmdet_format/images/{split}'
            
            os.makedirs('data/mmdet_format/images', exist_ok=True)
            
            if os.path.exists(img_target):
                os.unlink(img_target)
            os.symlink(img_source, img_target)
            
            print(f"Created symlink: {img_target} -> {img_source}")
        else:
            print(f"Warning: {orig_ann} not found!")


def verify_data():
    """Verify the prepared data"""
    print("\n=== Verifying MMDet Data ===")
    
    for split in ['train', 'val', 'test']:
        ann_file = f'data/mmdet_format/annotations/instances_{split}.json'
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            # Count annotations per category
            cat_counts = {}
            for ann in data['annotations']:
                cat_id = ann['category_id']
                cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
            
            print(f"\n{split} split:")
            print(f"  Images: {len(data['images'])}")
            print(f"  Annotations: {len(data['annotations'])}")
            print(f"  Annotations per category: {cat_counts}")
            
            # Check a few images exist
            img_dir = f'data/mmdet_format/images/{split}'
            if os.path.exists(img_dir):
                img_files = os.listdir(img_dir)[:3]
                print(f"  Sample images: {img_files}")


if __name__ == '__main__':
    print("Preparing data for MMDetection...")
    create_mmdet_data_structure()
    verify_data()
    print("\nData preparation complete!")
    print("You can now use 'data/mmdet_format/' as your data_root in MMDetection configs")