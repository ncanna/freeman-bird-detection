'''
Load image annotations from COCO json format, for example:
/home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/wz_data/H23/Annotations/instances_merged.json
'''

import json
import torch

def load_coco_annotations(annotation_path):
    """Load and validate a COCO annotation file."""
    with open(annotation_path, 'r') as f:
        coco = json.load(f)
    
    # validate required keys
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        assert key in coco, f"Missing required key in COCO file: {key}"
    
    return coco

def get_image_dimensions(coco):
    """
    Extract consistent image dimensions from COCO annotations.
    In current annotation file of H23, the frames have W*H = 720*1280
    """
    sizes = set([(img['height'], img['width']) for img in coco['images']])
    assert len(sizes) == 1, f"Inconsistent image sizes found: {sizes}"
    height, width = sizes.pop()
    return height, width

def build_filename_to_annotations(coco):
    """
    Build a lookup dictionary from filename to its annotations.
    Return:
    - fname_to_annos: a dictionary with items like {'IMG_0070_frame_000000.png':[{'id': 1,
                                                                                  'image_id': 1,
                                                                                  'category_id': 1,
                                                                                  'segmentation': [],
                                                                                  'area': 10292.10250000001,
                                                                                  'bbox': [1177.5, 551.77, 101.45, 101.45],
                                                                                  'iscrowd': 0,
                                                                                  'attributes': {'occluded': False,
                                                                                                 'rotation': 0.0,
                                                                                                 'track_id': 0,
                                                                                                 'keyframe': True}}]
                                                    }
    """
    # The file name in annotation file does not contain absolute path
    # An example file name: IMG_0070_frame_000000.png
    image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
    count_multi_anno = 0 # Track number of images with multiple annotations (They will be excluded)
    fname_to_annos = {img['file_name']: [] for img in coco['images']} # Create an empty dictionary to populate later
    for ann in coco['annotations']: 
        fname = image_id_to_filename[ann['image_id']]
        # Keep the annotation an empty list if there is no annotation
        if fname not in fname_to_annos:
            # Already removed due to myltple annotations, so skip
            continue
        elif fname_to_annos[fname]!=[]:
            # Exclude the image if it has > annotations (ie. detect multiple birds)
            fname_to_annos.pop(fname)
            count_multi_anno += 1
        else:
            fname_to_annos[fname].append(ann)
    print(f'# Frames with >1 annotations were excluded: N={count_multi_anno}')
    return fname_to_annos


def coco_annos_to_detr_targets(annotations, img_h, img_w, device):
    """
    Convert COCO annotation format to DETR target format.
    This is required as specified in the detr_detector/model/matcher.py (forward fucntion)
    
    COCO bbox format: [x_min, y_min, width, height] (absolute pixels)
    DETR bbox format: [cx, cy, w, h] (normalized 0-1 by image dimensions)
    
    Args:
        annotations: list of annotation lists from DataLoader, one per image. e.g.:
                     [[{'bbox': [x, y, w, h], 'category_id': 1, ...}], [], ...]
        img_h:       image height in pixels
        img_w:       image width in pixels  
        device:      torch device to put tensors on
    Returns:
        targets: list of dicts with 'labels' and 'boxes' tensors, one per image.
                 The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
    targets = []
    for anno_list in annotations:
        if len(anno_list) == 0:
            # empty frame - no birds
            targets.append({
                'labels': torch.zeros(0, dtype=torch.long).to(device),
                'boxes':  torch.zeros((0, 4), dtype=torch.float32).to(device)
            })
        else:
            boxes  = []
            labels = []
            for ann in anno_list:
                # convert COCO bbox [x_min, y_min, w, h] → DETR [cx, cy, w, h] normalized
                x_min, y_min, w, h = ann['bbox']
                cx = (x_min + w / 2) / img_w    # normalize by image width
                cy = (y_min + h / 2) / img_h    # normalize by image height
                w  = w / img_w
                h  = h / img_h
                boxes.append([cx, cy, w, h])
                labels.append(0)                 # 0 = bird class (0-indexed)
            
            targets.append({
                'labels': torch.tensor(labels, dtype=torch.long).to(device),
                'boxes':  torch.tensor(boxes,  dtype=torch.float32).to(device)
            })
    
    return targets