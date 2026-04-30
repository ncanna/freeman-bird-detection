'''
The original coco_utils.py, without removing frames with multi annotations

Load image annotations from COCO json format, for example:
/home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/wz_data/H23/Annotations/instances_merged.json
'''

import json

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
    
    fname_to_annos = {img['file_name']: [] for img in coco['images']} # Create an empty dictionary to populate later
    for ann in coco['annotations']:
        # Keep the annotation an empty list if there is no annotation
        fname = image_id_to_filename[ann['image_id']]
        fname_to_annos[fname].append(ann)
    
    return fname_to_annos