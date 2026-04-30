'''
Create a dataset from annotation json for train, test, validation split.
Split on the frame paths, such as:
/home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/wz_data/H23/frames/IMG_0158_frame_000218.png

'''

from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utilities.coco_utils import (load_coco_annotations, 
                                  get_image_dimensions,
                                  build_filename_to_annotations)
from utilities.transforms import get_transforms
import pandas as pd

class BirdFrameDataset(Dataset):
    '''
    Check here: https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    Pytorch DataLoader takes a Dataset object, so should subclass from Dataset
    Examples to create custom dataset are here:
    * https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
    * https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
    '''
    def __init__(self, frame_paths, fname_to_annos, transforms=None):
        """
        Args:
            frame_paths: list of paths to frame files (will be converted to Path object), such as:
                         /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/wz_data/H23/frames/IMG_0158_frame_000218.png
            fname_to_annos: dictionary mapping filename to list of annotations, such as:
                            {'IMG_0070_frame_000000.png':[{'id': 1,
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
            transforms: torchvision transforms to apply (resize, normalization, etc.). For example:
                        transforms = T.Compose([T.Resize((1280, 720)),
                                                T.ToTensor(),
                                                T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])   # ImageNet normalization, standard for DETR
                                               ])
        """
        self.frame_paths = [Path(p) for p in frame_paths] # Convert to Path object so it is easy to extrat file name later
        self.fname_to_annos = fname_to_annos
        self.transforms = transforms

    def __len__(self):
        '''
        Get number of images.
        Reccommand to have by pytroch
        '''
        return len(self.frame_paths)

    def __getitem__(self, idx):
        '''
        Access a single element (image) by the given index idx.
        This is necessary for Pytorch DataLoader.
        > supporting fetching a data sample for a given key
        
        Also check out here:
        https://discuss.pytorch.org/t/a-clear-explanation-of-what-num-workers-0-means-for-a-dataloader/177614
        '''
        path = self.frame_paths[idx]
        image = Image.open(path).convert('RGB')
        annos = self.fname_to_annos.get(path.name, [])

        if self.transforms:
            image = self.transforms(image)

        return image, annos

def collate_fn(batch):
    """
    Custom collate function to handle variable number of annotations per image.
    Default collate fails because each image can have different numbers of annotations:
    - frames with 0 birds: annotations = []
    - frames with 1 bird:  annotations = [{'bbox': [...], ...}]
    - frames with 2 bird:  annotations = [{'bbox': [...], ...}, {'bbox': [...], ...}]
    Args:
        batch: list of (image, annotations) tuples from __getitem__()
    Returns:
        images:      stacked tensor of shape [batch_size, 3, H, W]
        annotations: list of annotation lists, one per image (variable length)
    """
    images, annotations = zip(*batch)
    images = torch.stack(images, dim=0)
    annotations = list(annotations)
    return images, annotations

def create_dataloaders(frames_csv, annotation_path, 
                       test_size=0.2, val_size=0.1,
                       batch_size=2, num_workers=4,
                       random_state=42):
    """
    Creates train, validation, and test DataLoaders from a frames 
    directory and COCO annotation file.

    Args:
        frames_csv:       a csv file contains path to frame files, such as (with header):
                          /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/wz_data/H23_frame_list.subset_0.01.csv
        annotation_path:  path to COCO JSON annotation file, such as:
                          /home/hice1/wzhu97/scratch/2026_freeman_bird_behavior/data/wz_data/H23/Annotations/instances_merged.json
        test_size:        proportion of videos for test set
        val_size:         proportion of videos for validation set
        batch_size:       number of samples per batch
        num_workers:      number of DataLoader worker processes. Enables multi-process data loading.
                          Some suggest use 4*num_GPU for this parameter.
        random_state:     random seed for reproducibility, default to 42.

    Returns:
        - train_loader, val_loader, test_loader: iterable objects containing images that will be loaded only when needed (lazy loading)
    """
    # load annotations
    coco           = load_coco_annotations(annotation_path) # Load a COCO json file
    img_h, img_w   = get_image_dimensions(coco) # H23 has W*H = 720*1280
    fname_to_annos = build_filename_to_annotations(coco) # {'IMG_0158_frame_000218.png':[some annotation]}

    # Get all frame paths. Assuming the file has a header row with columns 'Path' and 'Video'
    df_frames = pd.read_csv(frames_csv)
    # ### Special ###
    # Frames with multiple annotaions are excluded when loading the coco annotations.
    # Therefore also need to remove them from the path list.
    df_frames['Frame_name'] = df_frames['Path'].apply(lambda x: Path(x).name)
    df_frames = df_frames[df_frames['Frame_name'].isin(fname_to_annos.keys())]
    all_frames = list(df_frames['Path'])
    
    # split at video level
    video_names = df_frames['Video'].unique()
    train_videos, temp_videos = train_test_split(video_names, train_size=1-test_size-val_size, random_state=random_state)
    val_videos, test_videos = train_test_split(temp_videos,
                                               test_size=val_size/(test_size + val_size),
                                               random_state=random_state) # Further split test and validation

    # Step 3 - map back to frame paths
    train_frames = df_frames[df_frames['Video'].isin(train_videos)]['Path']
    val_frames = df_frames[df_frames['Video'].isin(val_videos)]['Path']
    test_frames = df_frames[df_frames['Video'].isin(test_videos)]['Path']

    # create datasets with appropriate transforms
    train_dataset = BirdFrameDataset(train_frames, fname_to_annos, 
                                     get_transforms(img_h, img_w, augment=True))
    val_dataset   = BirdFrameDataset(val_frames,   fname_to_annos,
                                     get_transforms(img_h, img_w, augment=False))
    test_dataset  = BirdFrameDataset(test_frames,  fname_to_annos,
                                     get_transforms(img_h, img_w, augment=False))

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers,
                              collate_fn=collate_fn)
    
    print(f"\n# ##### Data split summary #####")
    print(f"{'# Split':<10} {'Videos':>8} {'Frames':>8}") # >: left alignment; the number (10, 8): minimum width of characters
    print(f"{'# Train':<10} {len(train_videos):>8} {len(train_frames):>8}")
    print(f"{'# Val':<10} {len(val_videos):>8} {len(val_frames):>8}")
    print(f"{'# Test':<10} {len(test_videos):>8} {len(test_frames):>8}")
    print(f"{'# Total':<10} {len(video_names):>8} {len(all_frames):>8}")

    return train_loader, val_loader, test_loader