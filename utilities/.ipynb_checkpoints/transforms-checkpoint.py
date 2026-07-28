'''
Transform the loaded images
'''

# TODO: !! also need to scale bonding box in annotation if used T.Resize((img_height, img_width))

import torchvision.transforms as T

def get_transforms(img_height, img_width, augment=False):
    """
    Returns transforms for train (augment=True) or val/test (augment=False).
    Perform resizing and normalization.
    Args:
        img_height: image height from annotation file
        img_width: image width from annotation file
        augment: whether to apply data augmentation (True for train, False for test and validation data)
    """
    base_transforms = [
        T.Resize((img_height, img_width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ]
    
    if augment:
        base_transforms.insert(1, T.RandomHorizontalFlip(p=0.5))
        base_transforms.insert(2, T.ColorJitter(brightness=0.2,
                                                 contrast=0.2,
                                                 saturation=0.2))
    return T.Compose(base_transforms)