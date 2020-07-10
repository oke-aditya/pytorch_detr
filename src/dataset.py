import os
import numpy as np
import cv2
import torch
import albumentations as A
from torch.utils.data import DataLoader, Dataset

class DetectionDataset(Dataset):
    def __init__(self, dataframe, image_dir, target, transforms=None, train=True):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.image_dir = image_dir
        self.transforms = transforms
        self.df = dataframe
        self.train = train
        self.target = target

    def __len__(self):
        return self.image_ids.shape[0]
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_src = os.path.join(self.image_dir, str(image_id))
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Scale down the pixel values of image
        image /= 255.0

        # if self.transforms is not None:  # Apply transformation
        #     image = self.transforms(image)
        
        if(self.train is False):  # For test data
            return image, image_id
        
        # Else for train and validation data
        records = self.df[self.df['image_id'] == image_id]

        # DETR takes in data in coco format 
        boxes = records[['xtl', 'ytl', 'w', 'h']].values

        # Area of bb
        area = boxes[:, 2] * boxes[:, 3]
        area = torch.as_tensor(area, dtype=torch.float32)

        # We have a labels column it is multi object supported.
        labels = records[self.target].values
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels,
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']

        # Normalize the bounding boxes
            
        _, h, w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'], rows=h, cols=w)
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.long)
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        
        return image, target, image_id  
