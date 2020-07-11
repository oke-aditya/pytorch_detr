import dataset
import utils
from pprint import pprint
import config
from torchvision import transforms as T
import pandas as pd
from tqdm import tqdm
import model
# from model import detr_model
import engine
import numpy as np
import detr_loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def train_transforms(img_height, img_width):
    A.Compose([A.Resize(height=img_height, width=img_width, p=1.0), ToTensorV2(p=1.0)], p=1.0, 
        bbox_params=A.BboxParams(format='coco', min_area=0, 
        min_visibility=0, label_fields=['labels']))

def val_transforms(img_height, img_width):
    A.Compose([A.Resize(height=img_height, width=img_width, p=1.0), ToTensorV2(p=1.0)], p=1.0, 
        bbox_params=A.BboxParams(format='coco', min_area=0, 
        min_visibility=0, label_fields=['labels']))

def run():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_df = pd.read_csv(config.TRAIN_CSV_PATH)
    valid_df = pd.read_csv(config.VALIDATION_CSV_PATH)
    
    train_dataset = dataset.DetectionDataset(train_df, config.IMAGE_DIR, config.TARGET, 
    transforms=train_transforms(config.IMG_HEIGHT, config.IMG_WIDTH),)
    
    valid_dataset = dataset.DetectionDataset(valid_df, config.IMAGE_DIR, config.TARGET, 
    transforms=train_transforms(config.IMG_HEIGHT, config.IMG_WIDTH), )

    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
    shuffle=False, collate_fn=utils.collate_fn)

    valid_dataloader = DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, 
                                collate_fn=utils.collate_fn)
    
    print("Data Loaders created")

    detector = model.detr_model(num_classes=config.NUM_CLASSES, num_queries=config.NUM_QUERIES,
    backbone=config.BACKBONE, pretrained=config.PRETRAINED)

    print('''Model Created with backbone = {}, pretrained = {}, 
        number of classes = {}, number of queries = {}'''.format(config.BACKBONE, config.pretrained,
        config.NUM_CLASSES, config.NUM_QUERIES))
    
    matcher = detr_loss.HungarianMatcher()
    weight_dict = {"loss_ce" : 1, "loss_bbox" : 1, "loss_giou" : 1}
    losses = ['labels', 'boxes', 'cardinality']

    optimizer = optim.Adam(detector.parameters(), lr=config.LEARNING_RATE)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    detector.to(device)

    criterion = detr_loss.SetCriterion(config.NUM_CLASSES - 1, matcher, weight_dict, eos_coef=config.NULL_CLASS_COEF, losses=losses)
    criterion.to(device)

    max_loss = 99999

    print("------- Training Started ----- ")

    for epoch in tqdm(range(config.EPOCHS)):
        print("Epoch = {}".format(epoch))
        train_loss = engine.train_fn(train_dataloader, detector, criterion, optimizer, device)
        validation_loss = engine.eval_fn(valid_dataloader, detector, criterion, device)

        if validation_loss.avg < max_loss:
            max_loss = validation_loss.avg
            print("Validation Loss reduced than previous stage. Saving new model")
            torch.save(detector.state_dict(), config.MODEL_SAVE_PATH)
            print('-' * 25)
            print("Model Trained and Saved to Disk")

if __name__ == "__main__":
    run()


