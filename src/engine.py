import torch
import torch.nn as nn
from tqdm import tqdm
import utils
import albumentations as A
import config

# Note this engine differs from the engine.py provided by the FB detr team.
# This engine is for fine-tuning and does a training_step.
# The train function is different for detr as we have to train the criterion too.

def train_fn(train_dataloader, detector, criterion, optimizer, device, scheduler=None):
    detector.train()
    criterion.train()

    total_loss = utils.AverageMeter()
    bbox_loss = utils.AverageMeter()
    giou_loss = utils.AverageMeter()
    labels_loss = utils.AverageMeter()

    for images, targets, image_ids in tqdm(train_dataloader):
        images = list(image.to(device) for image in images)
        # it's key:value for t in targets.items
        # This is the format detr expects
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = detector(images)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        
        total_loss.update(losses.item(), config.BATCH_SIZE)
        bbox_loss.update(loss_dict['loss_bbox'].item())
        giou_loss.update(loss_dict['loss_giou'].item())
        labels_loss.update(loss_dict['loss_ce'].item())

    print("Training: ")
    print("Total_loss = {}, BBox_Loss = {}, GIOU_Loss = {}, Labels_Loss = {}".format(total_loss.avg, bbox_loss.avg, giou_loss.avg, labels_loss.avg))
    return total_loss

@torch.no_grad()
def eval_fn(val_dataloader, detector, criterion, device):
    detector.eval()
    criterion.eval()
    eval_loss = utils.AverageMeter()

    for images, targets, image_ids in tqdm(val_dataloader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = detector(images)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    
        bbox_loss = loss_dict['loss_bbox'].item()
        giou_loss = loss_dict['loss_giou'].item()
        labels_loss = loss_dict['loss_ce'].item()
        eval_loss.update(losses.item(), config.BATCH_SIZE)

    print("Validation: ")
    print("Total_loss = {}, BBox_Loss = {}, GIOU_Loss = {}, Labels_Loss = {}".format(eval_loss.avg, bbox_loss.avg, giou_loss.avg, labels_loss.avg))

    return eval_loss


