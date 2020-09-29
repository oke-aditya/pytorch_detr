# import dataset
import utils
from pprint import pprint

# import config
from torchvision import transforms as T
# import pandas as pd
from tqdm import tqdm
import model

# from model import detr_model
import engine
import numpy as np
import detr_loss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class DummyDetectionDataset(Dataset):
    def __init__(
        self, img_shape=(3, 256, 256), num_boxes=1, num_classes=2, num_samples=10):
        super().__init__()
        self.img_shape = img_shape
        self.num_samples = num_samples
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def _random_bbox(self):
        c, h, w = self.img_shape
        xs = torch.randint(w, (2,))
        ys = torch.randint(h, (2,))
        return [min(xs), min(ys), max(xs), max(ys)]

    def __getitem__(self, idx):
        img = torch.rand(self.img_shape)
        boxes = torch.tensor([self._random_bbox() for _ in range(self.num_boxes)], dtype=torch.float32)
        labels = torch.randint(self.num_classes, (self.num_boxes,), dtype=torch.long)
        return img, {"boxes": boxes, "labels": labels}


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    detector = model.detr_model(
        num_classes=91, num_queries=10, backbone="detr_resnet50", pretrained=True
    )

    batch = {
        "image": torch.randn(1, 3, 224, 224),  # Image-net shape
        #  "padding_mask": torch.zeros(1, 224, 224, dtype=torch.long),
        "boxes": torch.tensor([1, 10, 78, 80]),
        "labels": torch.tensor([5]),
    }

    matcher = detr_loss.HungarianMatcher()
    weight_dict = {"loss_ce": 1, "loss_bbox": 1, "loss_giou": 1}
    losses = ["labels", "boxes", "cardinality"]

    optimizer = optim.Adam(detector.parameters(), lr=1e-3)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    detector.to(device)

    criterion = detr_loss.SetCriterion(91 - 1, matcher, weight_dict, eos_coef=0.5, losses=losses)
    criterion.to(device)

    # train_loss = engine.train_fn(train_dataloader, detector, criterion, optimizer, device)
    # validation_loss = engine.eval_fn(valid_dataloader, detector, criterion, device)
    # out = detector(batch["image"])
    # print(out)
    # print(type(out))
    # print(out.keys())
    # print(out["pred_logits"].shape)
    # print(out["pred_boxes"].shape)

    test_dataset = DummyDetectionDataset(img_shape=(3, 256, 256), num_boxes=1, num_classes=91, num_samples=10)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    train_loss = engine.train_fn(test_loader, detector, criterion, optimizer, device)
    val_loss = engine.eval_fn(test_loader, detector, criterion, device)
