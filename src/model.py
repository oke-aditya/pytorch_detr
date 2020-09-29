import torch
import torch.nn as nn

# I wrote a function to get the DETR Model from torch hub
# I am planning to add support for pytorch image models by ross as backbones
# I load the Pytorch hub model from detr

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

__all__ = ["detr_model"]


class detr_model(nn.Module):
    def __init__(self, num_classes, num_queries, backbone, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.model = torch.hub.load("facebookresearch/detr", backbone, pretrained=True)
        self.in_features = self.model.class_embed.in_features

        self.model.class_embed = nn.Linear(
            in_features=self.in_features, out_features=self.num_classes
        )
        self.model.num_queries = self.num_queries

    def forward(self, images):
        return self.model(images)
