import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResnetExtClassifier(nn.Module):
    def __init__(self, num_classes, emb_dimension):
        """Uses resnet18 as backbone. Changes the last linear layer and adds one on top. The output dimension should be the number of emoji classes for image
        classification."""
        super(ResnetExtClassifier, self).__init__()
        self.resnet = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=True
        )
        self.resnet.fc = nn.Linear(512, emb_dimension)
        self.prediction_head = nn.Linear(emb_dimension, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.prediction_head(x)
        return x


class Img2Vec(nn.Module):
    def __init__(self, emb_dimension, pretrained_path=None):
        """Uses resnet18 as backbone. Changes the last linear layer and outputs 300 dim vector for each image."""
        super(Img2Vec, self).__init__()

        pretrained_resnet = pretrained_path is None
        self.resnet = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=pretrained_resnet
        )
        self.resnet.fc = nn.Linear(512, emb_dimension)
        if (
            not pretrained_resnet
        ):  # if no pretrained resnet we would want to use our own pretrained model.
            print(
                self.load_state_dict(
                    torch.load(pretrained_path, map_location=device), strict=False
                )
            )

    def forward(self, x):
        x = self.resnet(x)
        return x


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):

        loss = torch.mean(
            1 / 2 * (label) * torch.pow(dist, 2)
            + 1
            / 2
            * (1 - label)
            * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )

        return loss
