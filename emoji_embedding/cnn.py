import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResnetExt(nn.Module):
    def __init__(self, output_dim):
        """Uses resnet18 as backbone. Changes the last linear layer and adds one on top. The output dimension should be the number of emoji classes for image
        classification."""
        super(ResnetExt, self).__init__()
        self.resnet = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=True
        )
        self.resnet.fc = nn.Linear(512, 300)
        self.prediction_head = nn.Linear(300, output_dim)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        x = self.prediction_head(x)
        return x


class Img2Vec(nn.Module):
    def __init__(self, pretrained_path):
        """Uses resnet18 as backbone. Changes the last linear layer and outputs 300 dim vector for each image."""
        super(Img2Vec, self).__init__()
        self.resnet = torch.hub.load(
            "pytorch/vision:v0.10.0", "resnet18", pretrained=False
        )
        self.resnet.fc = nn.Linear(512, 300)
        print(
            self.load_state_dict(
                torch.load(pretrained_path, map_location=device), strict=False
            )
        )

    def forward(self, x):
        x = self.resnet(x)
        return x
