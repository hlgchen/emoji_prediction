import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class DescriptionSembert(nn.Module):
    def __init__(self, pretrained_path=None):
        super(DescriptionSembert, self).__init__()

        model_name = "all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"sentence-transformers/{model_name}"
        )
        self.model = AutoModel.from_pretrained(f"sentence-transformers/{model_name}")
        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, map_location=device))

    def forward(self, description_ls):
        encoded_input = self.tokenizer(
            description_ls,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        )

        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = mean_pooling(model_output, attention_mask)
        sentence_embeddings = F.normalize(embeddings, p=2, dim=1)

        return sentence_embeddings


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


class CosineDistance(nn.Module):
    def __init__(self):
        super(CosineDistance, self).__init__()
        self.cos_sim = nn.CosineSimilarity()

    def forward(self, x, y):
        return 1 - self.cos_sim(x, y)
