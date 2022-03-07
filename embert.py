import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    BertConfig,
    AutoModel,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.abspath(__file__))


def get_emoji_fixed_embedding(image=True, bert=True, wordvector=False):

    base_path = os.path.join(get_project_root(), "emoji_embedding/data/processed")
    emb_paths = []
    if image:
        emb_paths.append(os.path.join(base_path, "image_embeddings/all_embeddings.pt"))
    if bert:
        emb_paths.append(os.path.join(base_path, "bert_embeddings/all_embeddings.pt"))
    if wordvector:
        emb_paths.append(
            os.path.join(base_path, "word_vector_embeddings/all_embeddings.pt")
        )
    emb_ls = []
    for emb_path in emb_paths:
        emb = torch.load(emb_path, map_location=device)
        emb_ls.append(emb)
    emoji_embeddings = torch.cat(emb_ls, dim=-1)

    return emoji_embeddings


def mean_pooling(model_output, attention_mask):
    if not isinstance(model_output, torch.Tensor):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
    else:
        token_embeddings = model_output
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class SimpleEmbert(nn.Module):
    def __init__(self, mode="avg"):
        super(SimpleEmbert, self).__init__()
        self.emoji_embeddings = nn.Parameter(
            get_emoji_fixed_embedding(image=True, bert=True, wordvector=False),
            requires_grad=False,
        )
        self.emoji_embedding_size = self.emoji_embeddings.size(1)

        base_model_name = "distilbert-base-uncased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(base_model_name)
        self.model = DistilBertModel.from_pretrained(base_model_name)
        self.mode = mode if mode in ["avg", "last"] else "avg"
        self.sentence_embedding_size = BertConfig.from_pretrained(
            base_model_name
        ).hidden_size

        self.linear1 = nn.Linear(self.sentence_embedding_size, 500)
        self.linear2 = nn.Linear(self.emoji_embedding_size, 500)

    def forward(self, sentence_ls, emoji_ids):
        encoded_input = self.tokenizer(
            sentence_ls, return_tensors="pt", truncation=True, padding=True
        )

        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)
        text_model_output = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        sentence_embedding_ls = []
        if self.mode == "avg":
            for i, l in enumerate(encoded_input.attention_mask.sum(dim=1).tolist()):
                sentence_embedding_ls.append(text_model_output[i, :l].mean(dim=0))
        else:
            for i, l in enumerate(encoded_input.attention_mask.sum(dim=1).tolist()):
                sentence_embedding_ls.append(text_model_output[i, l - 1])

        sentences_embeddings = torch.stack(sentence_embedding_ls)
        emoji_embeddings = self.emoji_embeddings[emoji_ids]

        X_1 = sentences_embeddings.repeat_interleave(len(emoji_ids), dim=0)
        X_2 = emoji_embeddings.repeat(len(sentence_ls), 1)

        X_1 = self.linear1(X_1)
        X_2 = self.linear2(X_2)

        out = (X_1 * X_2).sum(dim=1).view(-1, len(emoji_ids))
        out = F.softmax(out, dim=1)

        return out


class SimpleSembert(nn.Module):
    def __init__(self):
        super(SimpleSembert, self).__init__()
        self.emoji_embeddings = nn.Parameter(
            get_emoji_fixed_embedding(image=True, bert=True, wordvector=False),
            requires_grad=False,
        )
        self.emoji_embedding_size = self.emoji_embeddings.size(1)

        model_name = "all-distilroberta-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"sentence-transformers/{model_name}"
        )
        self.model = AutoModel.from_pretrained(f"sentence-transformers/{model_name}")
        self.sentence_embedding_size = 768

        self.linear1 = nn.Linear(self.sentence_embedding_size, 500)
        self.linear2 = nn.Linear(self.emoji_embedding_size, 500)

    def forward(self, sentence_ls, emoji_ids):

        encoded_input = self.tokenizer(
            sentence_ls,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )

        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = mean_pooling(model_output, attention_mask)
        sentence_embeddings = F.normalize(embeddings, p=2, dim=1)

        emoji_embeddings = self.emoji_embeddings[emoji_ids]

        X_1 = sentence_embeddings.repeat_interleave(len(emoji_ids), dim=0)
        X_2 = emoji_embeddings.repeat(len(sentence_ls), 1)

        X_1 = self.linear1(X_1)
        X_2 = self.linear2(X_2)

        out = (X_1 * X_2).sum(dim=1).view(-1, len(emoji_ids))
        out = F.softmax(out, dim=1)

        return out


class VerySimpleSembert(nn.Module):
    def __init__(self):
        super(VerySimpleSembert, self).__init__()
        self.emoji_embeddings = nn.Parameter(
            get_emoji_fixed_embedding(image=True, bert=True, wordvector=False),
            requires_grad=False,
        )
        self.emoji_embedding_size = self.emoji_embeddings.size(1)

        model_name = "all-distilroberta-v1"
        self.model = SentenceTransformer(model_name)
        for _, params in self.model.named_parameters():
            params.requires_grad = False
        self.sentence_embedding_size = 768

        self.linear1 = nn.Linear(self.sentence_embedding_size, 500)
        self.linear2 = nn.Linear(self.emoji_embedding_size, 500)

    def forward(self, sentence_ls, emoji_ids):

        sentence_embeddings = self.model.encode(
            sentence_ls, normalize_embeddings=True, convert_to_tensor=True
        )
        emoji_embeddings = self.emoji_embeddings[emoji_ids]

        X_1 = sentence_embeddings.repeat_interleave(len(emoji_ids), dim=0)
        X_2 = emoji_embeddings.repeat(len(sentence_ls), 1)

        X_1 = self.linear1(X_1)
        X_2 = self.linear2(X_2)

        out = (X_1 * X_2).sum(dim=1).view(-1, len(emoji_ids))
        out = F.softmax(out, dim=1)

        return out


def get_emoji_descriptions():

    description_path = os.path.join(
        get_project_root(), "emoji_embedding/data/processed/emoji_descriptions.csv"
    )
    df = pd.read_csv(description_path)

    filler = "\u25A1" * 3
    s = df.emjpd_emoji_name_og.fillna("") + filler
    s += df.emjpd_aliases.fillna("") + filler
    s += df.emjpd_description_main.fillna("") + filler
    s_ls = s.tolist()
    return s_ls


class Sembert(nn.Module):
    def __init__(self, mode="avg"):
        super(Sembert, self).__init__()
        self.emoji_embeddings = nn.Parameter(
            get_emoji_fixed_embedding(image=True, bert=False, wordvector=False),
            requires_grad=False,
        )

        sentence_model_name = "all-distilroberta-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"sentence-transformers/{sentence_model_name}"
        )
        self.model = AutoModel.from_pretrained(
            f"sentence-transformers/{sentence_model_name}"
        )
        self.sentence_embedding_size = 768

        description_model_name = "all-MiniLM-L6-v2"
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            f"sentence-transformers/{description_model_name}"
        )
        self.description_model = AutoModel.from_pretrained(
            f"sentence-transformers/{description_model_name}"
        )
        self.description_embedding_size = 384

        descriptions = get_emoji_descriptions()
        self.dtoken = self.description_tokenizer(
            descriptions, return_tensors="pt", truncation=True, padding=True
        )

        self.emoji_embedding_size = (
            self.emoji_embeddings.size(1) + self.description_embedding_size
        )
        self.linear1 = nn.Linear(self.sentence_embedding_size, 500)
        self.linear2 = nn.Linear(self.emoji_embedding_size, 500)

    def partial_forward(self, sentence_ls, model, batch_size):
        if isinstance(sentence_ls, list):
            encoded_input = self.tokenizer(
                sentence_ls, return_tensors="pt", truncation=True, padding=True
            )
        else:
            encoded_input = sentence_ls

        encoded_input["input_ids"] = encoded_input["input_ids"].to(device)
        encoded_input["attention_mask"] = encoded_input["attention_mask"].to(device)

        input_id_ls = torch.split(encoded_input["input_ids"], batch_size)
        attention_mask_ls = torch.split(encoded_input["attention_mask"], batch_size)
        model_output_ls = []
        for input_ids, attention_mask in zip(input_id_ls, attention_mask_ls):
            temp = model(input_ids=input_ids, attention_mask=attention_mask)
            model_output_ls.append(temp[0])
        text_model_output = torch.cat(model_output_ls, dim=0)

        embeddings = mean_pooling(text_model_output, encoded_input["attention_mask"])
        sentence_embeddings = F.normalize(embeddings, p=2, dim=1)
        return sentence_embeddings

    def forward(self, sentence_ls, emoji_ids):
        batch_size = len(sentence_ls)

        # handle twitter text
        sentences_embeddings = self.partial_forward(sentence_ls, self.model, batch_size)

        # handle emoji embedding
        dtoken_input_ids = self.dtoken["input_ids"][emoji_ids]
        dtoken_attention_mask = self.dtoken["attention_mask"][emoji_ids]
        description_tokens = {
            "input_ids": dtoken_input_ids,
            "attention_mask": dtoken_attention_mask,
        }
        description_embeddings = self.partial_forward(
            description_tokens, self.description_model, batch_size
        )
        img_embedding = self.emoji_embeddings[emoji_ids]
        emoji_embeddings = torch.cat([img_embedding, description_embeddings], dim=1)

        # combine the two
        X_1 = sentences_embeddings.repeat_interleave(len(emoji_ids), dim=0)
        X_2 = emoji_embeddings.repeat(len(sentence_ls), 1)

        X_1 = self.linear1(X_1)
        X_2 = self.linear2(X_2)
        out = (X_1 * X_2).sum(dim=1).view(-1, len(emoji_ids))
        out = F.softmax(out, dim=1)

        return out


class EmbertLoss(nn.Module):
    def __init__(self):
        super(EmbertLoss, self).__init__()

    def forward(self, probas, labels):
        loss = 0
        for i in range(len(probas)):
            mask = torch.zeros(probas.shape[1], dtype=torch.bool)
            mask[labels[i]] = True
            correct_average_p = torch.sum(torch.log(probas[i][mask]))
            wrong_average_p = torch.sum(torch.log(1 - probas[i][~mask]))
            loss += (1 / len(labels)) * -(wrong_average_p + correct_average_p)
        return loss


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, probas, labels):
        accuracy = 0
        for i in range(len(probas)):
            y = set(labels[i])
            predicted_emojis = torch.topk(probas[i], len(y))[1]
            predicted_emojis = set(predicted_emojis.tolist())
            accuracy += (1 / len(labels)) * (
                len(predicted_emojis.intersection(y)) / len(y)
            )
        return accuracy


class TopKAccuracy(nn.Module):
    """Top K accuracy score similar to the one used in
        Spencer Cappallo, Stacey Svetlichnaya, Pierre Garrigues, Thomas Mensink, and Cees GM
    Snoek. New modality: Emoji challenges in prediction, anticipation, and retrieval.

        It is counted as a success if at least one emoji in the top k predictions was correct."""

    def __init__(self, k):
        super(TopKAccuracy, self).__init__()
        self.k = k

    def forward(self, probas, labels):
        accuracy = 0
        for i in range(len(probas)):
            y = set(labels[i])
            predicted_emojis = torch.topk(probas[i], self.k)[1]
            predicted_emojis = set(predicted_emojis.tolist())
            accuracy += (1 / len(labels)) * (len(predicted_emojis.intersection(y)) > 0)
        return accuracy
