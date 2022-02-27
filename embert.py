import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel, BertConfig

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

        self.prediction_head = nn.Sequential(
            nn.Linear(self.emoji_embedding_size + self.sentence_embedding_size, 1),
            # nn.ReLU(),
            # nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.prediction_head.requires_grad_ = False

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

        X = torch.cat([X_1, X_2], dim=1)
        out = self.prediction_head(X)
        out = out.view(-1, len(emoji_ids))

        # max_probas, max_emojis = out.max(dim=1)
        # return max_probas, max_emojis

        return out


def get_emoji_descriptions():

    description_path = os.path.join(
        get_project_root(), "emoji_embedding/data/processed/emoji_description.csv"
    )
    df = pd.read_csv(description_path)

    filler = "\u25A1" * 3
    s = df.emjpd_emoji_name_og.fillna("") + filler
    s += df.emjpd_aliases.fillna("") + filler
    s += df.emjpd_description_main.fillna("") + filler
    s += df.emjpd_usage_info.fillna("") + filler
    s_ls = s.tolist()
    return np.array(s_ls)


class Embert(nn.Module):
    def __init__(self, mode="avg"):
        super(Embert, self).__init__()
        self.descriptions = get_emoji_descriptions()
        self.emoji_embeddings = nn.Parameter(
            get_emoji_fixed_embedding(image=True, bert=False, wordvector=False),
            requires_grad=False,
        )

        base_model_name = "distilbert-base-uncased"
        self.tokenizer = DistilBertTokenizer.from_pretrained(base_model_name)
        self.emoji_bert = DistilBertModel.from_pretrained(base_model_name)
        self.model = DistilBertModel.from_pretrained(base_model_name)
        self.mode = mode if mode in ["avg", "last"] else "avg"

        self.sentence_embedding_size = BertConfig.from_pretrained(
            base_model_name
        ).hidden_size
        self.emoji_embedding_size = (
            self.emoji_embeddings.size(1) + self.sentence_embedding_size
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(self.emoji_embedding_size + self.sentence_embedding_size, 1),
            nn.Sigmoid(),
        )

    def partial_forward(self, sentence_ls, model):
        encoded_input = self.tokenizer(
            sentence_ls, return_tensors="pt", truncation=True, padding=True
        )

        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)
        text_model_output = model(
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
        return sentences_embeddings

    def forward(self, sentence_ls, emoji_ids):
        # handle twitter text
        sentences_embeddings = self.partial_forward(sentence_ls, self.model)

        # handle emoji embedding
        description_ls = self.descriptions[emoji_ids].tolist()
        description_embeddings = self.partial_forward(description_ls, self.emoji_bert)
        img_embedding = self.emoji_embeddings[emoji_ids]
        emoji_embeddings = torch.cat(img_embedding, description_embeddings, dim=1)

        # combine the two
        X_1 = sentences_embeddings.repeat_interleave(len(emoji_ids), dim=0)
        X_2 = emoji_embeddings.repeat(len(sentence_ls), 1)

        X = torch.cat([X_1, X_2], dim=1)
        out = self.prediction_head(X)
        out = out.view(-1, len(emoji_ids))

        # max_probas, max_emojis = out.max(dim=1)
        # return max_probas, max_emojis

        return out


class EmbertLoss(nn.Module):
    def __init__(self):
        super(EmbertLoss, self).__init__()

    def forward(self, probas, labels):
        loss = 0
        for i in range(len(probas)):
            mask = torch.zeros(probas.shape[1], dtype=torch.bool)
            mask[labels[i]] = True
            correct_average_p = torch.mean(probas[i][mask])
            # wrong_average_p = torch.mean(probas[i][~mask])
            wrong_average_p = torch.topk(probas[i][~mask], 10)[0].mean()
            loss += (1 / len(labels)) * (wrong_average_p - correct_average_p)
            # print(
            #     f"correct_average_p: {correct_average_p:.4f} wrong_average_p: {wrong_average_p:.4f} loss: {loss:.4f} \n"
            # )
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
            print(
                f"predicted emojis: {torch.topk(probas[i], 5)[1]} actual: {y} predicted probas {torch.topk(probas[i], 5)[0]} \n"
            )
            accuracy += (1 / len(labels)) * (
                len(predicted_emojis.intersection(y)) / len(y)
            )
        return accuracy
