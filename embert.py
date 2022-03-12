import os
import pandas as pd
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
    """
    Returns concatenated pretrained embeddings.
    Params:
        - image {bool}: if True image embeddings are part of the concatenation
                        Note that these embeddings are NOT normalized --> shape (n_emojis, 768)
        - bert {bool}: if True bert description embeddings are part of the concatenation
                        Note that these embeddings are normalized --> shape (n_emojis, 200)
        - wordvector {bool}: if True glove word average embeddings (of the emoji descriptions)
                            are part of the concatenation (glove-twitter-200) --> shape (n_emojis, 200)
    Returns:
        - emoji_embeddings {torch.Tensor}: tensor of concatenated embeddings of all emojis
    """

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
    """
    Takes model output of bert model on token level
    (either as direct sentence bert output or as tensor),
    along with attention mask and calculates the mean of the tensors.

    This represents the sentence embedding of a sentence.
    """
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


class Baseline(nn.Module):
    """
    Baseline Sentence BERT emoji prediction model. It encodes each tweet with
    "all-mpnet-base-v1" and takes the dot product of the embedding with embeddings of
    emoji descriptions. The dot product with each emoji is returned.
    """

    def __init__(self):
        super(Baseline, self).__init__()
        self.emoji_embeddings = nn.Parameter(
            get_emoji_fixed_embedding(image=False, bert=True, wordvector=False),
            requires_grad=False,
        )
        self.emoji_embedding_size = self.emoji_embeddings.size(1)

        model_name = "all-mpnet-base-v1"
        self.model = SentenceTransformer(model_name)
        for _, params in self.model.named_parameters():
            params.requires_grad = False
        self.sentence_embedding_size = 768

    def forward(self, sentence_ls, emoji_ids):
        """
        Returns dot product of encoded sentence with each emoji embedding.
        Params:
            - sentence_ls {list}: list of string sentences
            - emoji_ids {list}: list of emoji ids for comparisson
        """

        sentence_embeddings = self.model.encode(
            sentence_ls, normalize_embeddings=True, convert_to_tensor=True
        )
        emoji_embeddings = self.emoji_embeddings[emoji_ids]

        X_1 = sentence_embeddings.repeat_interleave(len(emoji_ids), dim=0)
        X_2 = emoji_embeddings.repeat(len(sentence_ls), 1)

        out = (X_1 * X_2).sum(dim=1).view(-1, len(emoji_ids))

        return out


class LiteralModel(nn.Module):
    """
    Baseline word level BERT emoji prediction model. It takes the average embedding of words in emoji names
    (embeddings calculated with "all-MiniLM-L6-v2") and compares those with "all-MiniLM-L6-v2" embeddings of each
    word in a tweet. Returns the similarity between tweet and emojis as the maximum word similarity of any word
    in a tweet and the average emoji name embedding. (Note that I didn't use glove vectors, because these have
    performed quite bad when there is spelling issues or small word changes.)
    """

    def __init__(self):
        super(LiteralModel, self).__init__()
        description_path = os.path.join(
            get_project_root(), "emoji_embedding/data/processed/emoji_descriptions.csv"
        )
        df = pd.read_csv(description_path, usecols=["emjpd_emoji_name_og"])

        model_name = "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)
        for _, params in self.model.named_parameters():
            params.requires_grad = False
        self.sentence_embedding_size = 384

        self.emoji_embeddings = nn.Parameter(
            self.model.encode(
                df.emjpd_emoji_name_og.tolist(),
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_tensor=True,
            ),
            requires_grad=False,
        )
        self.emoji_embedding_size = self.emoji_embeddings.size(1)

    def forward(self, sentence_ls, emoji_ids):

        out = []
        for sentence in sentence_ls:
            X = sentence.split(" ")
            X_tensor = self.model.encode(
                X,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )
            dot = (X_tensor @ self.emoji_embeddings[emoji_ids].transpose(0, 1)).max(
                dim=0
            )[0]
            out.append(dot)
        out = torch.stack(out)

        return out


class Embert(nn.Module):
    """
    DistilBERT based emoji prediction model (Emoji-BERT). Has a  "distilbert-base-uncased" as
    basemodel, which is finetuned with training. DistilBERT encodes tweets. Encoded tweets are projected
    into a 500 dimensional space, just like the embeddings for emojis (which are calculated beforehand).
    The dot product within this 500 dimensional space is the similarity.
    Finally the similarity score is passed through a softmax layer.
    """

    def __init__(self, mode="avg"):
        super(Embert, self).__init__()
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


class Sembert(nn.Module):
    """
    SentenceBERT based emoji prediction model (Sentence-Emoji-BERT). Has a  "all-distilroberta-v1" as
    basemodel, which is finetuned with training. SentenceBERT encodes tweets. Encoded tweets are projected
    into a 500 dimensional space, just like the embeddings for emojis (which are calculated beforehand).
    The dot product within this 500 dimensional space is the similarity.
    Finally the similarity score is passed through a softmax layer.

    In this model there is an option to specify a dropout parameter. If no dropout float value is provided
    NO dropout layer is added. Otherwise there is. Be careful when loading finetuned models, add a dropout layer to model
    definition if model had been trained with dropout.
    """

    def __init__(self, dropout=None):
        super(Sembert, self).__init__()
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
        if isinstance(dropout, float):
            self.dropout = nn.Dropout(dropout)

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

        if hasattr(self, "dropout"):
            X_1 = self.dropout(X_1)
            X_2 = self.dropout(X_2)

        X_1 = self.linear1(X_1)
        X_2 = self.linear2(X_2)

        out = (X_1 * X_2).sum(dim=1).view(-1, len(emoji_ids))
        out = F.softmax(out, dim=1)

        return out


class SimpleSembert(nn.Module):
    """
    SentenceBERT based emoji prediction model (Sentence-Emoji-BERT). Has a  "all-distilroberta-v1" as
    basemodel, which is NOT finetuned with training. SentenceBERT encodes tweets. Encoded tweets are projected
    into a 500 dimensional space, just like the embeddings for emojis (which are calculated beforehand).
    The dot product within this 500 dimensional space is the similarity.
    Finally the similarity score is passed through a softmax layer.

    Note that in this version only the linear projection layers are trainable.
    """

    def __init__(self):
        super(SimpleSembert, self).__init__()
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


class EmbertLoss(nn.Module):
    """Custom loss function, inspired by triplet loss. Incentivizes model to lift up probabilities for
    emojis appearing with texts and push down those for all other emojis.Doesn't require us to find hard negatives.
    """

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
    """Custom Accuracy function. For each tweet the same number of emojis are considered for prediction
    as number of emojis that actually appear. The accuracy is the average accuracy of predictions for each sentence."""

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
