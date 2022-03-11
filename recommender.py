import os

import torch
import torch.nn as nn
from embert import Baseline, LiteralModel, SimpleSembert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_IDX = list(range(1711))
TEST_IDX = list(range(1810))


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.abspath(__file__))


def get_sembert_dropout_model(balanced=False):
    model = SimpleSembert(dropout=0.2)
    model = model.to(device)
    if balanced:
        pretrained_path = os.path.join(
            get_project_root(),
            "trained_models/balanced_sembert_dropout/balanced_sembert_dropout_chunk106.ckpt",
        )
    else:
        pretrained_path = os.path.join(
            get_project_root(),
            "trained_models/sembert_dropout/sembert_dropout_chunk77.ckpt",
        )
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.eval()
    return model


class EREC(nn.Module):
    def __init__(self, mapping_dict, l_threshold=0.3):
        super(EREC, self).__init__()
        self.e_model1 = get_sembert_dropout_model(balanced=False)
        self.e_model2 = get_sembert_dropout_model(balanced=True)
        self.l_model1 = Baseline().to(device)
        self.l_model2 = LiteralModel().to(device)
        self.eval()
        self.e_model_dict = {
            "e_model1": self.e_model1,
            "e_model2": self.e_model2,
        }
        self.l_model_dict = {
            "l_model1": self.l_model1,
            "l_model2": self.l_model2,
        }
        self.l_threshold = l_threshold
        self.mapping_dict = mapping_dict

    def get_top_predictions_e(self, pred):
        _, emoji_idx = torch.topk(pred, 100)
        return emoji_idx

    def get_top_predictions_l(self, preds):
        values, emoji_idx = torch.topk(preds, 100)
        ls = []
        for i in range(values.shape[0]):
            ls.append(emoji_idx[i][values[i] > self.l_threshold].tolist())
        return ls

    def forward(self, X, idxset=TEST_IDX):
        e_outputs = {name: m(X, idxset) for name, m in self.e_model_dict.items()}
        e_combined_prediction = sum([pred for pred in e_outputs.values()])
        e_top_preds = self.get_top_predictions_e(e_combined_prediction)

        l_outputs = {name: m(X, idxset) for name, m in self.l_model_dict.items()}
        l_combined_prediction = (l_outputs["l_model1"] * 2 + l_outputs["l_model2"]) / 3
        l_top_preds = self.get_top_predictions_l(l_combined_prediction)

        # custom rule to combine predictions:
        recs_e = e_top_preds.tolist()[:][:100]
        recs_l = l_top_preds
        if self.mapping_dict is not None:
            recs_e = [
                [self.mapping_dict[int(x)] for x in sublist] for sublist in recs_e
            ]
            recs_l = [
                [self.mapping_dict[int(x)] for x in sublist] for sublist in recs_l
            ]

        return recs_e, recs_l
