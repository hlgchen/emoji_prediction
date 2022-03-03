import torch
import torch.nn as nn
from time import time
import copy

import pandas as pd

from emoji_description_dataset import EDDataset
from ee_model import DescriptionSembert
import utils
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs,
    name,
    base=None,
):
    """ """
    start_time_training = time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    for epoch in range(num_epochs):
        start_time_epoch = time()

        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            with tqdm(enumerate(dataloaders[phase])) as tbatch:
                for i, (anchor_ls, pos_ls, neg_ls) in tbatch:
                    start_time_batch = time()

                    with torch.set_grad_enabled(phase == "train"):
                        optimizer.zero_grad()
                        data_ls = anchor_ls + pos_ls + neg_ls
                        data_tensors = model(data_ls)
                        anchor, positives, negatives = data_tensors.split(
                            len(anchor_ls), dim=0
                        )
                        loss = criterion(anchor, positives, negatives)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * anchor.size(0)
                    tbatch.set_postfix(
                        loss=loss.item() * anchor.size(0),
                        running_loss=running_loss / (i + 1),
                    )

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(
                "{} Loss: {:.4f}, took {}".format(
                    phase, epoch_loss, time() - start_time_batch
                )
            )

            if phase == "valid" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time() - start_time_epoch
        print(
            "Epoch {}/{} FINISHED, took {:.0f}m {:.0f}s".format(
                epoch, num_epochs - 1, time_elapsed // 60, time_elapsed % 60
            )
        )
        print("-" * 10)
        if epoch % 5 == 0:
            if base is None:
                base = os.path.join(
                    utils.get_project_root(), f"emoji_embedding/model/text_{name}/"
                )
            if not os.path.exists(base):
                os.makedirs(base)
            torch.save(
                model.state_dict(),
                os.path.join(
                    base,
                    f"bert_{name}_epoch{epoch}.ckpt",
                ),
            )
            print("model saved")

    time_elapsed = time() - start_time_training
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best valid Acc: {:4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":

    train_df_path = os.path.join(
        utils.get_project_root(),
        "emoji_embedding/data/processed/emoji_descriptions.csv",
    )
    df = pd.read_csv(train_df_path)
    train_df = df.loc[~df.zero_shot].copy()

    dataloaders = {
        "train": EDDataset(train_df, 10, 32),
        "valid": EDDataset(train_df, 2),
    }

    model = DescriptionSembert()
    model.to(device)
    print(utils.model_summary(model, verbose=False, only_trainable=True))
    print(utils.model_summary(model, verbose=False, only_trainable=False))

    criterion = nn.TripletMarginLoss(margin=10.0)
    optimizer = torch.optim.Adam(model.parameters())

    train_model(
        model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=42,
        name="description_sembert",
        base="/content/drive/MyDrive/cs224n_project/trained_models/decription_sembert",
    )
