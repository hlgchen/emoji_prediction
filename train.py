import os
from time import time
import torch
import torch.nn as nn
import copy
from tqdm import tqdm

from pprint import pprint
from emoji_embedding.utils import model_summary
from embert import EmbertLoss, SimpleEmbert, Accuracy
from twemoji.twemoji_dataset import TwemojiData, TwemojiDataChunks


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IDX = list(range(1711))
TEST_IDX = list(range(1810))


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.abspath(__file__))


def train_model(
    model,
    dataloader_ls,
    criterion,
    optimizer,
    num_epochs,
    name,
):
    """
    Trains model with specified setup. Models are saved after every chunk.

    Params:
        - model {torch.nn.Module}: model to be trained. For the task at hand this isa version of embert
        - dataloader_ls {list}: list of dictionaries containing the dataloaders. The dictionaries are expected
                                to contain dataloaders for train and valid.
        - criterion {torch loss function}: for the task at hand, this is the embertloss
        - optimizer {torch optimizer}: optimizer for training, i.e. adam
        - num_epochs {int}: number of epochs to train.
        - name {str}: name of the model to be saved
    """
    acc = Accuracy()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        print("-" * 10)

        for chunk_i, dataloaders in enumerate(dataloader_ls):
            start_time_chunk = time()
            for phase in ["train", "valid"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_accuracy = 0

                with tqdm(enumerate(dataloaders[phase])) as tbatch:
                    for i, batch in tbatch:
                        start_time_batch = time()

                        sentences_ls = batch[0]
                        labels_ls = batch[1]

                        with torch.set_grad_enabled(phase == "train"):
                            optimizer.zero_grad()
                            outputs = model(sentences_ls, TRAIN_IDX)
                            loss = criterion(outputs, labels_ls)
                            if phase == "train":
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_accuracy += acc(outputs, labels_ls)
                        running_loss += loss.item() * len(labels_ls)
                        tbatch.set_postfix(
                            loss=loss.item() * len(labels_ls),
                            running_loss=running_loss / (i + 1),
                            running_accuracy=running_accuracy / (i + 1),
                        )

                chunk_loss = running_loss / (i + 1)
                chunk_accuracy = running_accuracy / (i + 1)

                print(
                    "{} Loss: {:.4f}, Accuracy: {:.4f}, took {}".format(
                        phase,
                        chunk_loss,
                        chunk_accuracy,
                        time() - start_time_batch,
                    )
                )

            time_elapsed = time() - start_time_chunk
            print(
                "Chunk {}/{} FINISHED, took {:.0f}m {:.0f}s".format(
                    chunk_i,
                    len(dataloader_ls),
                    time_elapsed // 60,
                    time_elapsed % 60,
                )
            )
            print("-" * 10)
            base = os.path.join(get_project_root(), f"trained_models/run1/")
            if not os.path.exists(base):
                os.makedirs(base)
            torch.save(
                model.state_dict(),
                os.path.join(
                    base,
                    f"{name}_chunk{chunk_i+1}.ckpt",
                ),
            )
            print("model saved")


if __name__ == "__main__":

    pretrained_path = None
    model = SimpleEmbert()
    model.train()
    model = model.to(device)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    # pprint(model_summary(model))

    train_data_chunks = TwemojiDataChunks(
        "train", chunksize=64000, shuffle=True, batch_size=64
    )
    valid_data = TwemojiData("valid", shuffle=True, batch_size=64, limit=6400)
    dataloader_ls = [
        {"train": train_data, "valid": valid_data} for train_data in train_data_chunks
    ]

    criterion = EmbertLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    train_model(
        model,
        dataloader_ls,
        criterion,
        optimizer,
        num_epochs=1000,
        name="simpel_embert",
    )
