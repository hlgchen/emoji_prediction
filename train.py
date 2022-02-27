import os
from time import time
import torch
import torch.nn as nn
import copy
from tqdm import tqdm

from pprint import pprint
from emoji_embedding.utils import model_summary
from embert import EmbertLoss, SimpleEmbert, Accuracy
from twemoji.twemoji_dataset import TwemojiData


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IDX = list(range(1711))
TEST_IDX = list(range(1810))


def get_project_root():
    """Returns absolute path of project root."""
    return os.path.dirname(os.path.abspath(__file__))


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs,
    name,
    save_every=2,
):
    """
    Trains model with specified setup. Models are saved every 2 epoch.
    Returns the model with best evaluation performance.

    Params:
        - model {torch.nn.Module}: model to be trained. For the task at hand this is expected to be
                                    a vision classification model that predicts logits for each class
                                    of the image.
        - dataloaders {dict}: dictionary containing the dataloaders. By default it is expected
                                that this dictionary has "train" and "valid" as keys,
                                with data loaders (containing the images) for the respective phases
        - criterion {torch loss function}: for the task at hand, this should be cross entropy loss
        - optimizer {torch optimizer}: optimizer for training, i.e. adam
        - num_epochs {int}: number of epochs to train.

    Returns:
        - model {torch.nn.Module}: best model according to validation performance
        - val_acc_history {list}: validation accuracies
    """
    start_time_training = time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0
    acc = Accuracy()

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

            epoch_loss = running_loss / (i + 1)
            epoch_accuracy = running_accuracy / (i + 1)

            print(
                "{} Loss: {:.4f}, Accuracy: {:.4f}, took {}".format(
                    phase, epoch_loss, epoch_accuracy, time() - start_time_batch
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
        if epoch % save_every == 0:
            base = os.path.join(get_project_root(), f"trained_models/run1/")
            if not os.path.exists(base):
                os.makedirs(base)
            torch.save(
                model.state_dict(),
                os.path.join(
                    base,
                    f"{name}_epoch{epoch}.ckpt",
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

    pretrained_path = None
    model = SimpleEmbert()
    model.train()
    model = model.to(device)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    # pprint(model_summary(model))

    train_data = TwemojiData("train", shuffle=True)
    valid_data = TwemojiData("valid", shuffle=True)
    dataloaders = {"train": train_data, "valid": valid_data}

    criterion = EmbertLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        num_epochs=1000,
        name="simpel_embert",
        save_every=2,
    )
