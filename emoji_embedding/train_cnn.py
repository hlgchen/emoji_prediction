import torch
import torch.nn as nn
from time import time
import copy

from torch.utils.data import DataLoader

from emoji_image_dataset import EmojiImageDescriptionDataset
from ee_model import Img2Vec, ContrastiveLoss
import utils
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model, dataloaders, criterion, optimizer, num_epochs, name="semi_siamese"
):
    """
    Trains model with specified setup. Models are saved avery 5 epoch.
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
    cossim = nn.CosineSimilarity()

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
                for i, batch in tbatch:
                    start_time_batch = time()

                    X = batch["image"]
                    X = X.to(device)
                    X = X.view(-1, *X.shape[2:])

                    Xd = batch["description"]
                    Xd = Xd.to(device)
                    Xd = Xd.view(-1, *Xd.shape[2:])

                    y = batch["label"]
                    y = y.to(device)
                    y = y.view(-1, *y.shape[2:]).long()

                    with torch.set_grad_enabled(phase == "train"):
                        optimizer.zero_grad()
                        img_embeddings = model(X)
                        outputs = 1 - cossim(Xd, img_embeddings)
                        loss = criterion(outputs, y)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * y.size(0)
                    tbatch.set_postfix(
                        loss=loss.item() * y.size(0),
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
            base = os.path.join(
                utils.get_project_root(), f"emoji_embedding/model/vision_{name}/"
            )
            if not os.path.exists(base):
                os.makedirs(base)
            torch.save(
                model.state_dict(),
                os.path.join(
                    base,
                    f"res18_{name}_epoch{epoch}.ckpt",
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

    train_data = EmojiImageDescriptionDataset("train", num_neg_sample=9)
    valid_data = EmojiImageDescriptionDataset("valid", num_neg_sample=9)
    train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=32, shuffle=True)
    dataloaders = {"train": train_data_loader, "valid": valid_data_loader}

    model = Img2Vec(emb_dimension=200)
    model.to(device)

    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters())

    train_model(
        model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=42,
    )
