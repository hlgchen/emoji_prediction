import torch
import torch.nn as nn
from time import time
import copy

from torch.utils.data import DataLoader

from emoji_images import EmojiClassificationDataset
from cnn import ResnetExt
import utils
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
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

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            running_corrects = 0

            with tqdm(enumerate(dataloaders[phase])) as tbatch:
                for i, (X, y) in tbatch:
                    start_time_batch = time()

                    X = X.to(device)
                    y = y.to(device).long()

                    with torch.set_grad_enabled(phase == "train"):
                        optimizer.zero_grad()
                        outputs = model(X)
                        loss = criterion(outputs, y.squeeze(-1))
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    _, y_pred = torch.max(outputs, 1)

                    # statistics
                    running_loss += loss.item() * y_pred.size(0)
                    running_corrects += torch.sum(y_pred == y.data)
                    tbatch.set_postfix(
                        loss=loss.item() * y_pred.size(0),
                        running_loss=running_loss / (i + 1),
                    )

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(
                "{} Loss: {:.4f} Acc: {:.4f}, took {}".format(
                    phase, epoch_loss, epoch_acc, time() - start_time_batch
                )
            )

            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "valid":
                val_acc_history.append(epoch_acc)

        time_elapsed = time() - start_time_epoch
        print(
            "Epoch {}/{} FINISHED, took {:.0f}m {:.0f}s".format(
                epoch, num_epochs - 1, time_elapsed // 60, time_elapsed % 60
            )
        )
        print("-" * 10)
        if epoch % 5 == 0:
            base = os.path.join(
                utils.get_project_root(), f"emoji_embedding/model/vision/"
            )
            if not os.path.exists(base):
                os.makedirs(base)
            torch.save(
                model.state_dict(),
                os.path.join(
                    base,
                    f"res18_epoch{epoch}.ckpt",
                ),
            )
            print("model saved")

    time_elapsed = time() - start_time_training
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best valid Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == "__main__":

    train_data = EmojiClassificationDataset("train")
    valid_data = EmojiClassificationDataset("valid")
    train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=64, shuffle=True)
    dataloaders = {"train": train_data_loader, "valid": valid_data_loader}

    model = ResnetExt(1710)
    model.to(device)
    # utils.model_summary(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_model(
        model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=42,
    )
