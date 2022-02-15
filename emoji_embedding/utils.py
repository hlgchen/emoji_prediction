import numpy as np
from pathlib import Path


def model_summary(model):
    """Prints shapes of weights in each layer of the model.
    Prints total number of weights in the end."""
    total = 0
    for name, params in model.named_parameters():
        print(name, params.size())
        total += np.prod([x for x in params.size()])
    print("total num_params", total)


def get_project_root():
    """Returns absolute path of project root."""
    return Path(__file__).parent.parent
