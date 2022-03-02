import numpy as np
from pathlib import Path


def model_summary(model, only_trainable=False, verbose=True):
    """Prints shapes of weights in each layer of the model.
    Prints total number of weights in the end."""
    total = 0
    for name, params in model.named_parameters():
        if verbose:
            print(f"{name}: {params.size()}, requires grad: {params.requires_grad}")
        if only_trainable:
            if params.requires_grad:
                total += np.prod([x for x in params.size()])
        else:
            total += np.prod([x for x in params.size()])

    text = "total num trainable params" if only_trainable else "total num_params"
    print(text, total)


def get_project_root():
    """Returns absolute path of project root."""
    return Path(__file__).parent.parent
