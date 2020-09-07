import os
import random
import numpy as np
import torch
from settings.config import SEED_VALUE


def set_seed(seed=SEED_VALUE):
    """
    Set the random seed for generating embeddings
    """
    # Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)
    # set `torch` pseudo-random generator at a fixed value
    torch.manual_seed(seed)
