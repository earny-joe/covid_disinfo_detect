import os
import random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from config import SEED_VALUE, EMBED_MODEL_NAME


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
    print('Set seed value.\n')


def create_embedding_model(model_name=EMBED_MODEL_NAME):
    """
    Given string of pretrain embedding available in sentence-transformers
    library, create a SentenceTransformer object to encode embeddings with
    """
    model = SentenceTransformer(model_name)
    print(f'Loaded {model_name}.\n')
    return model
