"""
    (to be implemented)
"""

import torch

class Mapper(torch.nn.Module):
    """
    To perform constrained visualization.
    A light autoencoder focusing on generating main latent space embeddings.
    """

    def __init__(self):
        super(Mapper, self).__init__()
