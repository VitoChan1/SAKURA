"""
    (to be implemented)
"""

import torch

class Infuser(torch.nn.Module):
    """
    To imbue external information into embeddings.
    Only main latent space will be generated.
    """

    def __init__(self):
        super(Regularizer, self).__init__()
        raise NotImplementedError
