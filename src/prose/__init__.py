# -*- coding: utf-8 -*-


import sys
from typing import Union

import numpy as np
import torch

from .alphabets import Uniprot21
from .models.multitask import ProSEMT
from .models.lstm import SkipLSTM


def embed_sequence(model: Union[ProSEMT, SkipLSTM],
                   sequence: str,
                   pool='none',
                   use_cuda=False):
    """Embed protein sequences

    :param model: Model to be used (ProSEMT or SkipLSTM)
    :param sequence: Sequence to be embedded
    :param pool: Pooling startegy over the sequence ('none', 'sum', 'max', 'avg')
    :param use_cuda: Should CUDA be used
    """
    # No amino acid, then weights are 0s
    if len(sequence) == 0:
        n = model.embedding.proj.weight.size(1)
        z = np.zeros((1,n), dtype=np.float32)
        return z
    # Use the default alphabet
    alphabet = Uniprot21()
    if isinstance(sequence, str):
        sequence = sequence.encode()
    sequence = sequence.upper()
    # Encode the sequence to the alphabet
    sequence = alphabet.encode(sequence)
    sequence = torch.from_numpy(sequence)
    # Move to CUDA if needed
    if use_cuda:
        sequence = sequence.cuda()
        # Move model to CUDA if required
        if not next(model.parameters()).is_cuda:
            model = model.cuda()

    # Embed the sequence
    with torch.no_grad():
        sequence = sequence.long().unsqueeze(0)
        z = model.transform(sequence)
        # Pool if needed
        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z,_ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)
        z = z.cpu().numpy()
    return z