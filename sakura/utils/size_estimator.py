"""
PyTorch Model Memory Estimator

This module provides tools for estimating the memory footprint of neural network models,
including both parameter storage and activation memory during forward/backward passes.
"""

import numpy as np
import torch
from torch.autograd import Variable


class SizeEstimator(object):
    """
    Estimates memory consumption of PyTorch models

    Calculates:
        - Parameter storage requirements
        - Activation memory for forward pass
        - Gradient memory for backward pass
        - Input tensor memory

    :param model: Model to analyze
    :type model: torch.nn.Module
    :param input_size: Input dimensions (batch, channels, height, width)
    :type input_size: tuple
    :param bits: Bit precision for memory calculations, defaults to 32
    :type bits: int
    """

    def __init__(self, model, input_size=(1, 1, 32, 32), bits=32):
        self.model = model
        self.input_size = input_size
        self.bits = bits
        return

    def get_parameter_sizes(self):
        """
        Collect dimensions of all parameters in the model.

        :return: None
        """

        mods = list(self.model.modules())
        sizes = []

        for i in range(1, len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        self.param_sizes = sizes
        return

    def get_output_sizes(self):
        """
        Determine output dimensions for each layer by running a sample input through the model.

        :return: None
        """

        input_ = Variable(torch.FloatTensor(*self.input_size), volatile=True)
        mods = list(self.model.modules())
        out_sizes = []
        for i in range(1, len(mods)):
            m = mods[i]
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

        self.out_sizes = out_sizes
        return

    def calc_param_bits(self):
        """
        Calculate total bits required for parameter storage.

        :return: None
        """

        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        self.param_bits = total_bits
        return

    def calc_forward_backward_bits(self):
        """
        Calculate bits needed for activation storage during forward/backward passes.

        :return: None
        """

        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = (total_bits * 2)
        return

    def calc_input_bits(self):
        """
        Calculate bits required for input tensor storage.

        :return: None
        """
        self.input_bits = np.prod(np.array(self.input_size)) * self.bits
        return

    def estimate_size(self):
        """
        Calculate total memory requirements.

        :return: Memory requirement estimation in megabytes and bits
        :rtype: tuple (float, float)
        """

        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits + self.input_bits

        total_megabytes = (total / 8) / (1024 ** 2)
        return total_megabytes, total