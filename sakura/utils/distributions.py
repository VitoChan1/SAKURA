"""
Batch sample generation functions of various distributions
"""

from math import sqrt, sin, cos

import numpy as np
import torch
from sklearn.datasets import make_circles


def swiss_roll(batch_size, n_dim=2, n_labels=10, label_indices=None):
    """
    Generates samples from a Swiss roll manifold with optional label conditioning.

    The Swiss roll is a 2D manifold embedded in 2D space, shaped like a rolled spiral.
    Labels determine which segment of the spiral the samples come from.

    :param batch_size: Number of samples to generate
    :type batch_size: int
    :param n_dim: Dimension of output samples, must be 2
    :type n_dim: Literal[2]
    :param n_labels: Number of distinct label segments in the spiral
    :type n_labels: int
    :param label_indices: List of label indices (0 <= index < n_labels)
        for each batch sample, randomly assigned if none provided
    :type label_indices: list[int], optional

    :return: Tensor of shape (batch_size, n_dim) with Swiss roll samples
    :rtype: torch.FloatTensor
    """
    if n_dim != 2:
        raise Exception("n_dim must be 2.")

    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim/2)):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(np.random.randint(0, n_labels), n_labels)
    return torch.from_numpy(z).type(torch.FloatTensor)

def gaussian_mixture(batch_size, n_dim=2, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None):
    """
    Generates samples from a Gaussian mixture distribution arranged in a circle.

    Each Gaussian component is rotated around a circle and offset from the origin.
    Labels determine which Gaussian component a sample comes from.

    :param batch_size: Number of samples to generate
    :type batch_size: int
    :param n_dim: Dimension of output samples, must be 2
    :type n_dim: Literal[2]
    :param n_labels: Number of Gaussian components arranged around the circle
    :type n_labels: int
    :param x_var: Variance along the x-axis for each component
    :type x_var: float
    :param y_var: Variance along the y-axis for each component
    :type y_var: float
    :param label_indices: List of label indices (0 <= index < n_labels)
            for each batch sample, randomly assigned if none provided
    :type label_indices: list[int], optional

    :return: Tensor of shape (batch_size, n_dim) with uniform samples.
    :rtype: torch.FloatTensor
    """
    if n_dim != 2:
        raise Exception("n_dim must be 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, (int)(n_dim/2)))
    y = np.random.normal(0, y_var, (batch_size, (int)(n_dim/2)))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim/2)):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)
    return torch.from_numpy(z).type(torch.FloatTensor)

def rand_cirlce2d(batch_size):
    """
    Generates 2D samples from a uniform filled-circle distribution in a 2-dimensional space.

    Samples are drawn with uniform radial distance from the origin.

    :param batch_size: Number of samples to generate
    :type batch_size: int

    :return: Tensor of shape (batch_size, 2) with samples
    :rtype: torch.FloatTensor
    """
    r = np.random.uniform(size=(batch_size))
    theta = 2 * np.pi * np.random.uniform(size=(batch_size))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.array([x, y]).T
    return torch.from_numpy(z).type(torch.FloatTensor)


def rand_ring2d(batch_size, n_dim=2):
    """
    Generates 2D samples from a hollowed-circle (ring) distribution using sklearn.make_circles.

    Samples are drawn with uniform radial distance from the origin.

    :param batch_size: Number of samples to generate
    :type batch_size: int
    :param n_dim: Dimension of output samples, should be 2
    :type n_dim: Literal[2]

    :return: Tensor of shape (batch_size, 2) with ring samples
    :rtype: torch.FloatTensor
    """
    if n_dim != 2:
        raise NotImplementedError

    circles = make_circles(2 * batch_size, noise=.01)
    z = np.squeeze(circles[0][np.argwhere(circles[1] == 0), :])
    return torch.from_numpy(z).type(torch.FloatTensor)


def rand_uniform(batch_size, n_dim=2, low = -1., high = 1.,
                 n_labels=1, label_offsets=None, label_indices=None) -> torch.Tensor:
    """
    Generates samples from a uniform distribution with optional label-based offsets.

    When n_labels > 1, applies label-specific offsets to create separated clusters.

    :param batch_size: Number of batch samples
    :type batch_size: int
    :param n_dim: Number of latent dimension, defaults to 2
    :type n_dim: int
    :param low: Lower bound of uniform distribution (for each dimension)
    :type low: float
    :param high: Upper bound of uniform distribution (for each dimension)
    :type high: float
    :param n_labels: Number of labels to consider in supervision, when n_labels=1, supervision is off
    :type n_labels: int
    :param label_offsets: Offsets for each label cluster
    :type label_offsets: list[list[float]], optional
    :param label_indices: List of label indices (0 <= index < n_labels)
        for each batch sample, randomly assigned if none provided
    :type label_indices: list[int], optional

    :return: Tensor of shape (batch_size, n_dim) with uniform samples
    :rtype: torch.FloatTensor
    """
    z = np.random.uniform(size=(batch_size, n_dim), low = low, high = high)

    if n_labels > 1:
        if label_indices is not None:
            idx = np.array(label_indices, dtype=np.integer)
        else:
            idx = np.random.randint(0, n_labels)
        offset = np.array([label_offsets[i] for i in idx]).reshape((batch_size, n_dim))
        z += offset


    return torch.from_numpy(z).type(torch.FloatTensor)


def rand(dim_size):
    """
    Creates a function that generates uniform random samples in [0, 1).

    :param batch_size: Number of batch samples
    :type batch_size: int

    :return: Tensor of shape (batch_size, dim_size) with uniform samples
    :rtype: torch.FloatTensor
    """
    def _rand(batch_size):
        return torch.rand((batch_size, dim_size))
    return _rand


def randn(dim_size):
    """
    Creates a function that generates standard normal random samples.

    :param batch_size: Number of batch samples
    :type batch_size: int

    :return: Tensor of shape (batch_size, dim_size) with uniform samples
    :rtype: torch.FloatTensor
    """
    def _randn(batch_size):
        return torch.randn((batch_size, dim_size))
    return _randn
