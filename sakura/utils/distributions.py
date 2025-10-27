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
    :type n_dim: Literal[2], optional
    :param n_labels: Number of distinct label segments in the spiral, defaults to 10
    :type n_labels: int, optional
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

def gaussian_mixture(batch_size, n_dim, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None, shift_magnitude=1.4,
    angular_offset=0.0):
    """
    Generates samples from a Gaussian mixture distribution arranged in 2D subspace(s).

    The distribution consists of Gaussian components arranged in circle(s) within multiple 2D subspace(s).
    For even dimensions, each pair of dimensions forms a 2D subspace with rotated Gaussian components.
    For odd dimensions, the last dimension contains independent Gaussian noise.

    :param batch_size: Number of samples to generate
    :type batch_size: int
    :param n_dim: Dimension of output samples
    :type n_dim: Literal[2], optional
    :param n_labels: Number of Gaussian components arranged around the circle, defaults to 10
    :type n_labels: int, optional
    :param x_var: Variance along the x-axis for each component, defaults to 0.5
    :type x_var: float, optional
    :param y_var: Variance along the y-axis for each component, defaults to 0.1
    :type y_var: float, optional
    :param label_indices: List of label indices (0 <= index < n_labels)
            for each batch sample, randomly assigned if none provided
    :type label_indices: list[int], optional
    :param shift_magnitude: Distance from the origin (center) to the center of each Gaussian component, defaults to 1.4
    :type shift_magnitude: float
    :param angular_offset: Angle offset coefficient between different 2D subspaces, defaults to 0.0
    :type angular_offset: float

    :return: Tensor of shape (batch_size, n_dim) with uniform samples.
    :rtype: torch.FloatTensor
    """
    #if n_dim != 2:
    #    raise Exception("n_dim must be 2.")
    if n_dim < 1:
        raise ValueError("n_dim must be positive integer.")

    if label_indices is None:
        labels = np.random.randint(0, n_labels, size=batch_size)
    else:
        labels = np.array(label_indices)
        if len(labels) != batch_size:
            raise ValueError("number of labels must match batch size.")

    samples = np.zeros((batch_size, n_dim), dtype=np.float32)

    n_pairs = n_dim // 2
    for pair_idx in range(n_pairs):
        x = np.random.normal(0, x_var, batch_size)
        y = np.random.normal(0, y_var, batch_size)

        theta = (2 * np.pi / n_labels) * (labels + angular_offset * pair_idx)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_rot = x * cos_theta - y * sin_theta
        y_rot = x * sin_theta + y * cos_theta

        x_rot += shift_magnitude * cos_theta
        y_rot += shift_magnitude * sin_theta

        samples[:, pair_idx * 2] = x_rot
        samples[:, pair_idx * 2 + 1] = y_rot

    # n_dim is odd
    if n_dim % 2:
        samples[:, -1] = np.random.normal(0, x_var, batch_size)

    return torch.from_numpy(samples).type(torch.FloatTensor)


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
    :type n_dim: Literal[2], optional

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
    :type n_dim: int, optional
    :param low: Lower bound of uniform distribution (for each dimension), defaults to -1.
    :type low: float, optional
    :param high: Upper bound of uniform distribution (for each dimension), defaults to 1.
    :type high: float, optional
    :param n_labels: Number of labels to consider in supervision, defaults to 1, where supervision is off
    :type n_labels: int, optional
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
