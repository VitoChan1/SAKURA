"""
Various data transformations utilities
"""

import numpy as np
import pandas as pd
import scipy.sparse
import sklearn
import sklearn.preprocessing as skprep
import torch
from packaging import version


class ToTensor(object):
    """
    Callable class to convert input data to PyTorch Tensors

    Handles input-specific transformations such as transposing gene data or adjusting dimensions.

    For 'gene' input:

        - DataFrames are transposed (genes x samples → samples x genes).
        - Series become 2D tensors (1 sample x genes).
        - Sparse matrices are densified.

    For 'pheno' input, no transpose is applied.

    :param sample: Input data to convert
    :type sample: pd.DataFrame, pd.Series, np.ndarray or scipy.sparse matrix
    :param input_type: Type of input data, can be 'gene' (gene expression)
        or 'pheno' (phenotype), defaults to 'gene'
    :type input_type: Literal['gene','pheno'], optional
    :param force_tensor_type: Force output tensor to a specific data type,
        can be 'float', 'int' or 'double'
    :type force_tensor_type: Literal['float', 'int','double'], optional

    :return: Converted tensor
    :rtype: torch.Tensor
    """

    def __call__(self, sample, input_type='gene', force_tensor_type=None):
        ret = None
        #print(type(sample))
        #print(sample)
        if input_type == 'gene':
            if type(sample) is pd.core.frame.DataFrame:
                ret = torch.from_numpy(sample.astype(float).values).transpose(0, 1).float()
            elif type(sample) is pd.core.series.Series:
                ret = torch.from_numpy(sample.astype(float).values).unsqueeze(0).float()
            elif type(sample) is np.ndarray:
                ret = torch.from_numpy(sample).float()
            elif scipy.sparse.isspmatrix(sample):
                # In case some transformations unwrapped the pd.DataFrame.sparse
                ret = torch.from_numpy(sample.todense()).float()
            else:
                raise NotImplementedError
        elif input_type == 'pheno':
            if type(sample) is pd.core.frame.DataFrame:
                ret = torch.from_numpy(sample.astype(np.float).values)
            elif type(sample) is np.ndarray:
                ret = torch.from_numpy(sample)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        if force_tensor_type is not None:
            if force_tensor_type == 'float':
                ret = ret.float()
            elif force_tensor_type == 'int':
                ret = ret.int()
            elif force_tensor_type == 'double':
                ret = ret.double()
            else:
                raise NotImplementedError('Expected tensor type not supported yet')
        return ret

class ToBinary(object):
    """
    Convert input data into binary form (0 or 1) using sklearn Binarizer

    To handle floating point error, a threshold (epsilon) is applied to check
    if the value should be classified as 0 or 1.

    :param sample: Input data of shape (n_samples, n_features) to binarize
    :type sample: array-like or sparse matrix
    :param threshold: Values > threshold become 1, others 0, defaults to 1e-6
    :type threshold: float, optional
    :param inverse: Whether to invert binary values (1 → 0, 0 → 1), defaults to False
    :type inverse: bool, optional
    :param scale_factor: Multiply final output by this value, defaults to 1.0
    :type scale_factor: float, optional

    :return: Transformed binarized output data
    :rtype: numpy.ndarray or scipy.sparse matrix
    """

    def __call__(self, sample, threshold=1e-6, inverse=False, scale_factor=1.0):
        binarizer = skprep.Binarizer(threshold=threshold).fit(sample)
        ret = binarizer.transform(sample)
        if inverse:
            ret = 1-ret
        ret = ret*scale_factor
        return ret


class ToOnehot(object):
    """
    Callable class to convert categorical labels to one-hot encodings using sklearn.preprocessing OneHotEncoder

    Useful when the loss is not compatible directly with class labels and expected to be used on Phenotype only.

    :param sample: Input data of shape (n_samples, n_features) to determine the categories of each feature
    :param sample: array-like
    :param order: Expected order of categories (unique values per feature), defaults to 'auto', where
        categories are determined automatically from the input data
    :type order: 'auto' or a list of array-like, optional

    :return: Transformed one-hot encoded data
    :rtype: array-like
    """

    def __call__(self, sample, order='auto'):
        # Adaptations
        if order != 'auto':
            if type(order) is list:
                order = [order]

        ohtrs = skprep.OneHotEncoder(categories=order, sparse=False).fit(sample)
        return ohtrs.transform(sample)


class ToOrdinal(object):
    """
    Callable class to convert categorical labels to an integer array
    using sklearn OrdinalEncoder

    Useful for losses like torch.nn.CrossEntropyLoss and expected to be used on Phenotype.

    :param sample: Input data of shape (n_samples, n_features) containing categorical features
    :type sample: array-like
    :param order: Expected order of categories (unique values per feature), defaults to 'auto', where
        categories are determined automatically from the input data
    :type order: 'auto' or a list of array-like, optional
    :param handle_unknown*: Strategy for handling unknown categories, defaults to 'use_encoded_value'
        which sets unknown categories to <unknown_value>
    :type handle_unknown: Literal['error', 'use_encoded_value'], optional
    :param unknown_value: Encoded value to assign unknown categories, must be numerical if using
        'use_encoded_value' strategy, defaults to np.nan
    :type unknown_value: int or np.nan, optional

    .. note::
        **<handle_unknown>:** When set to ‘use_encoded_value’,
        the encoded value of unknown categories will be set to the value given for the parameter;
        When set to ‘error’, an error will be raised in case an unknown categorical feature is
        present during transform.

    :return: Transformed ordinal encoded data
    :rtype: array-like
    """

    def __call__(self, sample, order='auto', handle_unknown='use_encoded_value', unknown_value=np.nan):
        # Adaptations
        if order != 'auto':
            if type(order) is list:
                order = [order]

        # Resolving compatibility of older sklearns
        if version.parse(sklearn.__version__) >= version.parse('0.24'):
            ortrs = skprep.OrdinalEncoder(categories=order, handle_unknown=handle_unknown, unknown_value=unknown_value,
                                          dtype=int).fit(sample)
        else:
            ortrs = skprep.OrdinalEncoder(categories=order, dtype=int).fit(sample)

        return ortrs.transform(sample)


class ToKBins(object):
    """
    Callable class to discretize continuous data into intervals
    using sklearn KBinsDiscretizer with binning strategies

    Useful for preprocessing continuous phenotypes into categorical representations.

    :param sample: Input data of shape (n_samples, n_features) containing continuous features
    :type sample: array-like
    :param n_bins: The number of bins for all features or each feature to produce, defaults to 2 for all
    :type n_bins: int or array-like of shape (n_features,), optional
    :param encode: Method used to encode the transformed result
    :type encode: Literal['ordinal', 'onehot', 'onehot-dense'], optional
    :param strategy*: Strategy used to define the widths of the bins
    :type strategy: Literal['quantile', 'uniform', 'kmeans'], optional

    :return: Transformed K-bins discretized data
    :rtype: numpy.ndarray or scipy.sparse matrix

    .. Note::
        **<strategy>:** 'kmeans' strategy may produce irregular bin widths depending on data distribution.

    Options:

    Encoding method for transformed bins:
        - 'ordinal': Integer representation (0 to n_bins-1)
        - 'onehot': Sparse matrix one-hot encoding
        - 'onehot-dense': Dense array one-hot encoding
    Binning strategy:
        - 'quantile': Equal-frequency bins
        - 'uniform': Equal-width bins
        - 'kmeans': Clustering-based bin edges

    """

    def __call__(self, sample, n_bins=2, encode='ordinal', strategy='quantile'):

        kbintrs = skprep.KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
        return kbintrs.fit_transform(sample)


class StandardNormalizer(object):
    """
    (To be implemented)

    Allow log-transformation (like in Seurat, first multiply with a size factor,
    then plus a pseudocount, then log), and standardization (scaling and centering, to obtain z-score).
    """

    def __call__(self, center=True, scale=True, normalize=True):
        raise NotImplementedError
