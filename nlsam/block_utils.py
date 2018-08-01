from __future__ import division

import numpy as np
from nlsam.utils import _im2col3D_overlap, _im2col3D, _im2col4D, _col2im3D_overlap, _col2im3D, _col2im4D


def im2col_nd(A, block_shape, overlap, dtype=np.float64):
    """
    Returns a 2d array of shape flat(block_shape) by A.shape/block_shape made
    from blocks of a nd array.
    """

    block_shape = np.array(block_shape, dtype=np.int32)
    overlap = np.array(overlap, dtype=np.int32)

    if (overlap.any() < 0) or ((block_shape < overlap).any()):
        raise ValueError('Invalid overlap value, it must lie between 0' +
                         'and min(block_size)-1', overlap, block_shape)
    A = padding(A, block_shape, overlap)
    A = np.asarray(A, dtype=dtype)

    if len(A.shape) != len(block_shape):
        raise ValueError("Number of dimensions mismatch!", A.shape, block_shape)

    dim0 = np.prod(block_shape)
    dim1 = np.prod(A.shape - block_shape + 1)
    R = np.zeros((dim0, dim1), dtype=dtype)

    # if A is zeros, R is also gonna be zeros
    if not np.any(A):
        return R

    if len(A.shape) == 3:
        if np.sum(block_shape - overlap) > len(A.shape):
            _im2col3D_overlap(A, R, block_shape, overlap)
        else:
            _im2col3D(A, R, block_shape)
    elif len(A.shape) == 4:
            _im2col4D(A, R, block_shape)
    else:
        raise ValueError("3D or 4D supported only!", A.shape)

    return R


def col2im_nd(R, block_shape, end_shape, overlap, weights=None, dtype=np.float64):
    """
    Returns a nd array of shape end_shape from a 2D array made of flatenned
    block that had originally a shape of block_shape.
    Inverse function of im2col_nd.
    """

    block_shape = np.array(block_shape, dtype=np.int32)
    end_shape = np.array(end_shape, dtype=np.int32)
    overlap = np.array(overlap, dtype=np.int32)

    if (overlap.any() < 0) or ((block_shape < overlap).any()):
        raise ValueError('Invalid overlap value, it must lie between 0 \
                         \nand min(block_size)-1', overlap, block_shape)

    if weights is None:
        weights = np.ones(R.shape[1], dtype=dtype)
    else:
        weights = np.asarray(weights, dtype=dtype)

    R = np.asarray(R, dtype=np.float64)
    A = np.zeros(end_shape, dtype=dtype)
    div = np.zeros(end_shape[:3], dtype=dtype)

    # if R is zeros, A is also gonna be zeros
    if not np.any(R):
        return A

    if len(A.shape) == 3:
        block_shape = block_shape[:3]
        overlap = overlap[:3]

        if np.sum(block_shape - overlap) > len(A.shape):
            _col2im3D_overlap(A, div, R, weights, block_shape, overlap)
        else:
            _col2im3D(A, div, R, weights, block_shape)
    elif len(A.shape) == 4:
        _col2im4D(A, div, R, weights, block_shape)
        div = div[..., None]
    else:
        raise ValueError("3D or 4D supported only!", A.shape)

    return A / div


def padding(A, block_shape, overlap, dtype=np.float64):
    """
    Pad A at the end so that block_shape will cut an integer number of blocks
    across all dimensions. A is padded with 0s.
    """

    block_shape = np.array(block_shape)
    overlap = np.array(overlap)
    fit = ((A.shape - block_shape) % (block_shape - overlap))
    fit = np.where(fit > 0, block_shape - overlap - fit, fit)

    if len(fit) > 3:
        fit[3:] = 0

    if np.sum(fit) == 0:
        return A

    padding = np.array(A.shape) + fit
    padded = np.zeros(padding, dtype=dtype)
    padded[:A.shape[0], :A.shape[1], :A.shape[2], ...] = A

    return padded
