#cython: wraparound=False, cdivision=True, boundscheck=False

from __future__ import print_function

import numpy as np
from autodmri.blocks import extract_patches

cimport numpy as np
cimport cython


def im2col_nd(A, block_shape, overlap):
    """
    Returns a 2d array of shape flat(block_shape) by A.shape/block_shape made
    from blocks of a nd array.
    """

    block_shape = np.array(block_shape, dtype=np.int32)
    overlap = np.array(overlap, dtype=np.int32)
    step = block_shape - overlap

    if (overlap.any() < 0) or ((block_shape < overlap).any()):
        raise ValueError('Invalid overlap value, it must lie between 0 and min(block_size)-1', overlap, block_shape)

    A = padding(A, block_shape, step)

    if len(A.shape) != len(block_shape):
        raise ValueError("Number of dimensions mismatch!", A.shape, block_shape)

    out = extract_patches(A, block_shape, step).reshape(-1, np.prod(block_shape)).T
    return out


cdef void _col2im4D(double[::1,:,:,:] A, double[::1,:,:] div, double[::1,:] R, double[:] weights, int[:] block_shape) nogil:
    cdef:
        int k = 0, l = 0
        int a, b, c, d, m, n, o, p
        int x = A.shape[0], y = A.shape[1], z = A.shape[2], t = A.shape[3]
        int s0 = block_shape[0], s1 = block_shape[1], s2 = block_shape[2]

    with nogil:
        for a in range(x - s0 + 1):
            for b in range(y - s1 + 1):
                for c in range(z - s2 + 1):

                    l = 0

                    for m in range(s0):
                        for n in range(s1):
                            for o in range(s2):
                                for p in range(t):
                                    A[a+m, b+n, c+o, p] += R[l, k] * weights[k]
                                    l += 1

                                div[a+m, b+n, c+o] += weights[k]
                    k += 1


def col2im_nd(R, block_shape, end_shape, overlap, weights=None, order='F'):
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
        weights = np.ones(R.shape[1], dtype=np.float64, order=order)
    else:
        weights = np.asarray(weights, dtype=np.float64, order=order)

    R = np.asarray(R, dtype=np.float64, order=order)
    A = np.zeros(end_shape, dtype=np.float64, order=order)
    div = np.zeros(end_shape[:3], dtype=np.float64, order=order)

    # if R is zeros, A is also gonna be zeros
    if not np.any(R):
        return A

    if len(A.shape) == 4:
        _col2im4D(A, div, R, weights, block_shape)
        div = div[..., None]
    else:
        raise ValueError("3D or 4D supported only!", A.shape)

    return A / div


def padding(A, block_shape, overlap, padvalue=0):
    """
    Pad A at the end so that block_shape will cut an integer number of blocks
    across all dimensions. A is padded with padvalue (default is 0).
    """

    block_shape = np.array(block_shape)
    overlap = np.array(overlap)

    step = block_shape - overlap
    fit = np.mod(A.shape, step)

    # fit = ((A.shape - block_shape) % (block_shape - overlap))
    # fit = np.where(fit > 0, block_shape - overlap - fit, fit)

    if len(fit) > 3:
        fit[3:] = 0

    if np.sum(fit) == 0:
        return A

    # Pad on the right only
    fit = [(0, f) for f in fit]
    padded = np.pad(A, fit, 'constant', constant_values=padvalue)

    return padded
