#cython: wraparound=False, cdivision=True, boundscheck=False

from __future__ import division

cimport cython

ctypedef fused my_fused_type:
    cython.integral # short, int, long
    cython.floating
    cython.bint
    char
    unsigned char

cpdef void _im2col3D(my_fused_type[:,:,:] A, my_fused_type[:,:] R, int[:] size) nogil:

    cdef:
        int k = 0, l = 0
        int a, b, c, m, n, o
        int x = A.shape[0], y = A.shape[1], z = A.shape[2]
        int s0 = size[0], s1 = size[1], s2 = size[2]

    with nogil:
        for a in range(x - s0 + 1):
            for b in range(y - s1 + 1):
                for c in range(z - s2 + 1):

                    l = 0

                    for m in range(s0):
                        for n in range(s1):
                            for o in range(s2):
                                R[l, k] = A[a+m, b+n, c+o]
                                l += 1
                    k += 1


cpdef void _im2col3D_overlap(my_fused_type[:,:,:] A, my_fused_type[:,:] R, int[:] size, int[:] overlap):

    cdef:
        int k = 0, l = 0
        int a, b, c, m, n, o
        int x = A.shape[0], y = A.shape[1], z = A.shape[2]
        int s0 = size[0], s1 = size[1], s2 = size[2]
        int over0 = overlap[0], over1 = overlap[1], over2 = overlap[2]


    for a in range(0, x - over0, s0 - over0):
        for b in range(0, y - over1, s1 - over1):
            for c in range(0, z - over2, s2 - over2):
                with nogil:
                    l = 0

                    for m in range(s0):
                        for n in range(s1):
                            for o in range(s2):
                                R[l, k] = A[a+m, b+n, c+o]
                                l += 1

                    k += 1


cpdef void _im2col4D(my_fused_type[:,:,:,:] A, my_fused_type[:,:] R, int[:] size) nogil:

    cdef:
        int k = 0, l = 0
        int a, b, c, d, m, n, o, p
        int x = A.shape[0], y = A.shape[1], z = A.shape[2], t = A.shape[3]
        int s0 = size[0], s1 = size[1], s2 = size[2]


    for a in range(x - s0 + 1):
        for b in range(y - s1 + 1):
            for c in range(z - s2 + 1):
                with nogil:
                    l = 0

                    for m in range(s0):
                        for n in range(s1):
                            for o in range(s2):
                                for p in range(t):

                                    R[l, k] = A[a+m, b+n, c+o, p]
                                    l += 1

                    k += 1


cpdef void _col2im3D_overlap(my_fused_type[:,:,:] A, my_fused_type[:,:,:] div, my_fused_type[:,:] R,
                            my_fused_type[:] weights, int[:] block_shape, int[:] overlap):
    cdef:
        int k = 0, l = 0
        int a, b, c, m, n, o
        int x = A.shape[0], y = A.shape[1], z =  A.shape[2]
        int s0 = block_shape[0], s1 = block_shape[1], s2 = block_shape[2]
        int over0 = overlap[0], over1 = overlap[1], over2 = overlap[2]

    for a in range(0, x - over0, s0 - over0):
        for b in range(0, y - over1, s1 - over1):
            for c in range(0, z - over2, s2 - over2):
                with nogil:
                    l = 0

                    for m in range(s0):
                        for n in range(s1):
                            for o in range(s2):

                                A[a+m, b+n, c+o] += R[l, k] * weights[k]
                                div[a+m, b+n, c+o] += weights[k]
                                l += 1
                    k += 1


cpdef void _col2im3D(my_fused_type[:,:,:] A, my_fused_type[:,:,:] div, my_fused_type[:,:] R, my_fused_type[:] weights, int[:] block_shape) nogil:

    cdef:
        int k = 0, l = 0
        int a, b, c, m, n, o
        int x = A.shape[0], y = A.shape[1], z =  A.shape[2]
        int s0 = block_shape[0], s1 = block_shape[1], s2 = block_shape[2]

    with nogil:
        for a in range(x - s0 + 1):
            for b in range(y - s1 + 1):
                for c in range(z - s2 + 1):

                    l = 0

                    for m in range(s0):
                        for n in range(s1):
                            for o in range(s2):

                                A[a+m, b+n, c+o] += R[l, k] * weights[k]
                                div[a+m, b+n, c+o] += weights[k]
                                l += 1
                    k += 1


cpdef void _col2im4D(my_fused_type[:,:,:,:] A, my_fused_type[:,:,:] div, my_fused_type[:,:] R, my_fused_type[:] weights, int[:] block_shape) nogil:
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
