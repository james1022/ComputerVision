from numpy import array, zeros, argmin, inf
from numpy.linalg import norm
import sys
import numpy as np
import cv2
import os
import cv2.cv as cv

reference_file_name = 'video02_numpy_refined.npy'
compared_file_name = 'video06_numpy_refined.npy'

    #""" Computes the DTW of two sequences.
    #:param array x: N1*M array
    #:param array y: N2*M array
    #:param func dist: distance used as cost measure (default L1 norm)
    #Returns the minimum distance, the accumulated cost matrix and the wrap path.
#code from dtw.py of python dtw module

def dtw(x, y, dist=lambda x, y: norm(x - y, ord=1)):

    x = array(x)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    y = array(y)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    r, c = len(x), len(y)

    D = np.zeros((r + 1, c + 1))
    D[0, 1:] = inf
    D[1:, 0] = inf

    for i in range(r):
        for j in range(c):
            D[i+1, j+1] = dist(x[i], y[j])

    for i in range(r):
        for j in range(c):
            D[i+1, j+1] += min(D[i, j], D[i, j+1], D[i+1, j])

    D = D[1:, 1:]

    dist = D[-1, -1] / sum(D.shape)

    return dist, D, _trackeback(D)


def _trackeback(D):
    i, j = array(D.shape) - 1
    p, q = [i], [j]
    while (i > 0 and j > 0):
        tb = argmin((D[i-1, j-1], D[i-1, j], D[i, j-1]))

        if (tb == 0):
            i = i - 1
            j = j - 1
        elif (tb == 1):
            i = i - 1
        elif (tb == 2):
            j = j - 1

        p.insert(0, i)
        q.insert(0, j)

    p.insert(0, 0)
    q.insert(0, 0)
    return (array(p), array(q))

def main(argv):
    a1 = np.load(reference_file_name)
    a2 = np.load(compared_file_name)

    #just testing with the 0th row for now.
    #Later, we can use a for loop to iterate through each row (attribute)
    dist, cost, path = dtw(a1[0], a2[0])
    print dist


if __name__ == "__main__":
    main(sys.argv)
