from numpy import array, zeros, argmin, inf
from numpy.linalg import norm
import sys
import numpy as np
import os
from matplotlib.pylab import *

#substantial code help from Colin Lea,
#adapted by Joon Hyuck Choi.

reference_file_name = 'video07_numpy_refined_trimmed.npy'
input_file_name = 'video06_numpy_refined_trimmed.npy'

cost_outfile_name = 'video07_06_dtw_cost.npy'
cost_outfile_text_name = 'video07_06_dtw_cost.txt'

path_outfile_name = 'video07_06_dtw_path.npy'
path_outfile_text_name = 'video07_06_dtw_path.txt'

in_vid = 0
ref_vid = 0

def _traceback(D):
    n_ref, n_in = D.shape
    correspondences = np.zeros(n_in, np.int)
    correspondences[-1] = n_ref-1

    c = n_ref-1
    i = n_in-1
    while i > 0 and c > 0:
        a = np.argmin([
                       D[c-1, i-1],
                       D[c-1, i],
                       D[c, i-1]
                       ])
        if a==0 or a==1:
            c -= 1
        if a==0 or a==2:
            i -= 1
        correspondences[i] = c

        # print a

    return correspondences

def dtw(ref_array, input_array):
    # Cols are for reference, Rows are for input
    n_ref, n_in = len(ref_array[0]), len(input_array[0])

    # Initialize distances
    D = np.zeros((n_ref, n_in)) - inf

    # Initialize first row
    D[0,:] = np.cumsum((ref_array[:, 0][:,None]!=input_array).sum(0))

    # Compute DTW
    for ref_ in range(1,n_ref):
        # Precomputed pairwise distances from step i to all j
        D[ref_,:] = (ref_array[:, ref_][:,None]!=input_array).sum(0)
        for in_ in range(1,n_in):
            # Pick best previous step
            D[ref_, in_] += min(
                                D[ref_-1, in_-1],
                                D[ref_-1, in_],
                                D[ref_, in_-1]
                                )

    correspondences = _traceback(D)

    # Viz
    # if 0:
        # Highlight path


    #Visualize: save figures
    D_ = D.copy()

    D_[correspondences, np.arange(D.shape[1])] = D.max()
    subplot(1,1,1); imshow(D_); axis("tight");
    # show()
    savefig("/home/james/Desktop/JamesResearch/EndoscopicVideos/" + "video0" + str(in_vid) + "_0" + str(ref_vid) + "_distances_wo_pancreas.jpg")

    subplot(2,1,1); imshow(ref_array[:,correspondences], interpolation="nearest"); axis("tight");
    # subplot(2,1,1); imshow(ref_array); axis("tight");
    subplot(2,1,2); imshow(input_array, interpolation="nearest"); axis("tight");
    # show()
    savefig("/home/james/Desktop/JamesResearch/EndoscopicVideos/" + "video0" + str(in_vid) + "_0" + str(ref_vid) + "_dtw_wo_pancreas.jpg")

    dist = D[-1, -1] / sum(D.shape)
    return dist, D, correspondences

def main(argv):

    print "program started"

    global in_vid, ref_vid

    for input_video_num in range(1, 8):
        for ref_video_num in range(1, 8):

            in_vid = input_video_num
            ref_vid = ref_video_num

            input_array = np.load('video0' + str(input_video_num) + '_numpy_refined_trimmed.npy')
            ref_array = np.load('video0' + str(ref_video_num) + '_numpy_refined_trimmed.npy')

            input_array = input_array[:,::1]
            input_array[4] = 0
            ref_array = ref_array[:,::1]
            ref_array[4] = 0

            dist = 0
            _dist, _cost, _path = dtw(ref_array, input_array)
            dist += _dist

            print "input: " + str(input_video_num) + ", ref: " + str(ref_video_num) + " => dist: " + str(dist)
            # print _cost
            # print _path

            np.save('video0' + str(input_video_num) + '_0' + str(ref_video_num) + '_dtw_path_wo_pancreas.npy', _path)
            np.savetxt('video0' + str(input_video_num) + '_0' + str(ref_video_num) + '_dtw_path_wo_pancreas.txt', _path)
        print ""

    print "program finished"

if __name__ == "__main__":
    main(sys.argv)
