import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv2.cv as cv

# Takes in rough annoation file with specific frame numbers
# and converts it into sequence of 0s and 1s.
# 0 indicates the absence of an attribute while 1 indicates the presence.

infile_name = 'video06_numpy_rough.npy'
outfile_name = 'video06_numpy_refined.npy'
outfile_text_name = 'video06_text_output_refined.txt'

array_rough = np.load(infile_name, mmap_mode = 'r')
n_attributes, length = array_rough.shape

s = (n_attributes, length)
refined = np.zeros(s)
array_rough = array_rough.astype(int)

for i in range(0, n_attributes):
	if array_rough[i, 0] != 0:
		isPresent = False
		j = 0
		for k in range(0, array_rough[i, j]):
			refined[i, k] = 0

		while array_rough[i, j + 1] != 0:
			prevFrame = array_rough[i, j]
			nextFrame = array_rough[i, j + 1]
			isPresent = not isPresent

			for k in range(prevFrame, nextFrame):
				if isPresent:
					refined[i, k] = 1
				else:
					refined[i, k] = 0
			j+=1
			
		isPresent = not isPresent

		if array_rough[i, j] < length:
			for k in range(array_rough[i, j], length):
				if isPresent:
					refined[i, k] = 1
				else:
					refined[i, k] = 0

np.save(outfile_name, refined)
np.savetxt(outfile_text_name, refined)