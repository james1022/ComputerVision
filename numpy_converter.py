import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv2.cv as cv

infile_name = 'video01_numpy_rough.npy'
outfile_name = 'video01_numpy_refined.npy'
outfile_text_name = 'video01_text_output_refined.txt'
outfile_name_trimmed = 'video01_numpy_refined_trimmed.npy'
outfile_text_name_trimmed = 'video01_text_output_refined_trimmed.txt'

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

newLen = length / 25
s2 = (n_attributes, newLen)
refined2 = np.zeros(s2)

for i in range(0, n_attributes):
	count = 1
	for j in range(0, newLen):
		if count * 25 <= length:
			refined2[i, j] = refined[i, count * 25]
			count += 1

np.save(outfile_name, refined)
np.savetxt(outfile_text_name, refined)
np.save(outfile_name_trimmed, refined2)
np.savetxt(outfile_text_name_trimmed, refined2)