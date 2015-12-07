import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pylab import *

infile_name = 'video07_numpy_refined.npy'
outfile_name = 'video07_plot.jpg'

X = np.load(infile_name)
n_attributes, length = X.shape

X_colored = np.zeros([n_attributes,length, 3])

for i in range(10):
    color = cm.Dark2(i / 10.)[:3]
    X_colored[i,:] = X[i,:,None]*color


pylab.imshow(X_colored, interpolation="nearest")
pylab.axis("tight")
savefig("/home/james/Desktop/JamesResearch/EndoscopicVideos/" + outfile_name)
# pylab.show()
