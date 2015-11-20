# All code from line 4 should be copied and pasted
# and run on an ipython terminal.

import numpy as np
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import matplotlib.pyplot as plt

infile_name = 'video06_numpy_refined.npy'

X = np.load(infile_name)
n_attributes, length = X.shape

X_colored = np.zeros([n_attributes,length, 3])

for i in range(10):
    color = cm.Dark2(i / 10.)[:3]
    X_colored[i,:] = X[i,:,None]*color


pylab.imshow(X_colored, interpolation="nearest")
pylab.axis("tight")
pylab.show()
