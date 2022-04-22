import numpy as np

a= np.arange(9).reshape((3,3))
print(a)
print(a.shape[1])
print(a[:,1:a.shape[1]])
print(np.power(a[:,1:a.shape[1]],2))