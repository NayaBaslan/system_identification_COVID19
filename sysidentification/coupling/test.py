import numpy as np
import pdb
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(a)

#array([[1, 2, 3],
#       [4, 5, 6],
#       [7, 8, 9]])

print(a[np.triu_indices(3)])
#or
list(a[np.triu_indices(3)])

pdb.set_trace()