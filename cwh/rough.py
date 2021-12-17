import numpy as np
# l = np.array([[1,2,4] , [6,7,8]])
# m = l[:, np.newaxis, :]
# n = l[:, np.newaxis, 2]
# p = m[:,:,2]
# print(m) 
# print(n) 
# print(p) 

X_new = np.linspace(0,3,10).reshape(1,-1)
print(X_new)