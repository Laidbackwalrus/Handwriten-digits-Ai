import numpy as np


arr1 = [
    [1,2,3],
    [4,5,6],
    [7,8,9],
]

arr2 = np.array([[1,2,3],[4,5,6],[7,8,9]])

#print(arr1.shape)
print(arr2.shape)

print(np.matmul(arr1,arr2))

print(np.transpose(arr2))