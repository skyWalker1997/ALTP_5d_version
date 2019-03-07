# from knn import classify
import numpy as np
#
# data = [[1,1],[0,0],[2,2],[-1,-1]]
# data = np.array((data))
# label = ['a','b','c','d']
# label = np.array((label))
# index = [[0.3,0.3]]
# index = np.array((index))
#
# knn_goe ,distance_slot_index,distance_value = classify(index,data,label,4)
# print(knn_goe ,distance_slot_index,distance_value)
a = np.load('/Users/PINKFLOYD/Desktop/graduatedesign/ALTP_5d_version/Data/driver.npy')
b = np.load('/Users/PINKFLOYD/Desktop/graduatedesign/ALTP_5d_version/Data/speed.npy')
print(a[0][0])
print(b[0][0])