
import numpy as np
from sklearn import datasets
#
# mnist = datasets.fetch_mldata('MNIST original', data_home='.')
#
# data = mnist["data"]
# print(data.size)
# data = data[0:5]
# target = mnist["target"]
# data = data[0:5]
#
# print(data[1])
# class_number = np.max(target) + 1
# one_hot = np.eye(class_number)[target.tolist()] #one_hot_vector of target(use list instead of ndarray for index)
# print(one_hot[1])
#
# def softmax(z):
#     e_z = np.exp(z - np.max(z))
#     out = e_z / e_z.sum()
#     return out #P
#
# class LogisticRegression(object):
#
#     def __init__(self, lr, input, label, n_in, n_out, num):
#         self.lr = lr
#         self.x = input
#         self.y = label
#         self.W = np.random.rand(n_in,n_out)  # initialize W
#         self.num = num
#
#         # self.params = [self.W, self.b]
#
#     def train(self):
#         print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
#         for i in range(self.num):
#             P = softmax(numpy.dot(self.x, self.W))
#             T = one_hot
#             self.W -= self.lr * numpy.dot(self.x.T, P-T)
#         print(W)
#
#     def predict(self, x):
#         return softmax(numpy.dot(x, self.W))
#
# # NewLearn = LogisticRegression(0.1, data, one_hot, data.size, one_hot.size, 10)
# # NewLearn.train()
# # print(NewLearn.predict(data[1:10]))
