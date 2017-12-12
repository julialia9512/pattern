
import numpy as np
from sklearn import datasets

mnist = datasets.fetch_mldata('MNIST original', data_home='.')

# print(mnist["data"].shape)
learningdata = mnist["data"][:60000]
predictdata = mnist["data"][60000:]
# print(learningdata.shape)
# print(predictdata.shape)
target = mnist["target"][:60000]
checktarget = mnist["target"][60000:]
class_number = np.max(target) + 1
one_hot = np.eye(class_number)[target.tolist()] #one_hot_vector of target(use list instead of ndarray for index)

def softmax(z):
    e_z = np.exp(z - np.max(z))
    out = e_z / e_z.sum()
    return out #P

class LogisticRegression:

    def __init__(self, lr, learningdata, label, n_in, n_out, num):
        self.lr = lr
        self.x = learningdata
        self.y = label
        self.W = np.random.rand(784,10)  # initialize W
        self.num = num

        # self.params = [self.W, self.b]

    def train(self):
        print('$$$$$$')
        for i in range(self.num):
            P = softmax(np.dot(self.x, self.W))
            T = one_hot
            self.W -= self.lr * np.dot(self.x.T, P-T)
        print(self.W)

    def predict(self, x):
        return softmax(np.dot(x, self.W))

NewLearn = LogisticRegression(0.1, learningdata, one_hot, learningdata.shape[1], one_hot.shape[1], 10)
NewLearn.train()
print(NewLearn.predict(predictdata))
