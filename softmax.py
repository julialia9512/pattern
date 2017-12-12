
import numpy as np
from sklearn import datasets

mnist = datasets.fetch_mldata('MNIST original', data_home='.')

# print(mnist["data"].shape)
learningdata = mnist["data"][:60000]/255.0
print(learningdata[1])
predictdata = mnist["data"][60000:60010]/255.0
# print(learningdata.shape)
# print(predictdata.shape)
target = mnist["target"][:60000]
checktarget = mnist["target"][60000:]
class_number = np.max(target) + 1
one_hot = np.eye(class_number)[target.tolist()] #one_hot_vector of target(use list instead of ndarray for index)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    print(e_x)
    out = e_x / e_x.sum()
    return out

class LogisticRegression:

    def __init__(self, lr, learningdata, label, n_in, n_out, num):
        self.lr = lr
        self.x = learningdata
        self.y = label
        self.W = np.random.rand(784,10)  # initialize W
        self.num = num

    def train(self):
        for i in range(self.num):
            print('$$$$$$')
            print(self.W.shape)
            print(self.x.shape)
            P = softmax(np.matmul(self.x, self.W))
            print(P.shape)
            T = one_hot
            self.W -= self.lr * np.matmul(self.x.T, P-T)
        print(self.W.shape)

    def predict(self, x):
        return softmax(np.matmul(x, self.W))

NewLearn = LogisticRegression(0.1, learningdata, one_hot, learningdata.shape[1], one_hot.shape[1], 10)
NewLearn.train()
print(NewLearn.predict(predictdata))
