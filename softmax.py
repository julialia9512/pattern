
import numpy as np
from sklearn import datasets

mnist = datasets.fetch_mldata('MNIST original', data_home='.')

# print(mnist["data"].shape)
learningdata = mnist["data"][:60000]/255.0
predictdata = mnist["data"][60000:]/255.0
# print(learningdata.shape)
# print(predictdata.shape)
target = mnist["target"][:60000]
checktarget = mnist["target"][60000:]
class_number = np.max(target) + 1
one_hot = np.eye(class_number)[target.tolist()] #one_hot_vector of target(use list instead of ndarray for index)

def softmax(x):
    e_x = np.exp(x - np.max(x))
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
            P = softmax(np.dot(self.x, self.W))
            T = one_hot
            self.W -= self.lr * np.matmul(self.x.T, P-T)

    def predict(self, x):
        return softmax(np.matmul(x, self.W))

    def check(self, predicted, label):
        correct = 0
        rate = 0
        for i in range(predicted.shape[0]):
            print(predicted[i])
            print(predicted[i].argmax())
            print(label[i])
            if predicted[i].argmax() == label[i]:
                correct += 1
        rate = correct/predicted.shape[0]
        print(rate)
        return rate


NewLearn = LogisticRegression(0.00005, learningdata, one_hot, learningdata.shape[1], one_hot.shape[1], 100)
NewLearn.train()
predicted = NewLearn.predict(predictdata)
NewLearn.check(predicted, checktarget)
