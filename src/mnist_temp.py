'''
Author: 彭瑶
Date: 2019/10/18
Description: 手动实现全连接前馈神经网络用于mnist手写数字识别
'''


import numpy as np
import pickle


class NeuralNetwork:
    '''前馈网络'''
    def __init__(self, inode: int, hnode: int, onode: int):
        '''创建包含一个隐层的全连接网络'''
        # self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        # self.relu = lambda x: np.where(x < 0, 0, x)

        self.h_weight = np.random.normal(0.0, pow(hnode, -0.5), (inode, hnode))
        self.h_bias = np.random.normal(0.0, pow(hnode, -0.5), (1, hnode))
        self.o_weight = np.random.normal(0.0, pow(onode, -0.5), (hnode, onode))
        self.o_bias = np.random.normal(0.0, pow(onode, -0.5), (1, onode))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        temp = np.exp(x)
        return temp / np.sum(temp, axis=1, keepdims=True)

    def cross_entropy(self, p, q):
        return -p * np.log(q + 1e-6) - (1 - p) * np.log(1 - q + 1e-6)

    def train(self, inputs: list, labels: list):
        '''训练网络

        Args:
            inputs: 输入数据
            labels: 数据标签
        '''
        inputs = np.array(inputs, ndmin=2)
        labels = np.array([[0 if i != label else 1 for i in range(10)] 
        for label in labels], dtype=np.int8, ndmin=2)

        h_outputs = self.sigmoid(np.dot(inputs, self.h_weight) + self.h_bias)
        outputs = self.softmax(np.dot(h_outputs, self.o_weight) + self.o_bias)
        loss = np.sum(self.cross_entropy(labels, outputs))
        
        delta = outputs - labels
        delta_o_bias = np.sum(delta, axis=0, keepdims=True)
        delta_o_weight = np.dot(h_outputs.T, delta)
        delta = np.dot(delta, self.o_weight.T) * h_outputs * (1 - h_outputs)
        delta_h_bias = np.sum(delta, axis=0, keepdims=True)
        delta_h_weight = np.dot(inputs.T, delta)

        sum_delta = np.sum(np.abs(delta_o_bias))
        sum_delta += np.sum(np.abs(delta_o_weight))
        sum_delta += np.sum(np.abs(delta_h_bias))
        sum_delta += np.sum(np.abs(delta_h_weight))

        l_rate = loss / sum_delta

        self.o_bias -= l_rate * delta_o_bias
        self.o_weight -= l_rate * delta_o_weight
        self.h_bias -= l_rate * delta_h_bias
        self.h_weight -= l_rate * delta_h_weight
        return loss

    # def query(self, inputs: list) -> list:
    #     '''识别数字

    #     传入数据，返回识别结果

    #     Args: 
    #         input: 传入的数据

    #     Returns:
    #         返回识别的结果
    #     '''
    #     inputs = np.array(inputs, ndmin=2)
    #     # outputs = self.relu(np.dot(inputs, self.h_weight) + self.h_bias)
    #     outputs = self.sigmoid(np.dot(inputs, self.h_weight) + self.h_bias)
    #     outputs = self.softmax(np.dot(outputs, self.o_weight) + self.o_bias)
    #     return outputs.tolist()

    def predict(self, inputs):
        inputs = np.array(inputs, ndmin=2)
        outputs = self.sigmoid(np.dot(inputs, self.h_weight) + self.h_bias)
        outputs = self.softmax(np.dot(outputs, self.o_weight) + self.o_bias)
        return outputs.tolist()

    def get_correct_ratio(self, inputs, labels):
        outputs = self.predict(inputs)
        correct_count = 0
        for output, label in zip(outputs, labels):
            if np.argmax(output) == label:
                correct_count += 1
        return correct_count / len(labels)

    def get_data(self, path):
        '''获取mnist数据集中的数据'''
        file = open(path, 'r')
        raw_data = file.readlines()
        file.close()
        data = [[int(i) for i in d.split(',')] for d in raw_data]
        inputs = [[i / 255 for i in dat[1:]] for dat in data]
        labels = [dat[0] for dat in data]
        return inputs, labels


def main():
    # init
    try:
        with open('E:/DataSet/mnist/model.dat', 'rb') as model:
            nn = pickle.load(model)
            inputs, labels = nn.get_data('E:/DataSet/mnist/my_test.csv')
            print('correct ration:', nn.get_correct_ratio(inputs, labels))
    except Exception:
        nn = NeuralNetwork(inode=784, hnode=150, onode=10)
        # train
        inputs, labels = nn.get_data('E:/DataSet/mnist/mnist_train.csv')
        test_inputs, test_labels = nn.get_data('E:/DataSet/mnist/mnist_test.csv')
        for batch_size in (1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
            print('batch size:', batch_size)
            for epoch in range(10):
                print('epoch: ', epoch)
                loss = 0.
                for batch in range(0, len(inputs) // batch_size):
                    beg, end = batch * batch_size, batch * batch_size + batch_size
                    loss += nn.train(inputs[beg: end], labels[beg: end]) / batch_size
                print('loss:', loss / (len(inputs) // batch_size))
                print('correct ration:', nn.get_correct_ratio(test_inputs, test_labels))
        print('correct ration: ', nn.get_correct_ratio(inputs, labels))

        # save model
        with open('E:/DataSet/mnist/model.dat', 'wb') as file:
            pickle.dump(nn, file)
        
    
    # # train
    # inputs, labels = nn.get_data('E:/DataSet/mnist/mnist_train.csv')
    # test_inputs, test_labels = nn.get_data('E:/DataSet/mnist/mnist_test.csv')
    # for batch_size in (13, 17, 100):
    #     print('batch size:', batch_size)
    #     for epoch in range(2):
    #         print('epoch: ', epoch)
    #         loss = 0.
    #         for batch in range(0, len(inputs) // batch_size):
    #             beg, end = batch * batch_size, batch * batch_size + batch_size
    #             loss += nn.train(inputs[beg: end], labels[beg: end]) / batch_size
    #         print('loss:', loss / (len(inputs) // batch_size))
    #         print('correct ration:', nn.get_correct_ratio(test_inputs, test_labels))

    # test_inputs, test_labels = nn.get_data('E:/DataSet/mnist/mnist_test.csv')
    # print('correct ration:', nn.get_correct_ratio(test_inputs, test_labels))


if __name__ == '__main__':
    main()