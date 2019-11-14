'''
Author: 彭瑶
Date: 2019/10/18
Description: 手动实现全连接前馈神经网络用于mnist手写数字识别
'''


import numpy as np
import pickle
import matplotlib.pyplot as plt


def convs(inputs: np.array, filters: np.array) -> np.array:
    '''对图像进行卷积操作，返回卷积后的图像
    
    Args:
        inputs: 输入图像
        filters: 卷积核
    '''
    batch_size, input_w, input_h = inputs.shape[0: 3]
    output_d, size = filters.shape[0: 2]
    output_w, output_h = input_w - size + 1, input_h - size + 1
    outputs = np.zeros((batch_size, output_w, output_h, output_d))
    for k in range(output_d):
        filter = filters[k].reshape((3, 3, 1))
        for i in range(output_w):
            for j in range(output_h):
                outputs[:, i, j, k] += np.sum(inputs[:, i: i + size, 
                j: j + size] * filter, axis=(1, 2, 3))
    return outputs


def pooling(inputs: np.array, size: int, stride: int) -> np.array:
    '''对图像进行池化操作，返回池化后的图像
    
    Args:
        inputs: 输入图像
        size: 池化窗口大小
        stride: 池化步长
    '''
    batch_size, input_w, input_h, input_d = inputs.shape
    output_w = (input_w - size) // stride + 1
    output_h = (input_h - size) // stride + 1
    output_d = input_d
    outputs = np.empty((batch_size, output_w, output_h, output_d))
    for i in range(output_w):
        i_s = i * stride
        for j in range(output_h):
            j_s = j * stride
            outputs[:, i, j] = np.max(inputs[:, i_s: i_s + size, 
            j_s: j_s + size], axis=(1, 2))
    return outputs


def get_data(path):
    '''获取mnist数据集中的数据'''
    file = open(path, 'r')
    raw_data = file.readlines()
    file.close()
    data = [[int(i) for i in d.split(',')] for d in raw_data]
    inputs = [[i / 255 for i in dat[1:]] for dat in data]
    labels = [dat[0] for dat in data]
    return inputs, labels


def extract_feature(inputs):
    filters = np.array([[[1, 1, 1], [0, 0, 0], [-1, -1, -1]], 
    [[1, 0, -1], [1, 0, -1], [1, 0, -1]], [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], 
    [[1, 1, 0], [1, 0, -1], [0, -1, -1]]])
    batch_size = len(inputs)
    inputs = np.reshape(inputs, (batch_size, 28, 28, 1))
    outputs = pooling(np.abs(convs(inputs, filters)), 2, 2)
    outputs = outputs.reshape((batch_size, -1))
    # normalization
    outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
    return outputs


class NeuralNetwork:
    '''前馈网络'''
    def __init__(self, inode: int, hnode: int, onode: int):
        '''创建包含一个隐层的全连接网络'''
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

    def train(self, inputs, labels):
        '''训练网络

        Args:
            inputs: 输入数据
            labels: 数据标签
        '''
        # batch_size = len(inputs)

        h_outputs = self.sigmoid(np.dot(inputs, self.h_weight) + self.h_bias)
        outputs = self.softmax(np.dot(h_outputs, self.o_weight) + self.o_bias)
        # loss = np.sum(self.cross_entropy(labels, outputs))
        # l_rate, delta = min(0.0003 / batch_size * loss, 0.001) , outputs - labels

        # self.o_bias -= l_rate / batch_size * np.sum(delta, axis=0, keepdims=True)
        # self.o_weight -= l_rate / batch_size * np.dot(h_outputs.T, delta)
        # # delta = np.dot(delta, self.o_weight.T)
        # delta = np.dot(delta, self.o_weight.T) * h_outputs * (1 - h_outputs)
        # self.h_bias -= l_rate / batch_size * np.sum(delta, axis=0, keepdims=True)
        # self.h_weight -= l_rate / batch_size * np.dot(inputs.T, delta)
        # return loss

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

        l_rate = loss / sum_delta * 20

        self.o_bias -= l_rate * delta_o_bias
        self.o_weight -= l_rate * delta_o_weight
        self.h_bias -= l_rate * delta_h_bias
        self.h_weight -= l_rate * delta_h_weight
        return loss

    def predict(self, inputs):
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

    def plot(self):
        ax1 = plt.subplot()
        ax1.plot(self.losses[:, 0], self.losses[:, 1], 'r')
        ax1.set_xlabel('Pass')
        ax1.set_ylabel('Loss')
        ax2 = ax1.twinx()
        ax2.plot(self.accurations[:100, 0], self.accurations[:100, 1], 'b')
        ax2.set_ylabel('Accuration')
        plt.annotate(str(self.last_test_accuration), (self.accurations[-1]))
        plt.show()

    def auto_train(self, inputs, labels, test_inputs, test_labels):
        labels_array = np.array([[0 if i != label else 1 for i in range(10)] 
        for label in labels], dtype=np.int8, ndmin=2)
        total_count, loss, losses, accurations = 0, 0., [[100, -1]], [[2, -1]]
        for batch_size in (1, 2, 5):
            print('batch size:', batch_size)
            for epoch in range(1):
                epoch_loss = 0.
                for batch in range(0, len(inputs) // batch_size):
                    beg, end = batch * batch_size, batch * batch_size + batch_size
                    batch_loss = self.train(inputs[beg: end], labels_array[beg: end]) 
                    epoch_loss += batch_loss
                    loss += batch_loss
                    total_count += batch_size
                    count = total_count - losses[-1][0]
                    if count >= 100:
                        losses.append(([total_count, loss / count]))
                        loss = 0
                    count = total_count - accurations[-1][0]
                    if count >= 10000 or count >= accurations[-1][0]:
                        accuration = self.get_correct_ratio(test_inputs, test_labels)
                        accurations.append([total_count, accuration])
                print('epoch:', epoch, '\tloss:', epoch_loss / len(inputs), 
                '\taccuration:', accurations[-1][1])
        self.losses = np.array(losses[1:])
        self.accurations = np.array(accurations[1:])
        self.last_test_accuration = self.get_correct_ratio(test_inputs, test_labels)
        self.last_train_accuration = self.get_correct_ratio(inputs, labels)
        print('last test accuration: ', self.last_test_accuration)
        print('last train accuration: ', self.last_train_accuration)
        self.plot()

def main():
    try:
        with open('E:/DataSet/mnist/model.dat', 'rb') as model:
            nn = pickle.load(model)
            inputs, labels = get_data('E:/DataSet/mnist/my_test.csv')
            inputs = extract_feature(inputs)
            # inputs = np.array(inputs)
            print('correct ration:', nn.get_correct_ratio(inputs, labels))
    except Exception:
        inputs, labels = get_data('E:/DataSet/mnist/mnist_test.csv')
        test_inputs, test_labels = get_data('E:/DataSet/mnist/mnist_train_100.csv')
        inputs = extract_feature(inputs)
        # inputs = np.array(inputs)
        test_inputs = extract_feature(test_inputs)
        # test_inputs = np.array(test_inputs)
        nn = NeuralNetwork(inode=inputs.shape[1], hnode=150, onode=10)
        nn.auto_train(inputs, labels, test_inputs, test_labels)

        confirm = input('save model?(n/y)')
        if confirm == 'y':
            # save model
            with open('E:/DataSet/mnist/model.dat', 'wb') as file:
                pickle.dump(nn, file)


if __name__ == '__main__':
    main()