import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.color
import PIL
import mnist
import math

# result = []
# for i in range(10):
#     result.append([i])
#     img = plt.imread('E:/DataSet/mnist/my_test/' + str(i) + '.png')
#     img = skimage.color.rgb2gray(img)
#     img = np.abs(1 - img) * 255
#     img = np.array(img, dtype=np.int)
#     # plt.imshow(img)
#     # plt.show()
#     result[i].extend(img.reshape((-1)))
# s = '\n'.join([','.join([str(c) for c in res]) for res in result])
# file = open('E:/DataSet/mnist/my_test.csv', 'w')
# file.write(s)
# file.close()

# def relu(x):
#     return np.where(x < 1, 0, x)

# inputs, labels = mnist.get_data('E:/DataSet/mnist/mnist_test_10.csv')
# filters = np.array([[[1, 1, 1], [0, 0, 0], [-1, -1, -1]], 
# [[1, 0, -1], [1, 0, -1], [1, 0, -1]], [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], 
# [[1, 1, 0], [1, 0, -1], [0, -1, -1]]])
# batch_size = len(inputs)
# inputs = np.reshape(inputs, (batch_size, 28, 28, 1))
# # outputs = mnist.pooling(np.abs(mnist.convs(inputs, filters)), 2, 2)
# outputs = relu(mnist.convs(inputs, filters))
# for i in range(5):
#     for j in range(batch_size):
#         ax = plt.subplot(5, 10, i * 10 + j + 1)
#         if i == 0:
#             ax.imshow(inputs[j, :, :, 0], cmap='gray')
#         else:
#             ax.imshow(outputs[j, :, :, i - 1], cmap='gray')
#         ax.axis('off')
# plt.show()