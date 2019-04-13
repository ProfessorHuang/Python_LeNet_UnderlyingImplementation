# coding: utf-8


# 使用Python从底层实现了一个全连接神经网络


# 先对MNIST数据集进行读入以及处理
# 数据从MNIST官网直接下载

import numpy as np
from struct import unpack

def read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I',f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784, 1)  # 在这里可以调整图片读入格式
    return img

def read_label(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I',f.read(8))
        label = np.fromfile(f, dtype=np.uint8)
    return label

def normalize_image(image):
    img = image.astype(np.float32)/255.0
    return img

def one_hot_label(label):
    lab = np.zeros((label.size, 10))
    for i, row in enumerate(lab):
        row[label[i]] = 1
    return lab



# 加载数据集以及数据预处理

def dataset_loader():
    train_image = read_image(r'C:\Users\95410\Downloads\数据集\MNIST\train-images.idx3-ubyte')
    train_label = read_label(r'C:\Users\95410\Downloads\数据集\MNIST\train-labels.idx1-ubyte')
    test_image = read_image(r'C:\Users\95410\Downloads\数据集\MNIST\t10k-images.idx3-ubyte')
    test_label = read_label(r'C:\Users\95410\Downloads\数据集\MNIST\t10k-labels.idx1-ubyte')

    train_image = normalize_image(train_image)
    train_label = one_hot_label(train_label)
    train_label = train_label.reshape(train_label.shape[0],train_label.shape[1],1)

    test_image = normalize_image(test_image)
    test_label = one_hot_label(test_label)
    test_label = test_label.reshape(test_label.shape[0],test_label.shape[1],1)
    
    return train_image, train_label, test_image, test_label



# 编写神经网络类
class NetWork(object):
    
    def __init__(self, sizes):
        '''
        初始化神经网络，给每层的权重和偏置赋初值
        权重为一个列表，列表中每个值是一个二维n×m的numpy数组
        偏置为一个列表，列表中每个值是一个二维n×1的numpy数组'''
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(n,m) for m,n in zip(sizes[:-1], sizes[1:])]   # 一定得用rnadn而不是random
        self.biases = [np.random.randn(n,1) for n in sizes[1:]]
        
    def sigmoid(self, z):
        '''sigmoid激活函数'''
        a = 1.0 / (1.0 + np.exp(-z))
        return a
    
    def sigmoid_prime(self, z):
        '''sigmoid函数的一阶导数'''
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    
    def feed_forward(self, x):
        '''完成前向传播过程，由输入值计算神经网络最终的输出值
        输入为一个列向量，输出也为一个列向量'''
        value = x
        for i in range(len(self.weights)):
            value = self.sigmoid(np.dot(self.weights[i], value) + self.biases[i])
        y = value
        return y
    
    def evaluate(self, images, labels):
        result = 0
        for img, lab in zip(images, labels):
            predict_label = self.feed_forward(img)
            if np.argmax(predict_label) == np.argmax(lab):
                result += 1
        return result
        
    
    def SGD(self, train_image, train_label, test_image, test_label, epochs, mini_batch_size, eta):
        '''Stochastic gradiend descent随机梯度下降法，将训练数据分多个batch
        一次使用一个mini_batch_size的数据，调用update_mini_batch函数更新参数'''
        for j in range(epochs):
            mini_batches_image = [train_image[k:k+mini_batch_size] for k in range(0, len(train_image), mini_batch_size)]
            mini_batches_label = [train_label[k:k+mini_batch_size] for k in range(0, len(train_label), mini_batch_size)]
            for mini_batch_image, mini_batch_label in zip(mini_batches_image, mini_batches_label):
                self.update_mini_batch(mini_batch_image, mini_batch_label, eta, mini_batch_size)
            print("Epoch{0}: accuracy is {1}/{2}".format(j+1, self.evaluate(test_image, test_label), len(test_image)))
                
    def update_mini_batch(self, mini_batch_image, mini_batch_label, eta, mini_batch_size):
        '''通过一个batch的数据对神经网络参数进行更新
        需要对当前batch中每张图片调用backprop函数将误差反向传播
        求每张图片对应的权重梯度以及偏置梯度，最后进行平均使用梯度下降法更新参数'''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in zip(mini_batch_image, mini_batch_label):
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/mini_batch_size)*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/mini_batch_size)*nb for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        '''计算通过单幅图像求得的每层权重和偏置的梯度'''
        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # 前向传播，计算各层的激活前的输出值以及激活之后的输出值，为下一步反向传播计算作准备
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        # 先求最后一层的delta误差以及b和W的导数
        cost = activations[-1] - y    
        delta = cost * self.sigmoid_prime(zs[-1])
        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 将delta误差反向传播以及各层b和W的导数，一直计算到第二层
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * self.sigmoid_prime(zs[-l])
            delta_nabla_b[-l] = delta
            delta_nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return delta_nabla_b, delta_nabla_w
        
            


def main():
    # 加载数据集
	train_image, train_label, test_image, test_label = dataset_loader()

	# 训练神经网络
	net_trained = NetWork([784, 30, 10])
	net_trained.SGD(train_image, train_label, test_image, test_label, 30, 10, 3)

if __name__ == '__main__':
    main()