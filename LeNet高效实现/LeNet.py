# coding: utf-8

'''
代码实现了LeNet网络，并完成了手写数字数据集MNIST的训练。
代码直接按顺序依次运行即可。
'''


# 先对MNIST数据集进行读入以及处理
# 数据从MNIST官网直接下载
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack

def read_image(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I',f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols, 1)   # 将图片格式进行规定，加上通道数
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

def padding(image, zero_num):
    if len(image.shape) == 4:
        image_padding = np.zeros((image.shape[0],image.shape[1]+2*zero_num,image.shape[2]+2*zero_num,image.shape[3]))
        image_padding[:,zero_num:image.shape[1]+zero_num,zero_num:image.shape[2]+zero_num,:] = image
    elif len(image.shape) == 3:
        image_padding = np.zeros((image.shape[0]+2*zero_num, image.shape[1]+2*zero_num, image.shape[2]))
        image_padding[zero_num:image.shape[0]+zero_num, zero_num:image.shape[1]+zero_num,:] = image
    else:
        print("维度错误")
        sys.exit()
    return image_padding



# 加载数据集以及数据预处理

def dataset_loader():
	train_image = read_image(r'C:\Users\95410\Downloads\数据集\MNIST\train-images.idx3-ubyte')
	train_label = read_label(r'C:\Users\95410\Downloads\数据集\MNIST\train-labels.idx1-ubyte')
	test_image = read_image(r'C:\Users\95410\Downloads\数据集\MNIST\t10k-images.idx3-ubyte')
	test_label = read_label(r'C:\Users\95410\Downloads\数据集\MNIST\t10k-labels.idx1-ubyte')

	train_image = normalize_image(train_image)
	train_label = one_hot_label(train_label)
	train_label = train_label.reshape(train_label.shape[0], train_label.shape[1], 1)

	test_image = normalize_image(test_image)
	test_label = one_hot_label(test_label)
	test_label = test_label.reshape(test_label.shape[0], test_label.shape[1], 1)

	train_image = padding(train_image,2) #对初始图像进行零填充，保证与LeNet输入结构一致60000*32*32*1
	test_image = padding(test_image,2)

	return train_image, train_label, test_image, test_label






def conv(img, conv_filter):
   
	if len(img.shape)!=3 or len(conv_filter.shape)!=4:
	    print("卷积运算所输入的维度不符合要求")
	    sys.exit()
	    
	if img.shape[-1] != conv_filter.shape[-1]:
	    print("卷积输入图片与卷积核的通道数不一致")
	    sys.exit()
	    
	img_h, img_w, img_ch = img.shape
	filter_num, filter_h, filter_w, img_ch = conv_filter.shape
	feature_h = img_h - filter_h + 1
	feature_w = img_w - filter_w + 1

	# 初始化输出的特征图片，由于没有使用零填充，图片尺寸会减小
	img_out = np.zeros((feature_h, feature_w, filter_num))
	img_matrix = np.zeros((feature_h*feature_w, filter_h*filter_w*img_ch))
	filter_matrix = np.zeros((filter_h*filter_w*img_ch, filter_num))

    # 将输入图片张量转换成矩阵形式
	for j in range(img_ch):
	    img_2d = np.copy(img[:,:,j])   
	    shape=(feature_h,feature_w,filter_h,filter_w) 
	    strides = (img_w,1,img_w,1)
	    strides = img_2d.itemsize * np.array(strides)
	    x_stride = np.lib.stride_tricks.as_strided(img_2d, shape=shape, strides=strides)
	    x_cols = np.ascontiguousarray(x_stride)
	    x_cols = x_cols.reshape(feature_h*feature_w,filter_h*filter_w)
	    img_matrix[:,j*filter_h*filter_w:(j+1)*filter_h*filter_w]=x_cols
	    
    
    # 将卷积核张量转换成矩阵形式
	for i in range(filter_num):
	    filter_matrix[:,i] = conv_filter[i,:].transpose(2,0,1).reshape(filter_w*filter_h*img_ch)

	feature_matrix = np.dot(img_matrix, filter_matrix)

	for i in range(filter_num):
	    img_out[:,:,i] = feature_matrix[:,i].reshape(feature_h, feature_w)

	return img_out

def conv_cal_w(out_img_delta, in_img):
    # 同样利用img2col思想加速
    img_h, img_w, img_ch = in_img.shape
    feature_h, feature_w, filter_num = out_img_delta.shape
    filter_h = img_h - feature_h + 1
    filter_w = img_w - feature_w + 1
    
    in_img_matrix = np.zeros([filter_h*filter_w*img_ch, feature_h*feature_w])
    out_img_delta_matrix = np.zeros([feature_h*feature_w, filter_num])
    
    # 将输入图片转换成矩阵形式
    for j in range(img_ch):
        img_2d = np.copy(in_img[:,:,j])   
        shape=(filter_h,filter_w,feature_h,feature_w) 
        strides = (img_w,1,img_w,1)
        strides = img_2d.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(img_2d, shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(x_stride)
        x_cols = x_cols.reshape(filter_h*filter_w,feature_h*feature_w)
        in_img_matrix[j*filter_h*filter_w:(j+1)*filter_h*filter_w,:]=x_cols
        
    
    # 将输出图片delta误差转换成矩阵形式
    for i in range(filter_num):
        out_img_delta_matrix[:, i] = out_img_delta[:, :, i].reshape(feature_h*feature_w)
        
    filter_matrix = np.dot(in_img_matrix, out_img_delta_matrix)
    nabla_conv = np.zeros([filter_num, filter_h, filter_w, img_ch])
    
    for i in range(filter_num):
        nabla_conv[i,:] = filter_matrix[:,i].reshape(img_ch, filter_h, filter_w).transpose(1,2,0)
        
    return nabla_conv

def conv_cal_b(out_img_delta):
    nabla_b = np.zeros((out_img_delta.shape[-1],1))
    for i in range(out_img_delta.shape[-1]):
        nabla_b[i] = np.sum(out_img_delta[:,:,i])
    return nabla_b


def relu(feature):
    '''Relu激活函数，有两种情况会使用到
    当在卷积层中使用时，feature为一个三维张量，，[行，列，通道]
    当在全连接层中使用时，feature为一个列向量'''
    return feature*(feature>0)


def relu_prime(feature):  # 对relu函数的求导
    '''relu函数的一阶导数，间断点导数认为是0'''
    
    return 1*(feature>0)


def pool(feature, size=2, stride=2):
    feature_h, feature_w, feature_ch = feature.shape
    pool_h = np.uint16((feature_h - size)/stride + 1)
    pool_w = np.uint16((feature_w - size)/stride + 1)
    feature_reshaped = feature.reshape(pool_h, feature_h//pool_h, pool_w, feature_w//pool_w, feature_ch)
    out = feature_reshaped.max(axis=1).max(axis=2)
    out_location_c = feature_reshaped.max(axis=1).argmax(axis=2)
    out_location_r = feature_reshaped.max(axis=3).argmax(axis=1)
    out_location = out_location_r * size + out_location_c
    return out, out_location

def pool_delta_error_bp(pool_out_delta, pool_out_max_location, size=2, stride=2):
    pool_h, pool_w, pool_ch = pool_out_delta.shape
    in_h = np.uint16((pool_h-1)*stride+size)
    in_w = np.uint16((pool_w-1)*stride+size)
    in_ch = pool_ch
    
    pool_out_delta_reshaped = pool_out_delta.transpose(2,0,1)
    pool_out_delta_reshaped = pool_out_delta_reshaped.flatten()
    
    pool_out_max_location_reshaped = pool_out_max_location.transpose(2,0,1)
    pool_out_max_location_reshaped = pool_out_max_location_reshaped.flatten()
    
    in_delta_matrix = np.zeros([pool_h*pool_w*pool_ch,size*size])
    
    in_delta_matrix[np.arange(pool_h*pool_w*pool_ch), pool_out_max_location_reshaped] = pool_out_delta_reshaped
    
    in_delta = in_delta_matrix.reshape(pool_ch,pool_h, pool_w, size, size)
    in_delta = in_delta.transpose(1,3,2,4,0)
    in_delta = in_delta.reshape(in_h, in_w, in_ch)
    return in_delta

def rot180(conv_filters):
    rot180_filters = np.zeros((conv_filters.shape))
    for filter_num in range(conv_filters.shape[0]):
        for img_ch in range(conv_filters.shape[-1]):
            rot180_filters[filter_num,:,:,img_ch] = np.flipud(np.fliplr(conv_filters[filter_num,:,:,img_ch]))
    return rot180_filters
                
    
def soft_max(z):
        
    tmp = np.max(z)
    z -= tmp  # 用于缩放每行的元素，避免溢出，有效
    z = np.exp(z)
    tmp = np.sum(z)
    z /= tmp
    
    return z

def add_bias(conv, bias):
    if conv.shape[-1] != bias.shape[0]:
        print("给卷积添加偏置维度出错")
    else:
        for i in range(bias.shape[0]):
            conv[:,:,i] += bias[i,0]
    return conv


class ConvNet(object):
    
    def __init__(self):
        
        '''
        2层卷积，2层池化，3层全连接'''
        self.filters = [np.random.randn(6, 5, 5, 1)] #图像变成 28*28*6 池化后图像变成14*14*6
        self.filters_biases = [np.random.randn(6,1)]
        self.filters.append(np.random.randn(16, 5, 5, 6)) #图像变成 10*10*16 池化后变成5*5*16
        self.filters_biases.append(np.random.randn(16,1))
        
        self.weights = [np.random.randn(120,400)]
        self.weights.append(np.random.randn(84,120))
        self.weights.append(np.random.randn(10,84))
        self.biases = [np.random.randn(120,1)]
        self.biases.append(np.random.randn(84,1))
        self.biases.append(np.random.randn(10,1))
    
    def feed_forward(self, x):
        #第一层卷积
        conv1 = add_bias( conv(x, self.filters[0]), self.filters_biases[0] )
        relu1 = relu(conv1)
        pool1, pool1_max_locate = pool(relu1)
        
        #第二层卷积
        conv2 = add_bias( conv(pool1, self.filters[1]), self.filters_biases[1])
        relu2 = relu(conv2)
        pool2, pool2_max_locate = pool(relu2)
        
        #拉直
        straight_input = pool2.reshape(pool2.shape[0] * pool2.shape[1] * pool2.shape[2], 1)
        
        #第一层全连接
        full_connect1_z = np.dot(self.weights[0], straight_input) + self.biases[0]
        full_connect1_a = relu(full_connect1_z)
        
        #第二层全连接
        full_connect2_z = np.dot(self.weights[1], full_connect1_a) + self.biases[1]
        full_connect2_a = relu(full_connect2_z)
        
        #第三层全连接（输出）
        full_connect3_z = np.dot(self.weights[2], full_connect2_a) + self.biases[2]
        full_connect3_a = soft_max(full_connect3_z)
        return full_connect3_a
    
    def evaluate(self, images, labels):
        result = 0 # 用于记录分类正确率
        J = 0 # 用于记录损失大小
        eta = 1e-7 # 防止计算log溢出
        for img, lab in zip(images, labels):
            predict_label = self.feed_forward(img)
            if np.argmax(predict_label) == np.argmax(lab):
                result += 1
            J = J + sum(-lab*(np.log(predict_label+eta))-(1-lab)*(np.log(1-predict_label+eta)))
        return result, J # 以元组形式返回
    
    def SGD(self, train_image, train_label, test_image, test_label, epochs, mini_batch_size, eta):
        '''
        随机梯度下降法，需要送入训练数据，训练标签，测试数据，测试标签，训练轮数，batch_size大小，学习率
        '''
        batch_num = 0
        
        fx = []
        fy_loss = []
        fy_accuracy = []
        for j in range(epochs):
            mini_batches_image = [train_image[k:k+mini_batch_size] for k in range(0, len(train_image), mini_batch_size)]
            mini_batches_label = [train_label[k:k+mini_batch_size] for k in range(0, len(train_label), mini_batch_size)]
            for mini_batch_image, mini_batch_label in zip(mini_batches_image, mini_batches_label):
                batch_num += 1
                if batch_num * mini_batch_size > len(train_image):
                    batch_num = 1
                
                self.update_mini_batch(mini_batch_image, mini_batch_label, eta, mini_batch_size)
                
                print("\rEpoch{0}:{1}/{2}".format(j+1, batch_num*mini_batch_size, len(train_image)), end='')
            accurate_num, loss = self.evaluate(test_image, test_label)
            plt.figure(1)
            fx.append(j)
            fy_accuracy.append((0.0+accurate_num)/len(test_image))
            fy_loss.append(loss)
            print(" After epoch{0}: accuracy is {1}/{2},loss is {3}".format(j+1, accurate_num, len(test_image), loss))
        
        my_x_ticks = np.arange(1, epochs+1, 1)
        plt.figure(1)
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.xticks(my_x_ticks)
        plt.plot(fx, fy_loss, 'bo-')
        
        plt.figure(2)
        plt.xlabel('Epochs')
        plt.ylabel('accuracy')
        plt.xticks(my_x_ticks)
        plt.plot(fx, fy_accuracy, 'r+-')
        plt.show()
            
                
    def update_mini_batch(self, mini_batch_image, mini_batch_label, eta, mini_batch_size):
        '''通过一个batch的数据对神经网络参数进行更新
        需要先求这个batch中每张图片的误差反向传播求得的权重梯度以及偏置梯度'''
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        nabla_f = [np.zeros(f.shape) for f in self.filters]
        nabla_fb = [np.zeros(fb.shape) for fb in self.filters_biases]
        
        for x,y in zip(mini_batch_image, mini_batch_label):
            delta_nabla_w, delta_nabla_b, delta_nabla_f, delta_nabla_fb = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_f = [nf+dnf for nf, dnf in zip(nabla_f, delta_nabla_f)]
            nabla_fb = [nfb + dnfb for nfb, dnfb in zip(nabla_fb, delta_nabla_fb)]
        self.weights = [w-(eta/mini_batch_size)*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/mini_batch_size)*nb for b, nb in zip(self.biases, nabla_b)]
        self.filters = [f-(eta/mini_batch_size)*nf for f, nf in zip(self.filters, nabla_f)]
        self.filters_biases = [fb-(eta/mini_batch_size)*nfb for fb, nfb in zip(self.filters_biases, nabla_fb)]
    
    def backprop(self, x, y):
        
        '''计算通过单幅图像求得梯度'''
        
        #先前向传播，求出各中间量
        #第一层卷积
        conv1 = add_bias( conv(x, self.filters[0]), self.filters_biases[0] )
        relu1 = relu(conv1)
        pool1, pool1_max_locate = pool(relu1)

        #第二层卷积
        conv2 = add_bias( conv(pool1, self.filters[1]), self.filters_biases[1] )
        relu2 = relu(conv2)
        pool2, pool2_max_locate = pool(relu2)
        
        #拉直
        straight_input = pool2.reshape(pool2.shape[0] * pool2.shape[1] * pool2.shape[2], 1)
        
        #第一层全连接
        full_connect1_z = np.dot(self.weights[0], straight_input) + self.biases[0]
        full_connect1_a = relu(full_connect1_z)
        
        #第二层全连接
        full_connect2_z = np.dot(self.weights[1], full_connect1_a) + self.biases[1]
        full_connect2_a = relu(full_connect2_z)
        
        #第三层全连接（输出）
        full_connect3_z = np.dot(self.weights[2], full_connect2_a) + self.biases[2]
        full_connect3_a = soft_max(full_connect3_z)
            
        # 在这里我们使用交叉熵损失，激活函数为softmax，因此delta值就为 a-y，即对正确位置的预测值减1
        delta_fc3 = full_connect3_a - y
        delta_fc2 = np.dot(self.weights[2].transpose(), delta_fc3) * relu_prime(full_connect2_z)
        delta_fc1 = np.dot(self.weights[1].transpose(), delta_fc2) * relu_prime(full_connect1_z)
        delta_straight_input = np.dot(self.weights[0].transpose(), delta_fc1)
        delta_pool2 = delta_straight_input.reshape(pool2.shape)
        
        delta_conv2 = pool_delta_error_bp(delta_pool2, pool2_max_locate) * relu_prime(conv2)
        
        delta_pool1 = conv(padding(delta_conv2, self.filters[1].shape[1]-1), rot180(self.filters[1]).swapaxes(0,3))
        
        delta_conv1 = pool_delta_error_bp(delta_pool1, pool1_max_locate) * relu_prime(conv1)
        
        
        
        #求各参数的导数
        nabla_w2 = np.dot(delta_fc3, full_connect2_a.transpose())
        nabla_b2 = delta_fc3
        nabla_w1 = np.dot(delta_fc2, full_connect1_a.transpose())
        nabla_b1 = delta_fc2
        nabla_w0 = np.dot(delta_fc1, straight_input.transpose())
        nabla_b0 = delta_fc1
        
        
        nabla_filters1 = conv_cal_w(delta_conv2, pool1) 
        nabla_filters0 = conv_cal_w(delta_conv1, x)
        nabla_filters_biases1 = conv_cal_b(delta_conv2)
        nabla_filters_biases0 = conv_cal_b(delta_conv1)
        
        nabla_w = [nabla_w0, nabla_w1, nabla_w2]
        nabla_b = [nabla_b0, nabla_b1, nabla_b2]
        nabla_f = [nabla_filters0, nabla_filters1]
        nabla_fb = [nabla_filters_biases0, nabla_filters_biases1]
        return nabla_w, nabla_b, nabla_f, nabla_fb






def main():
    # image维度为 num×rows×cols×1，像素值范围在0-1
    # label维度为num×class_num×1
	train_image, train_label, test_image, test_label = dataset_loader()

    # 初试化卷积神经网络
	net = ConvNet()
	# 对卷积神经网络进行训练，设定好训练数据，验证数据，训练轮数，batch大小和学习率
	net.SGD(train_image, train_label, test_image, test_label, 50, 100, 1e-5) 

if __name__ == '__main__':
    main()