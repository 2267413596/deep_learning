import struct
import numpy as np

# 读取 idx3-ubyte 图像文件
def load_images(file_path):
    with open(file_path, 'rb') as f:
        # 读取魔数、图像数量、图像高度、图像宽度
        magic_number, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
        if magic_number != 2051:
            raise ValueError("Invalid magic number in IDX3 file")

        # 读取所有图像数据
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        print("训练图片原始形状为" + str(num_rows) + "*" + str(num_cols))
        images = image_data.reshape(num_images, num_rows * num_cols)
        print("训练图片加载完成")
        return images / 255.0  # 归一化

# 读取 idx1-ubyte 标签文件
def load_labels(file_path):
    with open(file_path, 'rb') as f:
        # 读取魔数和标签数量
        magic_number, num_labels = struct.unpack('>II', f.read(8))
        if magic_number != 2049:
            raise ValueError("Invalid magic number in IDX1 file")

        # 读取标签数据
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        print("标签加载完成")
        return labels

# 激活函数和导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# 交叉熵损失
def cross_entropy_loss(predictions, labels):
    n_samples = labels.shape[0]
    logp = -np.log(predictions[range(n_samples), labels])
    return np.sum(logp) / n_samples

# 交叉熵损失的导数
def cross_entropy_derivative(predictions, labels):
    n_samples = predictions.shape[0]
    predictions[range(n_samples), labels] -= 1
    return predictions / n_samples


# 初始化参数
def initialize_parameters(input_size, hidden_layer1_size, hidden_layer2_size, hidden_layer3_size, output_size):
    params = {
        'W1': np.random.randn(input_size, hidden_layer1_size) * 0.01,
        'b1': np.zeros((1, hidden_layer1_size)),
        'W2': np.random.randn(hidden_layer1_size, hidden_layer2_size) * 0.01,
        'b2': np.zeros((1, hidden_layer2_size)),
        'W3': np.random.randn(hidden_layer2_size, hidden_layer3_size) * 0.01,
        'b3': np.zeros((1, hidden_layer3_size)),
        'W4': np.random.randn(hidden_layer3_size, output_size) * 0.01,
        'b4': np.zeros((1, output_size))
    }
    return params


# 前向传播
def forward_propagation(X, params):
    Z1 = np.dot(X, params['W1']) + params['b1']
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, params['W2']) + params['b2']
    A2 = sigmoid(Z2)

    Z3 = np.dot(A2, params['W3']) + params['b3']
    A3 = sigmoid(Z3)

    Z4 = np.dot(A3, params['W4']) + params['b4']
    A4 = softmax(Z4)  # 输出层用 softmax

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3, 'Z4': Z4, 'A4': A4}
    return A4, cache


# 反向传播
def backward_propagation(X, y, cache, params):
    m = X.shape[0]
    A4 = cache['A4']
    A3 = cache['A3']
    A2 = cache['A2']
    A1 = cache['A1']

    dZ4 = cross_entropy_derivative(A4, y)
    dW4 = np.dot(A3.T, dZ4)
    db4 = np.sum(dZ4, axis=0, keepdims=True)

    dZ3 = np.dot(dZ4, params['W4'].T) * sigmoid_derivative(A3)
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dZ2 = np.dot(dZ3, params['W3'].T) * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = np.dot(dZ2, params['W2'].T) * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    grads = {'dW4': dW4, 'db4': db4, 'dW3': dW3, 'db3': db3, 'dW2': dW2, 'db2': db2, 'dW1': dW1, 'db1': db1}
    return grads


# 参数更新
def update_parameters(params, grads, learning_rate):
    params['W1'] -= learning_rate * grads['dW1']
    params['b1'] -= learning_rate * grads['db1']
    params['W2'] -= learning_rate * grads['dW2']
    params['b2'] -= learning_rate * grads['db2']
    params['W3'] -= learning_rate * grads['dW3']
    params['b3'] -= learning_rate * grads['db3']
    params['W4'] -= learning_rate * grads['dW4']
    params['b4'] -= learning_rate * grads['db4']
    return params
# 加载训练数据
# train_images = load_images('data/train-images.idx3-ubyte')
# train_labels = load_labels('data/train-labels.idx1-ubyte')