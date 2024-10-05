import numpy as np
import pickle
import os


class FullyConnectedNeuralNetwork:
    def __init__(self, input_size, hidden_layer1_size, hidden_layer2_size, hidden_layer3_size, output_size,
                 learning_rate=0.01):
        # 初始化网络参数
        self.params = {
            'W1': np.random.randn(input_size, hidden_layer1_size) * 0.01,
            'b1': np.zeros((1, hidden_layer1_size)),
            'W2': np.random.randn(hidden_layer1_size, hidden_layer2_size) * 0.01,
            'b2': np.zeros((1, hidden_layer2_size)),
            'W3': np.random.randn(hidden_layer2_size, hidden_layer3_size) * 0.01,
            'b3': np.zeros((1, hidden_layer3_size)),
            'W4': np.random.randn(hidden_layer3_size, output_size) * 0.01,
            'b4': np.zeros((1, output_size))
        }
        self.learning_rate = learning_rate

    # 激活函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    # 前向传播
    def forward_propagation(self, X):
        params = self.params
        Z1 = np.dot(X, params['W1']) + params['b1']
        A1 = self.relu(Z1)
        # A1 = self.sigmoid(Z1)

        Z2 = np.dot(A1, params['W2']) + params['b2']
        A2= self.relu(Z2)
        # A2 = self.sigmoid(Z2)

        Z3 = np.dot(A2, params['W3']) + params['b3']
        A3 = self.relu(Z3)
        # A3 = self.sigmoid(Z3)

        Z4 = np.dot(A3, params['W4']) + params['b4']
        A4 = self.relu(Z4)
        # A4 = self.sigmoid(Z4)

        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3, 'Z4': Z4, 'A4': A4}
        return A4, cache

    # 反向传播
    def backward_propagation(self, X, y, cache):
        m = X.shape[0]
        params = self.params
        A4 = cache['A4']
        A3 = cache['A3']
        A2 = cache['A2']
        A1 = cache['A1']

        # 计算损失的导数
        dZ4 = A4
        dZ4[range(m), y] -= 1
        dZ4 /= m

        dW4 = np.dot(A3.T, dZ4)
        db4 = np.sum(dZ4, axis=0, keepdims=True)

        dZ3 = np.dot(dZ4, params['W4'].T) * self.relu_derivative(A3)
        dW3 = np.dot(A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dZ2 = np.dot(dZ3, params['W3'].T) * self.relu_derivative(A2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, params['W2'].T) * self.relu_derivative(A1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        # print(f"Gradients:\ndW1: {np.linalg.norm(dW1)}\ndW2: {np.linalg.norm(dW2)}\ndW3: {np.linalg.norm(dW3)}\ndW4: {np.linalg.norm(dW4)}")

        grads = {'dW4': dW4, 'db4': db4, 'dW3': dW3, 'db3': db3, 'dW2': dW2, 'db2': db2, 'dW1': dW1, 'db1': db1}
        return grads

    # 更新参数
    def update_parameters(self, grads):
        for key in self.params:
            self.params[key] -= self.learning_rate * grads['d' + key]

    # 推理
    def predict(self, X):
        predictions, _ = self.forward_propagation(X)
        return np.argmax(predictions, axis=1)

    def save_model(self, folder='model', filename='mnist_model.pkl'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.params, f)
        print(f"Model saved to {filepath}")


# 有个问题，为什么说保存W就行，不用保存b
    def save_weights(self, folder='weights'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        wi1 = 'wi1.npy'
        w12 = 'w12.npy'
        w23 = 'w23.npy'
        w3o = 'w3o.npy'
        np.save(os.path.join(folder, wi1), self.params['W1'])
        np.save(os.path.join(folder, w12), self.params['W2'])
        np.save(os.path.join(folder, w23), self.params['W3'])
        np.save(os.path.join(folder, w3o), self.params['W4'])

    def load_weights(self, folder='weights'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        wi1 = 'wi1.npy'
        w12 = 'w12.npy'
        w23 = 'w23.npy'
        w3o = 'w3o.npy'
        self.params['W1'] = np.load(os.path.join(folder, wi1))
        self.params['W2'] = np.load(os.path.join(folder, w12))
        self.params['W3'] = np.load(os.path.join(folder, w23))
        self.params['W4'] = np.load(os.path.join(folder, w3o))

    def load_model(self, folder='model', filename='mnist_model.pkl'):
        # 组合文件夹路径和文件名
        filepath = os.path.join(folder, filename)

        # 加载模型
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.params = params
        print(f"Model loaded from {filepath}")