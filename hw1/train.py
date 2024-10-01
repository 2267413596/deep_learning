import numpy as np


def train(nn, X, y, batch_size=64, epochs=10):
    n_samples = X.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # 前向传播
            predictions, cache = nn.forward_propagation(X_batch)

            # 计算损失
            loss = -np.sum(np.log(predictions[range(X_batch.shape[0]), y_batch])) / X_batch.shape[0]

            # 反向传播
            grads = nn.backward_propagation(X_batch, y_batch, cache)

            # 更新参数
            nn.update_parameters(grads)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

# 加载数据
train_images = load_images('train-images.idx3-ubyte')
train_labels = load_labels('train-labels.idx1-ubyte')

# 实例化并训练模型
nn = FullyConnectedNeuralNetwork(input_size=784, hidden_layer1_size=128, hidden_layer2_size=64, hidden_layer3_size=32,
                                 output_size=10, learning_rate=0.1)
train(nn, train_images, train_labels)
