import numpy as np
import model
import model_cp
from utils import load_labels, load_images, load_labels_cp, load_images_cp
import cupy as cp


def train(nn, X, y, batch_size=64, epochs=10, gpu=False):
    n_samples = X.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        if gpu:
            indices = cp.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # 前向传播
            predictions, cache = nn.forward_propagation(X_batch)

            # 交叉熵计算损失
            # predictions[range(X_batch.shape[0]), y_batch]提取了每个样本中模型预测的真实类别对应的概率
            loss = 0
            if gpu:
                loss = -cp.sum(cp.log(predictions[range(X_batch.shape[0]), y_batch])) / X_batch.shape[0]
            else:
                loss = -np.sum(np.log(predictions[range(X_batch.shape[0]), y_batch])) / X_batch.shape[0]

            # 反向传播
            grads = nn.backward_propagation(X_batch, y_batch, cache)

            # 更新参数
            nn.update_parameters(grads)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

def test_model(X_test, y_test, nn, gpu=False):
    predictions = nn.predict(X_test)
    accuracy = 0
    if gpu:
        accuracy = cp.mean(predictions == y_test)
    else:
        accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


train_images = load_images_cp('data/train-images.idx3-ubyte')
train_labels = load_labels_cp('data/train-labels.idx1-ubyte')
nn = model_cp.FullyConnectedNeuralNetwork(input_size=784, hidden_layer1_size=256, hidden_layer2_size=128,
                                       hidden_layer3_size=64, output_size=10, learning_rate=0.1)
train(nn, train_images, train_labels, batch_size=128, epochs=100, gpu=True)
nn.save_model(filename="gpu.pkl")

# nn.load_model()
test_images = load_images_cp('data/t10k-images.idx3-ubyte')
test_labels = load_labels_cp('data/t10k-labels.idx1-ubyte')
print("test begin")
test_model(test_images, test_labels, nn)