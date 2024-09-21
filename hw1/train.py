import numpy as np
import model
from utils import load_labels, load_images


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

            # 交叉熵计算损失
            # predictions[range(X_batch.shape[0]), y_batch]提取了每个样本中模型预测的真实类别对应的概率
            loss = -np.sum(np.log(predictions[range(X_batch.shape[0]), y_batch])) / X_batch.shape[0]

            # 反向传播
            grads = nn.backward_propagation(X_batch, y_batch, cache)

            # 更新参数
            nn.update_parameters(grads)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

def test_model(X_test, y_test, nn):
    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


# train_images = load_images('data/train-images.idx3-ubyte')
# train_labels = load_labels('data/train-labels.idx1-ubyte')
# nn = model.FullyConnectedNeuralNetwork(input_size=784, hidden_layer1_size=256, hidden_layer2_size=128,
#                                        hidden_layer3_size=64, output_size=10, learning_rate=0.2)
# train(nn, train_images, train_labels, batch_size=128, epochs=60)
# nn.save_model()
nn = model.FullyConnectedNeuralNetwork(input_size=784, hidden_layer1_size=256, hidden_layer2_size=128,
                                        hidden_layer3_size=64, output_size=10, learning_rate=0.2)
nn.load_model()
test_images = load_images('data/t10k-images.idx3-ubyte')
test_labels = load_labels('data/t10k-labels.idx1-ubyte')
print("test begin")
test_model(test_images, test_labels, nn)