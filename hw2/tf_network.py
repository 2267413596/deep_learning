import tensorflow.compat.v1 as tf


class LeNet:

    def __init__(self):
        """
        Define some basic parameters here
        """

        pass

    def net(self, feats):
        """
        Define network.
        You can use init_weight() and init_bias() function to init weight matrix,
        for example:
            conv1_W = self.init_weight((3, 3, 1, 6))
            conv1_b = self.init_bias(6)
        :param feats: input features
        :return: logits

        python版本3.5-3.7，tensorflow版本1.15.0
        下载sklearn库：`pip install scikit-learn
        """
        # layer 1
        conv1_W = self.init_weight((5, 5, 1, 6))
        conv1_b = self.init_bias([6])
        conv1 = tf.nn.conv2d(feats, conv1_W, strides=[1, 1, 1, 1], padding='SAME')  # 卷积后大小为28*28*6
        conv1 = tf.nn.bias_add(conv1, conv1_b)
        conv1 = tf.nn.relu(conv1)
        # layer 2
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 池化后大小为14*14*6
        # layer 3
        conv2_W = self.init_weight((5, 5, 6, 16))
        conv2_b = self.init_bias([16])
        conv2 = tf.nn.conv2d(pool1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') # 卷积后大小为10*10*16
        conv2 = tf.nn.bias_add(conv2, conv2_b)
        conv2 = tf.nn.relu(conv2)
        # layer 4
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 池化后大小为5*5*16
        # Flatten layer
        flat = tf.reshape(pool2, [-1, 5 * 5 * 16])
        # layer 5
        fc1_W = self.init_weight([5 * 5 * 16, 120])
        fc1_b = self.init_bias([120])
        fc1 = tf.matmul(flat, fc1_W) + fc1_b
        fc1 = tf.nn.relu(fc1)
        # Layer 6: Fully connected layer with 84 units
        fc2_W = self.init_weight([120, 84])
        fc2_b = self.init_bias([84])
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b
        fc2 = tf.nn.relu(fc2)
        # Output Layer: Fully connected with 10 units (for the 10 digit classes)
        fc3_W = self.init_weight([84, 10])
        fc3_b = self.init_bias([10])
        logits = tf.matmul(fc2, fc3_W) + fc3_b

        return logits
        

    def forward(self, feats):
        """
        Forward the network
        """
        return self.net(feats)

    @staticmethod
    def init_weight(shape):
        """
        Init weight parameter.
        """
        w = tf.truncated_normal(shape=shape, mean=0, stddev=0.1)
        return tf.Variable(w)

    @staticmethod
    def init_bias(shape):
        """
        Init bias parameter.
        """
        b = tf.zeros(shape)
        return tf.Variable(b)
