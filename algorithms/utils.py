import numpy as np
import tensorflow as tf
import math

class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        self.classes = 10
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)
        self.train_label = self.train_label.astype(np.int32)
        self.test_label = self.test_label.astype(np.int32)
        self.train_size, self.test_size = self.train_data.shape[0], self.test_data.shape[0]

    def get_data_attr(self, data_type, attr):
        return getattr(self, data_type + '_' + attr)

    def batch_loader(self, batch_size=64, data_type='train', flatten=False):
        data_size = self.get_data_attr(data_type, 'size')
        data = self.get_data_attr(data_type, 'data')
        if flatten:
            data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        label = self.get_data_attr(data_type, 'label')

        permutation = np.random.permutation(data_size)
        batch_permutation_indices = (permutation[i: i + batch_size] for i in range(0, data_size, batch_size))
        for batch_permutation in batch_permutation_indices:
            yield data[batch_permutation], label[batch_permutation]

class ZeroMetric(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__()

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        return 0.


if __name__ == '__main__':
    data_loader = MNISTLoader()
    X, y = next(data_loader.batch_loader(64))
    print(X.shape, y.shape)
