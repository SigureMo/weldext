import tensorflow as tf

class AutoEncoder(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(name='w',
            shape=[input_shape[-1], self.units], initializer=tf.random_uniform_initializer())
        self.b_1 = self.add_weight(name='b_1',
            shape=[self.units], initializer=tf.random_uniform_initializer())
        self.b_2 = self.add_weight(name='b_2',
            shape=[input_shape[-1]], initializer=tf.random_uniform_initializer())

    def encode(self, inputs):
        embedding = tf.matmul(inputs, self.w) + self.b_1
        embedding = tf.nn.relu(embedding)
        return embedding

    def decode(self, embedding):
        outputs = tf.matmul(embedding, tf.transpose(self.w)) + self.b_2
        outputs = tf.nn.relu(outputs)
        return outputs

    def call(self, inputs):
        embedding = self.encode(inputs)
        outputs = self.decode(embedding)
        return outputs
