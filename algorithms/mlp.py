import tensorflow as tf

class MLP(tf.keras.Model):
    def __init__(self, units_list):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        layers = []
        for i in range(len(units_list)):
            layers.append(tf.keras.layers.Dense(units=units_list[i]))
            act = 'relu' if i < len(units_list)-1 else 'softmax'
            layers.append(tf.keras.layers.Activation(act))
        self.seq = tf.keras.Sequential(layers)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.seq(x)
        output = x
        return output
