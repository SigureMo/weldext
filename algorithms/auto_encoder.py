import tensorflow as tf
import numpy as np


class AutoEncoder(tf.keras.layers.Layer):
    def __init__(self, units, l2_const=1e-4, share_weights=False):
        super().__init__()
        self.units = units
        self.l2_const = l2_const
        self.share_weights = share_weights

    def build(self, input_shape):
        self.encoder = tf.keras.layers.Dense(self.units,
                                             activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.l2_const))
        if self.share_weights:
            self.decoder_w = None
        else:
            self.decoder_w = self.add_weight(name='decoder_w',
                                             shape=[self.units, input_shape[-1]], initializer=tf.random_uniform_initializer())
        self.decoder_b = self.add_weight(name='decoder_b',
                                         shape=[input_shape[-1]], initializer=tf.random_uniform_initializer())

    def encode(self, inputs):
        embedding = self.encoder(inputs)
        return embedding

    def decode(self, embedding):
        if self.share_weights:
            w = tf.transpose(self.encoder.weights[0])
        else:
            w = self.decoder_w
        outputs = tf.matmul(embedding, w) + self.decoder_b
        outputs = tf.nn.relu(outputs)
        return outputs

    def call(self, inputs):
        embedding = self.encode(inputs)
        outputs = self.decode(embedding)
        return outputs


class StackedAutoEncoder():
    def __init__(self, units_list, share_weights=False):
        self.units_list = units_list
        self.auto_encoders = []
        for units in units_list:
            self.auto_encoders.append(AutoEncoder(
                units, share_weights=share_weights))

    def __getitem__(self, index):
        return self.auto_encoders[index]

    def build(self, input_shape):
        input = np.zeros(input_shape, dtype=np.float32)
        for ae in self:
            ae(input)
            input = ae.encode(input)

    def encode(self, inputs, stop_at=None):
        embedding = inputs
        for auto_encoder in self.auto_encoders[: stop_at]:
            embedding = auto_encoder.encode(embedding)
        return embedding

    def decode(self, embedding, stop_at=None):
        outputs = embedding
        for auto_encoder in reversed(self.auto_encoders[: stop_at]):
            outputs = auto_encoder.decode(outputs)
        return outputs

    def __call__(self, inputs, stop_at=None):
        embedding = self.encode(inputs, stop_at)
        outputs = self.decode(embedding, stop_at)
        return outputs

    def appliy_noise(self, X_batch, block_density=0.05):
        X_batch_noise = X_batch.copy()
        block_mask = np.random.randint(
            0, 1//block_density, X_batch_noise.shape) > 1
        X_batch_noise *= block_mask
        return X_batch_noise

    def train(self, data_loader, train_layer=-1, num_epochs=5, learning_rate=1e-4, batch_size=64, sparse=True, denoise=True):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        @tf.function
        def train_on_batch(auto_encoder, X_batch, y_batch):

            with tf.GradientTape() as tape:
                expect_tho = 0.05
                kl_beta = 3
                emb = auto_encoder.encode(X_batch)
                y_pred = auto_encoder.decode(emb)
                tho_tensor = expect_tho * tf.ones(emb.shape[-1])
                loss = tf.reduce_mean((y_pred - y_batch) ** 2, axis=1)
                if sparse:
                    loss += kl_beta * KL_devergence(tho_tensor, emb)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, auto_encoder.variables)
            optimizer.apply_gradients(
                grads_and_vars=zip(grads, auto_encoder.variables))
            return loss

        for epoch in range(num_epochs):

            # Training
            for batch_index, (X_batch, y_batch) in enumerate(data_loader(batch_size=batch_size, data_type='train')):
                X_batch = X_batch.reshape(
                    X_batch.shape[0], X_batch.shape[1]*X_batch.shape[2])
                auto_encoder = self[train_layer]
                if denoise:
                    X_noise_batch = self.appliy_noise(X_batch)
                    X_noise_batch = self.encode(
                        X_noise_batch, stop_at=train_layer)
                    X_batch = self.encode(X_batch, stop_at=train_layer)
                    loss = train_on_batch(auto_encoder, X_noise_batch, X_batch)
                else:
                    X_batch = self.encode(X_batch, stop_at=train_layer)
                    loss = train_on_batch(auto_encoder, X_batch, X_batch)

                template = '[Training] Layer {}, Epoch {}, Batch {}/{} '
                print(template.format(train_layer,
                                      epoch+1,
                                      batch_index,
                                      data_loader.train_size // batch_size),
                      end='\r')


def KL_devergence(p, q):
    q = tf.nn.softmax(q)
    q = tf.reduce_mean(q, axis=0)
    s1 = tf.reduce_sum(p*tf.math.log(p/q))
    s2 = tf.reduce_sum((1-p)*tf.math.log((1-p)/(1-q)))
    return s1+s2


if __name__ == '__main__':
    from algorithms.utils import MNISTLoader

    data_loader = MNISTLoader()
    stacked_auto_encoder = StackedAutoEncoder(
        [1024, 512, 128], share_weights=True)
    stacked_auto_encoder.build(input_shape=(1, 784))
    stacked_auto_encoder.train(data_loader, train_layer=0, num_epochs=5,
                               learning_rate=1e-4, batch_size=256, sparse=True, denoise=True)
    stacked_auto_encoder.train(data_loader, train_layer=1, num_epochs=5,
                               learning_rate=1e-4, batch_size=256, sparse=True, denoise=True)
    stacked_auto_encoder.train(data_loader, train_layer=2, num_epochs=5,
                               learning_rate=1e-4, batch_size=256, sparse=True, denoise=True)
