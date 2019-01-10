import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Lambda, Layer
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, History
from tensorflow.contrib.tpu.python.tpu import keras_support

from keras.datasets import mnist
from PIL import Image
import os, json
import numpy as np

def reparameterize(inputs):
    # Connect a random number entered from input with skip-connection and adopt it at the time of test
    mu, logvar, skip = inputs[0], inputs[1], inputs[2]
    std = K.exp(0.5*logvar)
    eps = tf.random_normal(tf.shape(std))
    output = eps * std + mu
    return K.in_train_phase(output, skip)

def kld(inputs):
    mu, logvar = inputs[0], inputs[1]
    kld = -0.5 * K.sum(1 + logvar - mu **2 - K.exp(logvar), axis=-1, keepdims=True)
    return kld

def create_model():
    input_true = Input((28,28))
    input_rand = Input((64,))

    # encoder
    x = Flatten()(input_true)
    x = Dense(128, activation="relu")(x)
    logvar = Dense(64)(x)
    mu = Dense(64)(x)
    enc_reparam = Lambda(reparameterize)([mu, logvar, input_rand]) # randをskipconnectionにする
    enc_kld = Lambda(kld)([mu, logvar])

    # decoder
    x = Dense(128, activation="relu")(enc_reparam)
    x = Dense(784, activation="sigmoid")(x)
    # output-kld skip-connection
    x = Concatenate()([x, enc_kld])

    return Model([input_true, input_rand], x)


class Sampling(Callback):
    def __init__(self, model):
        self.model = model

    def tile_images(self, stacked_images):
        n = int(np.sqrt(stacked_images.shape[0]))
        assert stacked_images.shape[0] == n**2
        height = stacked_images.shape[1]
        width = stacked_images.shape[2]
        result = np.zeros((height*n, width*n), dtype=np.float32)
        for i in range(stacked_images.shape[0]):
            ind_y = i // n
            ind_x = i % n
            result[ind_y*height:(ind_y+1)*height, ind_x*width:(ind_x+1)*width] = stacked_images[i]
        result = (result * 255.0).astype(np.uint8)
        return result

    def on_epoch_end(self, epoch, logs):
        if not os.path.exists("sampling"):
            os.mkdir("sampling")
        dummy = np.zeros((64, 28, 28))
        rand = np.random.randn(64, 64)
        stacked_sampling = self.model.predict([dummy, rand])[:, :784].reshape(-1, 28, 28)
        output_image = self.tile_images(stacked_sampling)
        with Image.fromarray(output_image) as img:
            img.save(f"sampling/{epoch:03}.png")

def loss_function(y_true, y_pred):
    # 0-783 image, 784:kld
    bce = K.sum(K.binary_crossentropy(y_true[:,:784], y_pred[:,:784]), axis=-1)
    return bce + y_pred[:,784] # bce + kld

def train():
    (X_train, _), (_, _) = mnist.load_data()
    
    X_train = X_train / 255.0

    model = create_model()
    model.compile(tf.train.AdamOptimizer() , loss_function)
    model.summary()

    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    cb = Sampling(model)
    hist = History()
    dummy_rand = np.zeros((X_train.shape[0], 64))
    y_train = np.concatenate((X_train.reshape(-1, 784), np.zeros((X_train.shape[0], 1))), axis=-1)

    model.fit([X_train, dummy_rand], y_train, batch_size=1024, callbacks=[cb, hist], epochs=20)

    history = hist.history
    with open("vae_history.json", "w") as fp:
        json.dump(history, fp)

if __name__ == "__main__":
    K.clear_session()
    train()
