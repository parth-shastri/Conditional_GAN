import tensorflow as tf
from keras import layers, Model, Input
from keras.utils import Progbar, to_categorical
from keras.datasets.mnist import load_data
import numpy as np
import matplotlib.pyplot as plt
import config
import datetime

img_height, img_width, _ = config.IMAGE_SHAPE

(X, Y), (_, _) = load_data()
X = X.reshape((-1, img_height, img_width, 1))
X = X.astype("float32")
Y = to_categorical(Y, num_classes=10, dtype="float32")


def preprocess(img, lbl):
    img = (img - 127.5) / 127.5
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    return img, lbl


class Generator(Model):
    def __init__(self, name):
        super(Generator, self).__init__(name=name)
        self.dense = layers.Dense(7*7*128)
        self.conv1 = layers.Conv2DTranspose(128, kernel_size=5, padding="same")
        self.conv2 = layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same")
        self.conv3 = layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding="same")
        self.conv4 = layers.Conv2DTranspose(1, kernel_size=5, activation="tanh", padding="same")
        self.relu = layers.ReLU()
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):

        noise, label = inputs
        x = layers.Concatenate()([noise, label])
        x = self.dense(x)
        x = layers.Reshape(target_shape=(7, 7, 128))(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv4(x)

        return x

    def get_config(self):
        return {'name': self.name}


class Discriminator(Model):
    def __init__(self, name, img_shape=(28, 28, 1)):
        super(Discriminator, self).__init__(name=name)
        self.img_shape = img_shape
        self.conv1 = layers.Conv2D(32, kernel_size=5, strides=2)
        self.conv2 = layers.Conv2D(64, kernel_size=5, strides=2)
        self.conv3 = layers.Conv2D(128, kernel_size=5, strides=2, padding="same")
        self.conv4 = layers.Conv2D(256, kernel_size=5, padding="same")
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)
        self.flatten = layers.Flatten()
        self.dense_final = layers.Dense(1, activation='sigmoid')
        self.dense = layers.Dense(7*7*16)

    def call(self, inputs, training=None, mask=None):

        image, label = inputs
        lb = self.dense(label)
        lb = layers.Reshape(target_shape=(28, 28, 1))(lb)
        x = layers.Concatenate()([image, lb])
        x = self.leaky_relu(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense_final(x)

        return x

    def get_config(self):
        return {"img_shape": self.img_shape, "name": self.name}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


gen = Generator(name="generator")
disc = Discriminator(name="discriminator", img_shape=config.IMAGE_SHAPE)

gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
train_dataset = dataset.take(int(0.8 * len(X))).map(preprocess).shuffle(10000).batch(config.BATCH_SIZE)
val_dataset = dataset.skip(int(0.8 * len(X))).map(preprocess).shuffle(10000).batch(config.BATCH_SIZE)

checkpoint = tf.train.Checkpoint(generator=gen,
                                 gen_optimizer=gen_optimizer,
                                 discriminator=disc,
                                 disc_optimizer=disc_optimizer)
ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=config.CKPT_DIR, max_to_keep=3)

# creates a summary writer, writes a summary in a file to access on tensorboard later
summary_writer = tf.summary.create_file_writer(
                        logdir=config.LOG_DIR + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

'''LOSSES'''


def disc_loss(real_logits, fake_logits):
    real_loss = tf.losses.BinaryCrossentropy()(tf.ones_like(real_logits), real_logits)
    fake_loss = tf.losses.BinaryCrossentropy()(tf.zeros_like(fake_logits), fake_logits)
    loss = 0.5*(real_loss + fake_loss)
    return loss


def gen_loss(fake_logits):
    loss = tf.losses.BinaryCrossentropy()(tf.ones_like(fake_logits), fake_logits)
    return loss


# give signature to avoid retracing

signature = [
    tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int64)
]


@tf.function(input_signature=signature)
def train_step(image_batch, label_batch, epoch):
    noise = tf.random.normal((config.BATCH_SIZE, config.NOISE_DIM))
    with tf.GradientTape(persistent=True) as tape:

        fake_img_batch = gen([noise, label_batch], training=True)
        fake_logits = disc([fake_img_batch, label_batch], training=True)
        real_logits = disc([image_batch, label_batch], training=True)

        d_loss = disc_loss(real_logits, fake_logits)
        g_loss = gen_loss(fake_logits)

    gen_grads = tape.gradient(g_loss, gen.trainable_variables)
    disc_grads = tape.gradient(d_loss, disc.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, gen.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_grads, disc.trainable_variables))

    # writes a tensorboard summary (creates graph if scalar)
    with summary_writer.as_default():
        tf.summary.scalar("generator_loss", g_loss, step=epoch)
        tf.summary.scalar("discriminator_loss", d_loss, step=epoch)


g_loss = tf.metrics.Mean()
d_loss = tf.metrics.Mean()
prog_bar = Progbar(1500, stateful_metrics=[g_loss, d_loss])

if ckpt_manager.latest_checkpoint:
    checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print(f"Restored the training checkpoint...{ckpt_manager.latest_checkpoint}")


def train():
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS} :")
        for n, (image, label) in enumerate(train_dataset):
            train_step(image, label, epoch+1)
            prog_bar.update(n)

        if (epoch+1) % 5 == 0:
            ckpt_manager.save()


def generate():
    z = tf.random.normal((10, config.NOISE_DIM))
    indices = np.arange(0, 10)
    labels = tf.one_hot(indices, depth=10)
    print(labels)

    out = gen([z, labels])
    out = (out.numpy() * 127.5) + 127.5  # de-process
    for i in range(10):
        plt.subplot(1, 10, i + 1)
        plt.axis("off")
        plt.imshow(out[i].reshape((img_height, img_width)), cmap='gray')
    plt.show()


if __name__ == "__main__":
    train()   # train loop

    '''Test Code'''

    # gen_out = gen([tf.random.normal((config.BATCH_SIZE, config.NOISE_DIM)),
    #                tf.ones((config.BATCH_SIZE, 10))])
    # disc_out = disc([tf.random.normal((config.BATCH_SIZE,) + config.IMAGE_SHAPE),
    #                 tf.ones((config.BATCH_SIZE, 10))])
    #
    # assert gen_out.shape == (32, 28, 28, 1)







