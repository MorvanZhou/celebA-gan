import tensorflow as tf
from tensorflow import keras
import numpy as np
from cnn import dc_d, dc_g, resnet_g, resnet_d


class ACGANgp(keras.Model):
    def __init__(self, latent_dim, label_dim, img_shape, lambda_=10, summary_writer=None,
                 lr=0.0001, beta1=0., beta2=0.9, net="dcnet", norm="batch"):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.img_shape = img_shape
        self.lambda_ = lambda_
        self.net_name = net
        self.norm = norm
        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(lr, beta_1=beta1, beta_2=beta2)
        self.loss_class = lambda q, p: -tf.reduce_mean(q * 0.9 * tf.math.log(p * 0.9 + 0.05) + 0.9*(1-q) * tf.math.log(1-(p * 0.9 + 0.05)))

        self.summary_writer = summary_writer
        self._train_step = 0

    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            target_labels, noise = inputs
        else:
            target_labels = inputs
            noise = tf.random.normal((len(target_labels), self.latent_dim))
        if isinstance(target_labels, np.ndarray):
            target_labels = tf.convert_to_tensor(target_labels, dtype=tf.int32)
        return self.g.call([noise, target_labels], training=training)

    def _get_discriminator(self):
        net = dc_d if self.net_name == "dcnet" else resnet_d
        img = keras.Input(shape=self.img_shape)
        s = keras.Sequential([
            net(norm=self.norm),
        ], name="s")
        o = s(img)
        o_ls = keras.layers.Conv2D(1, 3, 2, "valid")(o)
        o_class = keras.layers.Flatten()(o_ls)
        o_class = tf.nn.sigmoid(keras.layers.Dense(self.label_dim)(o_class))
        model = keras.Model(img, [o_ls, o_class], name="discriminator")
        model.summary()
        return model

    def _get_generator(self):
        noise = keras.Input(shape=(self.latent_dim,))
        label = keras.Input(shape=(self.label_dim,), dtype=tf.float32)
        label_emb_dim = self.latent_dim // 7
        label_emb = keras.layers.Dense(label_emb_dim)(label)
        model_in = tf.concat((noise, label_emb), axis=1)
        net = dc_g if self.net_name == "dcnet" else resnet_g
        s = net((self.latent_dim+label_emb_dim,), norm=self.norm)
        o = s(model_in)
        model = keras.Model([noise, label], o, name="generator")
        model.summary()
        return model

    # gradient penalty
    def gp(self, real_img, fake_img):
        e = tf.random.uniform((len(real_img), 1, 1, 1), 0, 1)
        noise_img = e * real_img + (1. - e) * fake_img  # extend distribution space
        with tf.GradientTape() as tape:
            tape.watch(noise_img)
            o = self.d(noise_img)
        g = tape.gradient(o, noise_img)  # image gradients
        g_norm2 = tf.sqrt(tf.reduce_sum(tf.square(g), axis=[1, 2, 3]))  # norm2 penalty
        gp = tf.square(g_norm2 - 1.)
        return tf.reduce_mean(gp)

    @staticmethod
    def w_distance(real, fake):
        # the distance of two data distributions
        return tf.reduce_mean(real) - tf.reduce_mean(fake)

    def train_d(self, img, img_label):
        img_label = tf.cast(img_label, tf.float32)
        with tf.GradientTape() as tape:
            g_img = self.call(img_label, training=False)
            gp = self.gp(img, g_img)
            pred_fake, fake_class = self.d.call(g_img, training=True)
            pred_real, real_class = self.d.call(img, training=True)
            loss_class = (self.loss_class(img_label, fake_class) + self.loss_class(img_label, real_class))/2
            w_distance = -self.w_distance(pred_real, pred_fake)  # maximize W distance
            gp_loss = self.lambda_ * gp
            loss = gp_loss + loss_class + w_distance
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))

        if self._train_step % 100 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("d/w_distance", w_distance, step=self._train_step)
                tf.summary.scalar("d/gp", gp_loss, step=self._train_step)
                tf.summary.scalar("d/class", loss_class, step=self._train_step)
                tf.summary.histogram("d/pred_real", pred_real, step=self._train_step)
                tf.summary.histogram("d/pred_fake", pred_fake, step=self._train_step)
                tf.summary.histogram("d/last_grad", grads[-1], step=self._train_step)
                tf.summary.histogram("d/first_grad", grads[0], step=self._train_step)
        return w_distance, gp_loss, loss_class

    def train_g(self, batch_size):
        random_img_label = tf.convert_to_tensor(
            np.random.choice([0, 1], (batch_size, self.label_dim), replace=True), dtype=tf.float32)
        with tf.GradientTape() as tape:
            g_img = self.call(random_img_label, training=True)
            pred_fake, pred_class = self.d.call(g_img, training=False)
            loss_class = self.loss_class(random_img_label, pred_class)
            w_distance = tf.reduce_mean(-pred_fake)  # minimize W distance
            loss = w_distance + loss_class
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))

        if self._train_step % 100 == 0 and self.summary_writer is not None:
            with self.summary_writer.as_default():
                tf.summary.scalar("g/w_distance", w_distance, step=self._train_step)
                tf.summary.scalar("g/class", loss_class, step=self._train_step)
                tf.summary.histogram("g/pred_fake", pred_fake, step=self._train_step)
                tf.summary.histogram("g/first_grad", grads[0], step=self._train_step)
                tf.summary.histogram("g/last_grad", grads[-1], step=self._train_step)
                if self._train_step % 500 == 0:
                    tf.summary.image("g/img", (g_img + 1) / 2, max_outputs=5, step=self._train_step)
        self._train_step += 1
        return loss
