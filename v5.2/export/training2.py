import tensorflow as tf

from tqdm.auto import tqdm

from common.training import CustomSchedule
from common.schedule import linear_beta_schedule

from export.training import NAN_LOSS_ERR_CODE

def extract(a, t, x_shape):
    out = tf.gather(a, t, axis=-1)
    for _ in range(len(x_shape) - 1):
        out = tf.expand_dims(out, -1)
    return out

class UnetTrainer:
    def __init__(self, unet, optimizer, training_loss,
                 timesteps_limit=200,
                 loss_f=tf.keras.losses.Huber()):

        self.training_loss = training_loss

        self.timesteps_limit = timesteps_limit
        self.loss_f = loss_f

        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=timesteps_limit)

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = tf.math.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = tf.pad(self.alphas_cumprod[:-1],
                                          tf.constant([[1, 0]]),
                                          constant_values=1.0)
        self.sqrt_recip_alphas = tf.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = tf.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.unet = unet
        self.optimizer = optimizer

    def __call__(self, batch):

        debug_info = dict()
        with tf.GradientTape() as tape:
            loss = self._train_step(batch)

            if tf.math.is_nan(loss):
                debug_info['bad_batch'] = loss
                return debug_info, NAN_LOSS_ERR_CODE

            gradients = tape.gradient(loss, self.unet.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.unet.trainable_variables))
        # update metrics
        self.training_loss(loss)

        return debug_info, 0

    # generate a forward diffusion noisy image then predict through unet
    def _train_step(self, batch):
        timestamp = tf.random.uniform([batch.shape[0]], 0, self.timesteps_limit,
                tf.dtypes.int64)
        noisy_batch, noise = self._q_sample(batch, timestamp)
        predicted_noise = self.unet(noisy_batch, training=True)
        return self.loss_f(noise, predicted_noise)

    # forward diffusion
    def _q_sample(self, x, t, noise=None):
        if noise is None:
            noise = tf.random.normal(x.shape)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        return (sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise)

    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.unet(x) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean

        posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        noise = tf.random.normal(x.shape)
        # Algorithm 2 line 4:
        return model_mean + tf.sqrt(posterior_variance_t) * noise

    def sample(self, image_size, batch_size=16, channels=3):
        shape = (batch_size, image_size, image_size, channels)
        # start from pure noise (for each example in the batch)
        img = tf.random.normal(shape)
        imgs = []

        for i in tqdm(range(self.timesteps_limit-1, -1, -1),
                      desc='sampling loop time step',
                      total=self.timesteps_limit):
            img = self.p_sample(img, tf.constant(i,
                                                 dtype=tf.int64,
                                                 shape=(batch_size,)), i)
            imgs.append(img.numpy())
        return imgs
