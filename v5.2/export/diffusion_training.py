import tensorflow as tf

from tqdm.auto import tqdm

from common.training import CustomSchedule
from common.schedule import linear_beta_schedule

def extract(a, t, x_shape):
    out = tf.gather(a, t, axis=-1)
    for _ in range(len(x_shape) - 1):
        out = tf.expand_dims(out, -1)
    return out

class UnetTrainer:
    def __init__(self, model, timesteps_limit=200):
        self.timesteps_limit = timesteps_limit

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

        self.model = model

        learning_rate = CustomSchedule(model.dim)
        self.optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9)

        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

    # forward diffusion
    def q_sample(self, x, t, noise=None):
        if noise is None:
            noise = tf.random.normal(x.shape)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        return (sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise)

    def p_losses(self, image,
                 noise=None,
                 loss_f=tf.keras.losses.Huber()):
        timestamp = tf.random.uniform([image.shape[0]], 0, self.timesteps_limit,
                dtype=tf.dtypes.int64)
        (noisy_image, noise) = self.q_sample(image, t=timestamp, noise=noise)
        predicted_noise = self.model(noisy_image, training=True)
        return loss_f(noise, predicted_noise)

    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x) / sqrt_one_minus_alphas_cumprod_t)

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
