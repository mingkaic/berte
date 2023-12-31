import tensorflow as tf

from common.builder import Builder

def _linear_attention(query, key, value, scale):
    query = tf.nn.softmax(query, axis=-2)
    key = tf.nn.softmax(key, axis=-1)

    query = query * scale
    context = tf.matmul(key, value, transpose_b=True)
    out = tf.matmul(context, query, transpose_a=True)
    return out

def _attention(query, key, value, scale):
    query = query * scale

    sim = tf.matmul(query, key, transpose_a=True)
    sim = sim - tf.math.reduce_max(sim, axis=-1, keepdims=True)
    attn = tf.nn.softmax(sim, axis=-1)
    out = tf.matmul(attn, value, transpose_b=True)
    return out

def attention_init_builder():
    """ create Builder for attention or linear_attention init """
    return Builder(['num_heads', 'attention_dim', 'strategy'],
            default_values={'num_heads': 4, 'attention_dim': 128,
                'strategy': 'attention'})

class ConvAttention(tf.keras.layers.Layer):
    """
    ConvAttention for images
    """
    def __init__(self, model_dim, params):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = params['num_heads']
        self.params = params

        attention_dim = params['attention_dim']
        assert attention_dim % self.num_heads == 0
        dim_head = attention_dim / self.num_heads
        self.scale = dim_head**-0.5

        self.qkv = tf.keras.layers.Conv2D(attention_dim * 3, 1, use_bias=False)
        self.out = tf.keras.Sequential([
            tf.keras.layers.Conv2D(model_dim, 1),
            tf.keras.layers.GroupNormalization(1),
        ])

    def call(self, inputs):
        """
        Model call implementation
        """
        batch_size, height, width, channels = inputs.shape
        qkv = tf.split(self.qkv(inputs), 3, -1)
        query, key, value = tuple([
            tf.reshape(tf.transpose(tens, [0, 3, 1, 2]),
                (batch_size, self.num_heads, -1, height * width))
            for tens in qkv])

        if self.params['strategy'] == 'attention':
            out = _attention(query, key, value, self.scale)
        elif self.params['strategy'] == 'linear_attention':
            out = _linear_attention(query, key, value, self.scale)
        else:
            raise NotImplementedError

        out = tf.transpose(tf.reshape(out,
            (batch_size, -1, height, width)), [0, 2, 3, 1])
        return self.out(out)

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_dim': self.model_dim,
            'params': self.params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
