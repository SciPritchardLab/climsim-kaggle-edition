class GLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x, mask=None):
        x,gate = tf.split(x, 2, axis = -1)
        x = x*tf.keras.activations.swish(gate)
        return x

class GLUMlp(tf.keras.layers.Layer):
    def __init__(self, dim_expand, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim_expand = dim_expand
        self.dim = dim
        self.dense_1 = tf.keras.layers.EinsumDense("abc,cd->abd",output_shape=(None, self.dim_expand), activation = 'linear', bias_axes = 'd')
        self.glu_1 = GLU()
        self.dense_2 = tf.keras.layers.EinsumDense("abc,cd->abd",output_shape=(None, self.dim), activation = 'linear', bias_axes = 'd')
    def call(self, x, training = False):
        x = self.dense_1(x)
        x = self.glu_1(x)
        x = self.dense_2(x)
        return x

class ScaleBias(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.scale_bias = tf.keras.layers.EinsumDense("abc,c->abc",output_shape=(None, input_shape[-1]),activation = 'linear', bias_axes = 'c')
    def call(self, x, mask=None):
        return self.scale_bias(x)

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.ffn = GLUMlp(feed_forward_dim, embed_dim)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.scale_bias_1 = ScaleBias()
        self.scale_bias_2 = ScaleBias()
    def call(self, x, training = None):
        residual = x
        x = self.att(x, x)
        x = self.scale_bias_1(x)
        x = self.layer_norm_1(x + residual)
        residual = x
        x = self.ffn(x, training = training)
        x = self.scale_bias_2(x)
        x = self.layer_norm_2(x + residual)
        return x


class ECA(tf.keras.layers.Layer):
    # TF implementation from https://www.kaggle.com/code/hoyso48/1st-place-solution-training
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)
    def call(self, inputs):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:,None,:]
        return inputs * nn


class HeadDense(tf.keras.layers.Layer):
    def __init__(self, head_dim, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
    def build(self, input_shape):
        self.length = input_shape[1]
        self.dim = input_shape[2]
        self.dense = tf.keras.layers.EinsumDense("abc,cd->abd",output_shape=(self.length, self.head_dim), activation = 'swish', bias_axes = 'd')
    def call(self, x):
        x = self.dense(x)
        return x

class Conv1DBlockSqueezeformer(tf.keras.layers.Layer):
    def __init__(self, channel_size, kernel_size, dilation_rate=1,
                 expand_ratio=4, se_ratio=0.25, activation='swish', name=None, **kwargs):
        super().__init__()
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.activation = activation
        self.scale_bias = ScaleBias()
        self.glu_layer = GLU()
        self.ffn = GLUMlp(channel_size*4, channel_size)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.scale_bias_1 = ScaleBias()
        self.scale_bias_2 = ScaleBias()
    def build(self, input_shape):
        self.length = input_shape[1]
        self.channels_in = input_shape[2]
        self.channels_expand = self.channels_in * self.expand_ratio
        self.dwconv = tf.keras.layers.DepthwiseConv1D(self.kernel_size,dilation_rate=self.dilation_rate,padding='same',use_bias=False)
        self.BatchNormalization_layer = tf.keras.layers.BatchNormalization(momentum=0.95)
        self.conv_activation = tf.keras.layers.Activation(self.activation)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ECA_layer = ECA()
        self.expand = tf.keras.layers.EinsumDense("abc,cd->abd",output_shape=(self.length, self.channels_expand), activation = 'linear', bias_axes = 'd')
        self.project =tf.keras.layers.EinsumDense("abc,cd->abd",output_shape=(self.length, self.channel_size), activation = 'linear', bias_axes = 'd')
    def call(self, x, training = None):
        skip = x
        x = self.expand(x)
        x = self.glu_layer(x)
        x = self.dwconv(x)
        x = self.BatchNormalization_layer(x)
        x = self.conv_activation(x)
        x = self.ECA_layer(x)
        x = self.project(x)
        x = self.scale_bias_1(x)

        x = x+skip

        residual = x
        x = self.ffn(x)
        x = self.scale_bias_2(x)
        x = self.layer_norm_2(x + residual)
        return x

class Reshape1(tf.keras.layers.Layer):
    def __init__(self, col_len, **kwargs):
        super().__init__(**kwargs)
        self.col_len = col_len
    def call(self, x):
        x_seq = x[:, :self.col_len]
        x_seq = tf.reshape(x_seq, [-1, tf.cast(self.col_len/60, tf.int32), 60])
        x_seq = tf.transpose(x_seq, [0, 2, 1])
        x_seq_N = x[:, self.col_len:]

        x_seq_N = tf.expand_dims(x_seq_N, axis = 1)
        x_seq_N = tf.repeat(x_seq_N, 60, axis = 1)

        x = tf.concat([x_seq, x_seq_N], axis = -1)
        return x

class Reshape2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, x_pred, x_confidence):
        x = x_pred
        x_seq = x[:, :, :6]
        x_seq = tf.transpose(x_seq, [0,2,1])
        x_seq = tf.reshape(x_seq, [-1, 60*6])
        x_seq_N = x[:, :, 6:]
        x_seq_N = tf.reduce_mean(x_seq_N, axis = 1)
        x1 = tf.concat([x_seq, x_seq_N], axis = -1)

        x = x_confidence
        x_seq = x[:, :, :6]
        x_seq = tf.transpose(x_seq, [0,2,1])
        x_seq = tf.reshape(x_seq, [-1, 60*6])
        x_seq_N = x[:, :, 6:]
        x_seq_N = tf.reduce_mean(x_seq_N, axis = 1)
        x2 = tf.concat([x_seq, x_seq_N], axis = -1)

        x = tf.concat([x1, x2], axis = -1)
        return x