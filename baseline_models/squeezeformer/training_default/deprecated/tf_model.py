def get_model(dim = 384, head_dim = 2048):
    with strategy.scope():
        inp1 = tf.keras.Input([col_len+col_not_len])
        x = inp1

        x = Reshape1(col_len)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        groups = 1
        conv_filter = 15
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)

        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)

        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)
        x = Conv1DBlockSqueezeformer(dim,conv_filter)(x)
        x = TransformerEncoder(dim, 4, dim*4)(x)

        x = HeadDense(head_dim)(x)
        x = GLUMlp(head_dim*2, head_dim)(x)

        x_pred = tf.keras.layers.Dense(20)(x)
        x_confidence = tf.keras.layers.Dense(20)(x)

        x = Reshape2()(x_pred, x_confidence)

        model = tf.keras.Model(inp1, x)
        return model