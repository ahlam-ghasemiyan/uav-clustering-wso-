# classification/capsnet.py
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np

# 1. Utils
def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1) + K.epsilon())

# 2. PrimaryCaps Layer (Fixed)
class PrimaryCaps(layers.Layer):
    def __init__(self, num_capsules=8, dim_capsule=8, kernel_size=9, strides=2, padding='valid', **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
        # Convolution layer
        self.conv = layers.Conv2D(filters=num_capsules * dim_capsule,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  activation='relu',
                                  name='primary_conv2d')
        
        # Reshape layer specifically to handle dynamic/static shape inference correctly
        self.reshape_layer = layers.Reshape((-1, dim_capsule), name='primary_reshape')

    def call(self, inputs):
        x = self.conv(inputs)
        return self.reshape_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_capsules': self.num_capsules,
            'dim_capsule': self.dim_capsule,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config

# 3. Capsule Layer (DigitCaps)
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        # input_shape: (batch, input_num_capsules, input_dim_capsule)
        # We need input_num_capsules to be defined (not None)
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsules, input_dim_capsule]"
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Weight matrix
        self.W = self.add_weight(shape=[self.num_capsules, self.input_num_capsules,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer='glorot_uniform',
                                 name='W')
        self.built = True

    def call(self, inputs):
        # inputs.shape=[None, input_num_capsules, input_dim_capsule]
        # inputs_expanded.shape=[None, 1, input_num_capsules, input_dim_capsule, 1]
        inputs_expanded = tf.expand_dims(tf.expand_dims(inputs, 1), -1)

        # Replicate inputs for each output capsule
        # inputs_tiled.shape=[None, num_capsules, input_num_capsules, input_dim_capsule, 1]
        inputs_tiled = tf.tile(inputs_expanded, [1, self.num_capsules, 1, 1, 1])

        # Replicate weights for each batch
        # W_tiled.shape=[None, num_capsules, input_num_capsules, dim_capsule, input_dim_capsule]
        W_tiled = tf.expand_dims(self.W, 0)
        W_tiled = tf.tile(W_tiled, [tf.shape(inputs)[0], 1, 1, 1, 1])

        # Compute u_hat via matrix multiplication
        # [None, num_caps, input_caps, dim_caps, input_dim] x [None, num_caps, input_caps, input_dim, 1]
        # -> [None, num_caps, input_caps, dim_caps, 1]
        inputs_hat = tf.matmul(W_tiled, inputs_tiled)
        
        # Remove last dim -> [None, num_caps, input_caps, dim_caps]
        inputs_hat = tf.squeeze(inputs_hat, axis=-1)

        # Routing algorithm
        # b.shape = [None, num_capsules, input_num_capsules]
        b = tf.zeros(shape=[tf.shape(inputs)[0], self.num_capsules, self.input_num_capsules])

        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            
            # c_expanded.shape = [None, num_caps, input_caps, 1]
            c_expanded = tf.expand_dims(c, -1)
            
            # outputs.shape = [None, num_caps, dim_caps]
            outputs = squash(tf.reduce_sum(c_expanded * inputs_hat, axis=2))

            if i < self.routings - 1:
                # outputs_expanded.shape = [None, num_caps, 1, dim_caps]
                outputs_expanded = tf.expand_dims(outputs, 2)
                
                # agreement.shape = [None, num_caps, input_caps]
                agreement = tf.reduce_sum(outputs_expanded * inputs_hat, axis=-1)
                b += agreement

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_capsules': self.num_capsules,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        })
        return config

# -------------------------
# 4. Loss Function
# -------------------------
def margin_loss(y_true, y_pred):
    """
    y_true: [None, n_classes]
    y_pred: [None, n_classes]
    """
    m_plus = 0.9
    m_minus = 0.1
    lambda_val = 0.5
    
    L_present = y_true * tf.square(tf.maximum(0., m_plus - y_pred))
    L_absent = lambda_val * (1 - y_true) * tf.square(tf.maximum(0., y_pred - m_minus))
    
    return tf.reduce_mean(tf.reduce_sum(L_present + L_absent, axis=1))

# 5. Build Function
def build_capsnet(input_shape, n_classes, routing_iters=3):
    x_input = layers.Input(shape=input_shape)

    # 1. Conv1
    # Note: Stride 2 is used to reduce dimensionality early on for large images (like 64x64 or 256x256)
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=2, padding='valid', activation='relu', name='conv1')(x_input)

    # 2. PrimaryCaps
    primary = PrimaryCaps(num_capsules=32, dim_capsule=8, kernel_size=9, strides=2, name='primary_caps')(conv1)
    
    # 3. Squash activation for primary caps
    primary_squash = layers.Lambda(squash, name='primary_squash')(primary)

    # 4. DigitCaps
    digit_caps = CapsuleLayer(num_capsules=n_classes, dim_capsule=16, routings=routing_iters, name='digit_caps')(primary_squash)

    # 5. Output Length (Probabilities)
    out_caps = Length(name='capsnet')(digit_caps)

    # 6. Feature Extractor (Flattened Digit Caps for ESN)
    flattened_features = layers.Flatten(name='flat_features')(digit_caps)

    # Models
    train_model = models.Model(x_input, out_caps, name='CapsNet_Train')
    feature_model = models.Model(x_input, flattened_features, name='CapsNet_Features')

    return train_model, feature_model

if __name__ == "__main__":
    # Test build
    try:
        model, ext = build_capsnet((64, 64, 3), 21)
        model.summary()
        print("\nSUCCESS: CapsNet built successfully. Input shape (64,64,3) propagated correctly.")
    except Exception as e:
        print(f"\nERROR: {e}")