import tensorflow as tf
from tensorflow.keras import layers, models

# Attention Block (Keras Functional compatible)
def attention_block(inputs):
    q = layers.Dense(inputs.shape[-1])(inputs)
    k = layers.Dense(inputs.shape[-1])(inputs)
    v = layers.Dense(inputs.shape[-1])(inputs)

    def attn_fn(x):
        q, k, v = x
        attn_scores = tf.matmul(q, k, transpose_b=True)
        attn_scores = attn_scores / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        return tf.matmul(attn_weights, v)

    out = layers.Lambda(attn_fn)([q, k, v])
    out = layers.GlobalAveragePooling1D()(out)
    return out

#                ODL Full Class (features + classifier)
class ODLModel:
    def __init__(self, input_shape=(256,256,3), conv_filters=[64,128,256],
                 lstm_units=256, dropout=0.4):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.feature_model = self._build_feature_extractor()

    # Build CNN + BiLSTM + Attention (returns feature vector)
    def _build_feature_extractor(self):
        inp = layers.Input(shape=self.input_shape)
        x = inp

        for i, f in enumerate(self.conv_filters):
            x = layers.Conv2D(f, kernel_size=3 if i>0 else 7, strides=2,
                              padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)

        shape = tf.keras.backend.int_shape(x)
        t = shape[1] * shape[2]
        f = shape[3]
        seq = layers.Reshape((t, f))(x)

        lstm_out = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True,
                        dropout=self.dropout)
        )(seq)

        attn = attention_block(lstm_out)
        return models.Model(inputs=inp, outputs=attn, name="ODL_feature_extractor")

    # Extract Deep Features
    def extract_features(self, images):
        return self.feature_model.predict(images)

    # Build Final Classifier (Dynamic input_dim)
    def build_classifier(self, learning_rate=1e-3, hidden_units=128, n_classes=21, input_dim=None):
        if input_dim is None:
            input_dim = self.feature_model.output_shape[-1]
        inp = layers.Input(shape=(input_dim,))
        x = layers.Dense(hidden_units, activation='relu')(inp)
        x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(n_classes, activation='softmax')(x)

        clf = models.Model(inputs=inp, outputs=out)
        clf.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return clf
