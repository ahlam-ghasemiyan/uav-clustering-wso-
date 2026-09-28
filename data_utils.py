import tensorflow as tf
import numpy as np
import os

# ----------------------------
# Load dataset as tf.data.Dataset
# ----------------------------
def make_image_datasets(data_dir, image_size=(256,256), batch_size=32):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True
    )

    # Split 80/20
    n = sum(1 for _ in ds)
    train_size = int(0.8 * n)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)
    return train_ds, val_ds

# ----------------------------
# Preprocess for training
# ----------------------------
def preprocess_for_training(dataset, augment=True):
    def _augment(img, label):
        # Resize first
        img = tf.image.resize(img, [256,256])
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        # Normalize
        img = img / 255.0
        return img, label

    def _process(img, label):
        img = tf.image.resize(img, [256,256])
        img = img / 255.0
        return img, label

    if augment:
        return dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    else:
        return dataset.map(_process, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# ----------------------------
# For full pipeline: return numpy arrays + UAV features
# ----------------------------
def prepare_multi_modal(dataset, odl_model, uav_feature_dim=50):
    X_img, y = [], []
    for img_batch, labels in dataset:
        # extract image features
        img_features = odl_model.extract_features(img_batch)
        # simulate UAV features
        uav_features = np.random.normal(size=(img_features.shape[0], uav_feature_dim))
        # combine image + UAV features
        combined = np.concatenate([img_features, uav_features], axis=-1)
        X_img.append(combined)
        y.append(labels.numpy())
    X = np.vstack(X_img)
    y = np.hstack(y)
    return X, y
