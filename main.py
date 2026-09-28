import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import numpy as np
import tensorflow as tf
from data_utils import make_image_datasets, preprocess_for_training, prepare_multi_modal
from classification.model_odl import ODLModel
from clustering.wso import WhiteSharkOptimizer

# -----------------------------
# 1. Load dataset
# -----------------------------
data_dir = r"D:\MASTER\EC\peresentaion\code2\UCMerced_LandUse\images"

train_ds, val_ds = make_image_datasets(data_dir, batch_size=16)
train_ds = preprocess_for_training(train_ds, augment=True)
val_ds = preprocess_for_training(val_ds, augment=False)

# -----------------------------
# 2. Build ODL feature extractor
# -----------------------------
odl = ODLModel(input_shape=(256,256,3), conv_filters=[64,128,256], lstm_units=256, dropout=0.4)

# -----------------------------
# 3. Extract multi-modal features (image + UAV)
# -----------------------------
X_train, y_train = prepare_multi_modal(train_ds, odl_model=odl, uav_feature_dim=50)
X_test, y_test   = prepare_multi_modal(val_ds, odl_model=odl, uav_feature_dim=50)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -----------------------------
# 4. Hyperparameter tuning via WSO
# -----------------------------
def objective(params):
    lr, hidden = params
    clf = odl.build_classifier(learning_rate=lr, hidden_units=int(hidden), n_classes=21, input_dim=X_train.shape[1])
    clf.fit(X_train, y_train, epochs=3, verbose=0)
    loss, acc = clf.evaluate(X_test, y_test, verbose=0)
    return -acc  # WSO minimizes, we maximize accuracy

wso = WhiteSharkOptimizer(population_size=5, max_gen=3)
best = wso.run(objective_fn=objective, verbose=True)
best_lr, best_hidden = best

# -----------------------------
# 5. Train final classifier
# -----------------------------
clf = odl.build_classifier(learning_rate=best_lr, hidden_units=int(best_hidden), n_classes=21, input_dim=X_train.shape[1])
clf.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=16)

loss, acc = clf.evaluate(X_test, y_test)
print(f"Final accuracy: {acc*100:.2f}%")
