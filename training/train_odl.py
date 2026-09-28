import os
import json
import numpy as np
import tensorflow as tf
from classification.model_odl import ODLModel
from data_utils import make_image_datasets, preprocess_for_training, prepare_multi_modal
from sklearn.metrics import classification_report, confusion_matrix

def train_pipeline(data_dir, out_dir, input_shape=(256,256,3), n_classes=21,
                   batch_size=32, epochs=50, dropout=0.4, lstm_units=256, conv_filters='64,128,256'):

    os.makedirs(out_dir, exist_ok=True)
    conv_filters = [int(x) for x in conv_filters.split(',')]

    # ----------------------------
    # Load and preprocess datasets
    # ----------------------------
    train_ds, val_ds = make_image_datasets(data_dir, image_size=input_shape[:2], batch_size=batch_size)
    train_ds = preprocess_for_training(train_ds, augment=True)
    val_ds = preprocess_for_training(val_ds, augment=False)

    # ----------------------------
    # Build feature extractor
    # ----------------------------
    odl = ODLModel(input_shape=input_shape, conv_filters=conv_filters,
                   lstm_units=lstm_units, dropout=dropout)
    print("ODL feature extractor ready.")

    # ----------------------------
    # Prepare multi-modal features
    # ----------------------------
    X_train, y_train = prepare_multi_modal(train_ds, odl)
    X_val, y_val = prepare_multi_modal(val_ds, odl)
    print(f"Features prepared: train {X_train.shape}, val {X_val.shape}")

    # ----------------------------
    # Simulate WSO hyperparameter tuning (short search)
    # ----------------------------
    hidden_options = [128, 256]
    lr_options = [1e-3, 5e-4]
    best_acc = 0
    best_lr = 1e-3
    best_hidden = 128

    for h in hidden_options:
        for l in lr_options:
            clf = odl.build_classifier(learning_rate=l, hidden_units=h, n_classes=n_classes)
            clf.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=3, batch_size=batch_size, verbose=0)
            val_acc = clf.evaluate(X_val, y_val, verbose=0)[1]
            if val_acc > best_acc:
                best_acc = val_acc
                best_lr = l
                best_hidden = h

    print(f"WSO simulated: best_lr={best_lr}, best_hidden={best_hidden}, val_acc={best_acc:.4f}")

    # ----------------------------
    # Train final classifier
    # ----------------------------
    clf = odl.build_classifier(learning_rate=best_lr, hidden_units=best_hidden, n_classes=n_classes)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, 'best_model.h5'), monitor='val_accuracy', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]

    history = clf.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    # Save history
    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump({k: [float(x) for x in v] for k,v in history.history.items()}, f, indent=2)

    # Evaluate
    y_pred = np.argmax(clf.predict(X_val), axis=1)
    print(classification_report(y_val, y_pred))
    cm = confusion_matrix(y_val, y_pred)
    np.save(os.path.join(out_dir, 'confusion_matrix.npy'), cm)
    clf.save(os.path.join(out_dir, 'final_model.h5'))

    print("Training complete. Model and reports saved.")
    return history, clf

# ----------------------------
# Run pipeline
# ----------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_dir', default='out')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--lstm_units', type=int, default=256)
    parser.add_argument('--conv_filters', type=str, default='64,128,256')
    args = parser.parse_args()

    train_pipeline(args.data_dir, args.out_dir, batch_size=args.batch,
                   epochs=args.epochs, dropout=args.dropout,
                   lstm_units=args.lstm_units, conv_filters=args.conv_filters)
