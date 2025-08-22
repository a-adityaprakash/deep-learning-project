"""
Task 2 — Deep Learning for Image Classification (TensorFlow/Keras)
------------------------------------------------------------------
Implements a compact CNN for classifying handwritten digits using the
scikit-learn 'digits' dataset (8x8 grayscale images). Includes:
- Data preprocessing and train/test split
- TensorFlow/Keras CNN model
- Training with validation
- Visualizations: learning curves, confusion matrix, sample predictions
- Saved artifacts: trained model, plots, classification report

Author: [Your Name]
Project: Internship—Deep Learning Deliverable
Date: [YYYY-MM-DD]
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def prepare_data(test_size=0.2, val_split=0.2, random_state=42):
    """
    Load and preprocess the sklearn digits dataset.
    Returns: (x_train, y_train, x_test, y_test), n_classes
    """
    digits = load_digits()
    X = digits.images  # shape (n_samples, 8, 8)
    y = digits.target  # shape (n_samples,)

    # Normalize to [0,1]
    X = X.astype("float32") / 16.0  # pixel values are 0..16

    # Add channel dimension (H, W, C)
    X = np.expand_dims(X, axis=-1)  # (n, 8, 8, 1)

    # Stratified split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    n_classes = len(np.unique(y))
    return (x_train, y_train, x_test, y_test), n_classes


def build_model(n_classes: int, input_shape=(8, 8, 1)) -> keras.Model:
    """
    Define a compact CNN suitable for 8x8 images.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="digits_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, out_dir):
    """
    Plot training and validation loss/accuracy.
    One chart per figure (per instructions).
    """
    # Accuracy plot
    plt.figure()
    plt.plot(history.epoch, history.history["accuracy"], label="train_acc")
    plt.plot(history.epoch, history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    acc_path = os.path.join(out_dir, "training_accuracy.png")
    plt.savefig(acc_path, bbox_inches="tight", dpi=150)
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history.epoch, history.history["loss"], label="train_loss")
    plt.plot(history.epoch, history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    loss_path = os.path.join(out_dir, "training_loss.png")
    plt.savefig(loss_path, bbox_inches="tight", dpi=150)
    plt.close()

    return acc_path, loss_path


def plot_confusion_matrix(y_true, y_pred, out_dir):
    """
    Confusion matrix as an image (no seaborn).
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close()
    return cm_path


def plot_sample_predictions(x, y_true, y_pred, out_dir, n=12):
    """
    Grid of sample predictions with predicted/true labels.
    """
    idx = np.random.choice(len(x), size=min(n, len(x)), replace=False)
    imgs = x[idx]
    y_t = y_true[idx]
    y_p = y_pred[idx]

    cols = 6
    rows = int(np.ceil(len(idx) / cols))

    plt.figure(figsize=(cols * 2, rows * 2))
    for i, k in enumerate(idx):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(x[k].squeeze(), cmap="gray")
        plt.title(f"pred={y_pred[i]} | true={y_true[i]}")
        plt.axis("off")
    sp_path = os.path.join(out_dir, "sample_predictions.png")
    plt.tight_layout()
    plt.savefig(sp_path, bbox_inches="tight", dpi=150)
    plt.close()
    return sp_path


def main(epochs=15, batch_size=32, test_size=0.2, val_split=0.2, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)

    # Data
    (x_train, y_train, x_test, y_test), n_classes = prepare_data(test_size=test_size)

    # Model
    model = build_model(n_classes=n_classes, input_shape=x_train.shape[1:])

    # Training
    history = model.fit(
        x_train, y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # Predictions
    probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # Visualizations
    acc_path, loss_path = plot_history(history, out_dir)
    cm_path = plot_confusion_matrix(y_test, y_pred, out_dir)
    sp_path = plot_sample_predictions(x_test, y_test, y_pred, out_dir)

    # Reports and model save
    report = classification_report(y_test, y_pred, digits=4)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write("Test Accuracy: {:.4f}\n".format(test_acc))
        f.write("Test Loss: {:.4f}\n\n".format(test_loss))
        f.write(report)

    model_path = os.path.join(out_dir, "digits_cnn.keras")
    model.save(model_path)

    print("[INFO] Training complete.")
    print(f"[INFO] Test Accuracy: {test_acc:.4f}  |  Test Loss: {test_loss:.4f}")
    print("[INFO] Artifacts saved to:", os.path.abspath(out_dir))
    print(" -", acc_path)
    print(" -", loss_path)
    print(" -", cm_path)
    print(" -", sp_path)
    print(" -", model_path)
    print(" -", os.path.join(out_dir, "classification_report.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digits CNN (TensorFlow/Keras)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--out_dir", type=str, default="artifacts")
    args = parser.parse_args()

    # Ensure TF does not flood logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        val_split=args.val_split,
        out_dir=args.out_dir,
    )