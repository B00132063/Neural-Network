print("1. Script started") # Just to confirm the script is running
# Import basic libraries
import os                 # for file paths
print("2. os done", flush=True)

import time               # to measure training time
print("3. time done", flush=True)

import json               # to save training history
print("4. json done", flush=True)

import argparse           # to pass arguments from terminal
print("5. argparse done", flush=True)

from pathlib import Path  # easier file handling
print("6. pathlib done", flush=True)

# Import scientific and plotting libraries
import numpy as np
print("7. numpy done", flush=True)

import matplotlib.pyplot as plt
print("8. matplotlib done", flush=True)

# Import TensorFlow and Keras
import tensorflow as tf
print("9. tensorflow done", flush=True)

from tensorflow import keras
print("10. keras done", flush=True)

from tensorflow.keras import layers
print("11. layers done", flush=True)


# Import evaluation tools
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
print("12. sklearn done", flush=True)

print("13. Imports finished") # Just to confirm all libraries are imported successfully

# Performance settings
AUTOTUNE = tf.data.AUTOTUNE   # automatically imporove data loading
SEED = 123                    # random seed for reliability
IMG_SIZE = (224, 224)         # image size (better for pretrained models)
BATCH_SIZE = 16               # how many images per batch
EPOCHS = 15                   # number of training cycles

print("14. Entered main") # Just to confirm we are in the main function and ready to start processing

# ---------------------------
# Read arguments from terminal
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Pneumonia classification")

    # Paths to dataset folders
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)

    # Optional settings
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    # If used, will fine-tune pretrained model
    parser.add_argument("--fine_tune", action="store_true")

    # Output files
    parser.add_argument("--model_out", type=str, default="best_model.keras")
    parser.add_argument("--history_out", type=str, default="history.json")

    return parser.parse_args()

print("15. Counting images") # Just to confirm we are counting images in the dataset

# ---------------------------
# Count images per class
# ---------------------------
def count_images_by_class(directory):
    directory = Path(directory)
    counts = {}

    # Loop through each class folder (e.g. NORMAL, BACTERIAL)
    for class_dir in sorted([p for p in directory.iterdir() if p.is_dir()]):
        counts[class_dir.name] = len([
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
        ])

    return counts


# Print dataset distribution nicely
def print_distribution(name, counts):
    total = sum(counts.values())
    print(f"\n{name} distribution:")

    for cls, count in counts.items():
        pct = (count / total * 100)
        print(f"{cls}: {count} images ({pct:.2f}%)")

    print("Total:", total)

print("16. Loading datasets") # Just to confirm we are loading datasets and creating data pipelines

# ---------------------------
# Create datasets
# ---------------------------
def make_datasets(train_dir, test_dir, img_size, batch_size):

    print("17. Datasets loaded") # Just to confirm datasets are loaded and we are ready to create data pipelines

    # Create training dataset (80%)
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True
    )

    # Create validation dataset (20%)
    val_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True
    )

    # Test dataset (NO SHUFFLE)
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False
    )

    # Get class names (folder names)
    class_names = train_ds.class_names

    # Data augmentation (makes model more robust)
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),    # flip image
        layers.RandomRotation(0.05),        # small rotation
        layers.RandomZoom(0.1),             # zoom
        layers.RandomContrast(0.1),         # contrast
    ])

    # Improve performance using caching and prefetching
    def prepare(ds, training=False):
        ds = ds.cache()
        if training:
            ds = ds.shuffle(1000)
        return ds.prefetch(AUTOTUNE)

    return prepare(train_ds, True), prepare(val_ds), prepare(test_ds), class_names, augmentation

print("18. Building model") # Just to confirm we are building the model and ready to start training

# ---------------------------
# Build model
# ---------------------------
def build_model(num_classes, img_size, augmentation, lr, fine_tune):
    print("19. Model built") # Just to confirm the model is built and we are ready to start training

    # Input layer
    inputs = keras.Input(shape=(img_size, img_size, 3))

    # Apply augmentation
    x = augmentation(inputs)

    # Load pretrained EfficientNet model
    base_model = keras.applications.EfficientNetB0(
        include_top=False,   # remove final classification layer
        weights=None,  # pretrained weights
        input_shape=(img_size, img_size, 3)
    )

    # Freeze model unless fine-tuning
    base_model.trainable = fine_tune

    # Preprocess input for EfficientNet
    x = keras.applications.efficientnet.preprocess_input(x)

    # Pass through base model
    x = base_model(x, training=False)

    # Replace Flatten with GlobalAveragePooling (better!)
    x = layers.GlobalAveragePooling2D()(x)

    # Add normalization
    x = layers.BatchNormalization()(x)

    # Reduce overfitting
    x = layers.Dropout(0.35)(x)

    # Dense layer
    x = layers.Dense(128, activation="relu")(x)

    # More dropout
    x = layers.Dropout(0.25)(x)

    # Output layer (classification)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Build model
    model = keras.Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()]
    )

    return model


# ---------------------------
# Class weights (for imbalance)
# ---------------------------
def compute_class_weights(counts, class_names):
    total = sum(counts.values())
    n = len(class_names)

    weights = {}

    for i, cls in enumerate(class_names):
        weights[i] = total / (n * counts[cls])

    return weights


# ---------------------------
# Training graph
# ---------------------------
def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Accuracy")
    plt.legend(["train", "val"])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Loss")
    plt.legend(["train", "val"])
    plt.show()


# ---------------------------
# Main function
# ---------------------------
def main():
    args = parse_args()

    # Count images
    train_counts = count_images_by_class(args.train_dir)
    test_counts = count_images_by_class(args.test_dir)

    print_distribution("Train", train_counts)
    print_distribution("Test", test_counts)

    # Load datasets
    train_ds, val_ds, test_ds, class_names, augmentation = make_datasets(
        args.train_dir, args.test_dir, args.img_size, args.batch_size
    )

    # Compute class weights
    class_weights = compute_class_weights(train_counts, class_names)

    # Build model
    model = build_model(
        len(class_names),
        args.img_size,
        augmentation,
        args.learning_rate,
        args.fine_tune
    )

    model.summary()

    # Callbacks (important!)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(args.model_out, save_best_only=True),
    ]

    # Train model
    start = time.time()

    print("20. Starting training") # Just to confirm we are starting training

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    end = time.time()

    print("Training time:", end - start)

    # Evaluate model
    print("\nTesting model...")
    model.evaluate(test_ds)

    # Predictions
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds), axis=1)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Plot training graphs
    plot_history(history)


# Run script
if __name__ == "__main__":
    main()