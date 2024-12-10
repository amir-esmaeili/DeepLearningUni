import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import json
import traceback
from pathlib import Path


# Custom F1 Score metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8  # Reduced batch size
EPOCHS = 100
NUM_CLASSES = 2

# Source directories
main_dir = '../data/Corona'  # Replace with your directory path
covid_dir = os.path.join(main_dir, 'covid')
normal_dir = os.path.join(main_dir, 'normal')


# Calculate class weights
def calculate_class_weights(covid_dir, normal_dir):
    n_normal = len([f for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    n_covid = len([f for f in os.listdir(covid_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    total = n_normal + n_covid

    weights = {
        0: (total / (2 * n_normal)),  # Normal class
        1: (total / (2 * n_covid))  # COVID class
    }
    return weights, n_normal, n_covid


# Get class weights
CLASS_WEIGHTS, n_normal, n_covid = calculate_class_weights(covid_dir, normal_dir)
print(f"Class distribution - Normal: {n_normal}, COVID: {n_covid}")
print(f"Class weights: {CLASS_WEIGHTS}")


def create_data_splits(base_dir):
    splits = ['train', 'validation', 'test']
    classes = ['covid', 'normal']

    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)


def prepare_dataset():
    temp_dir = 'temp_dataset'
    os.makedirs(temp_dir, exist_ok=True)
    create_data_splits(temp_dir)

    def process_class_files(class_dir, class_name):
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Split with stratification
        train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

        # Copy files with augmentation for COVID class if needed
        for file, split in zip([train_files, val_files, test_files], ['train', 'validation', 'test']):
            for f in file:
                src = os.path.join(class_dir, f)
                dst = os.path.join(temp_dir, split, class_name, f)
                shutil.copy2(src, dst)

        return len(train_files), len(val_files), len(test_files)

    covid_counts = process_class_files(covid_dir, 'covid')
    normal_counts = process_class_files(normal_dir, 'normal')

    print("Dataset split complete:")
    print(f"COVID images - Train: {covid_counts[0]}, Validation: {covid_counts[1]}, Test: {covid_counts[2]}")
    print(f"Normal images - Train: {normal_counts[0]}, Validation: {normal_counts[1]}, Test: {normal_counts[2]}")

    return temp_dir


# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

# Validation and test data only get rescaled
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Prepare the dataset
dataset_dir = prepare_dataset()

# Create data generators
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'validation'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'test'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)


def build_model(input_size=(224, 224, 3)):
    """
    Build U-Net model for binary classification
    """
    # Encoder
    inputs = tf.keras.Input(input_size)

    # Encoder path (contracting)
    # Block 1
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = layers.Dropout(0.25)(pool1)

    # Block 2
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = layers.Dropout(0.3)(pool2)

    # Block 3
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = layers.Dropout(0.35)(pool3)

    # Block 4
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = layers.Dropout(0.4)(pool4)

    # Bridge
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    conv5 = layers.Dropout(0.5)(conv5)

    # Decoder path (expanding)
    # Block 6
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = layers.concatenate([up6, conv4])
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    conv6 = layers.Dropout(0.4)(conv6)

    # Block 7
    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    conv7 = layers.Dropout(0.35)(conv7)

    # Block 8
    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    conv8 = layers.Dropout(0.3)(conv8)

    # Block 9
    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = layers.Dropout(0.25)(conv9)

    # Classification block
    gap = layers.GlobalAveragePooling2D()(conv9)
    dense1 = layers.Dense(256, activation='relu')(gap)
    dense1 = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(128, activation='relu')(dense1)
    dense2 = layers.Dropout(0.3)(dense2)
    outputs = layers.Dense(1, activation='sigmoid')(dense2)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model


def build_alternative_model():
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    return model


# Callbacks
checkpoint = ModelCheckpoint(
    filepath='best_model_weights.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Modified callbacks for U-Net
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,  # Increased patience
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,  # Increased patience
    min_lr=1e-7,
    verbose=1
)


def plot_training_history(history_dict):
    # Get all metrics except loss and val_loss
    metrics = [key for key in history_dict.keys()
               if not key.startswith('val_') and key != 'loss']

    # Remove 'lr' from metrics as it doesn't have validation data
    if 'lr' in metrics:
        metrics.remove('lr')

    n_metrics = len(metrics) + 1  # +1 for loss
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])  # Ensure axes is always 2D
    axes = axes.ravel()

    # Plot loss
    axes[0].plot(history_dict['loss'])
    if 'val_loss' in history_dict:
        axes[0].plot(history_dict['val_loss'])
        axes[0].legend(['Train', 'Validation'])
    else:
        axes[0].legend(['Train'])
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    # Plot other metrics
    for i, metric in enumerate(metrics, 1):
        axes[i].plot(history_dict[metric])
        val_metric = f'val_{metric}'
        if val_metric in history_dict:
            axes[i].plot(history_dict[val_metric])
            axes[i].legend(['Train', 'Validation'])
        else:
            axes[i].legend(['Train'])
        axes[i].set_title(f'Model {metric.replace("_", " ").title()}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.replace('_', ' ').title())

    # Plot learning rate separately if it exists
    if 'lr' in history_dict:
        plt.figure(figsize=(10, 4))
        plt.plot(history_dict['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')  # Use log scale for learning rate
        plt.show()

    # Remove empty subplots
    if len(metrics) + 1 < len(axes):
        for i in range(len(metrics) + 1, len(axes)):
            fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_generator):
    predictions = []
    labels = []

    test_generator.reset()

    # Get predictions
    for i in range(len(test_generator)):
        x, y = test_generator[i]
        pred = model.predict(x)
        predictions.extend(pred)
        labels.extend(y)

        if len(labels) >= len(test_generator.labels):
            break

    predictions = np.array(predictions)
    labels = np.array(labels[:len(test_generator.labels)])

    # Convert predictions to binary
    y_pred = (predictions > 0.5).astype(int)
    y_true = labels

    # Evaluate model and get metrics
    metrics = model.evaluate(test_generator, verbose=1)
    metrics_names = model.metrics_names

    # Print all metrics
    print("\nTest Results:")
    for name, value in zip(metrics_names, metrics):
        print(f"{name}: {value:.4f}")

    # Calculate additional metrics
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'COVID']))

    return y_true, y_pred


# Modified create_ensemble function
def create_ensemble():
    models = [
        build_model(),  # First U-Net model
        build_model()  # Second U-Net model with different initialization
    ]

    ensemble_predictions = []

    for i, model in enumerate(models):
        print(f"\nTraining model {i + 1}/{len(models)}")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                F1Score(name='f1_score')
            ]
        )

        # Add model summary
        print(f"\nModel {i + 1} Summary:")
        model.summary()

        # Create callbacks
        model_callbacks = [
            ModelCheckpoint(
                filepath=f'best_model_{i + 1}_weights.h5',
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            early_stopping,
            reduce_lr
        ]

        # Train the model
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            class_weight=CLASS_WEIGHTS,
            callbacks=model_callbacks,
            verbose=1
        )

        # Plot training history for each model
        print(f"\nPlotting training history for Model {i + 1}")
        plot_training_history(history.history)

        # Make predictions
        print(f"\nMaking predictions with Model {i + 1}")
        predictions = model.predict(test_generator)
        ensemble_predictions.append(predictions)

        # Evaluate individual model
        print(f"\nEvaluating Model {i + 1}")
        evaluate_model(model, test_generator)

    # Average predictions
    final_predictions = np.mean(ensemble_predictions, axis=0)
    return final_predictions

class LearningRateMonitor(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            return
        logs = logs or {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        logs['lr'] = lr

# Update callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

# Add the LearningRateMonitor to your callbacks in create_ensemble
model_callbacks = [
    ModelCheckpoint(...),
    early_stopping,
    reduce_lr,
    LearningRateMonitor()
]
# # Modified evaluate_model function to include F1 score calculation
# def evaluate_model(model, test_generator):
#     predictions = []
#     labels = []
#
#     test_generator.reset()
#
#     for i in range(len(test_generator)):
#         x, y = test_generator[i]
#         pred = model.predict(x)
#         predictions.extend(pred)
#         labels.extend(y)
#
#         if len(labels) >= len(test_generator.labels):
#             break
#
#     predictions = np.array(predictions)
#     labels = np.array(labels[:len(test_generator.labels)])
#
#     y_pred = (predictions > 0.5).astype(int)
#     y_true = labels
#
#     # Calculate metrics
#     test_loss, test_accuracy, test_auc, test_precision, test_recall = model.evaluate(test_generator)
#
#     # Calculate F1 score manually
#     f1 = f1_score(y_true, y_pred)
#
#     print("\nTest Results:")
#     print(f"Loss: {test_loss:.4f}")
#     print(f"Accuracy: {test_accuracy:.4f}")
#     print(f"AUC: {test_auc:.4f}")
#     print(f"Precision: {test_precision:.4f}")
#     print(f"Recall: {test_recall:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#
#     # Plot confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.show()
#
#     # Print classification report
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=['Normal', 'COVID']))
#
#     return y_true, y_pred


# Main execution
try:
    print("Starting ensemble training...")
    ensemble_predictions = create_ensemble()

    # Evaluate ensemble predictions
    y_true = test_generator.classes
    y_pred_ensemble = (ensemble_predictions > 0.5).astype(int)

    print("\nEnsemble Model Results:")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_ensemble, target_names=['Normal', 'COVID']))

    # Plot confusion matrix for ensemble
    cm = confusion_matrix(y_true, y_pred_ensemble)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Ensemble Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

except Exception as e:
    print(f"An error occurred: {str(e)}")
    traceback.print_exc()

finally:
    # Clean up temporary directory
    try:
        shutil.rmtree(dataset_dir)
        print("Temporary directory cleaned up successfully")
    except:
        print("Error while deleting temporary directory")