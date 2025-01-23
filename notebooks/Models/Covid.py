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
import traceback
from pathlib import Path

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 2
NUM_CLASSES = 2
LEARNING_RATE = 1e-4

# Source directories
main_dir = '../../data/Corona'  # Replace with your directory path
covid_dir = os.path.join(main_dir, 'covid')
normal_dir = os.path.join(main_dir, 'normal')


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


# Custom Learning Rate Scheduler
class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, patience=5, factor=0.5, min_lr=1e-7):
        super(CustomLearningRateScheduler, self).__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.wait = 0
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    print(f'\nEpoch {epoch}: reducing learning rate to {new_lr}')
                    self.wait = 0


def calculate_class_weights(covid_dir, normal_dir):
    n_normal = len([f for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    n_covid = len([f for f in os.listdir(covid_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    total = n_normal + n_covid

    weights = {
        0: (total / (2 * n_normal)),  # Normal class
        1: (total / (2 * n_covid)) * 1.5  # COVID class with additional weight
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


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,            # Increased rotation range
    width_shift_range=0.4,        # Increased shift range
    height_shift_range=0.4,       # Increased shift range
    shear_range=0.4,             # Increased shear range
    zoom_range=[0.7, 1.3],       # More aggressive zoom
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.6, 1.4],  # More aggressive brightness
    channel_shift_range=50.0,     # Added channel shift
    preprocessing_function=None    # You can add custom preprocessing here
)


# Function to create additional augmented samples
def augment_data(image_generator, source_dir, target_dir, n_samples_per_image=5):
    """
    Generate augmented samples for each image in the source directory
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get list of images
    images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in images:
        # Load image
        img = tf.keras.preprocessing.image.load_img(
            os.path.join(source_dir, image_file),
            target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generate augmented images
        i = 0
        for batch in image_generator.flow(
                x,
                batch_size=1,
                save_to_dir=target_dir,
                save_prefix=f'aug_{os.path.splitext(image_file)[0]}',
                save_format='jpg'
        ):
            i += 1
            if i >= n_samples_per_image:
                break


def prepare_augmented_dataset():
    temp_dir = 'temp_dataset'
    augmented_dir = 'augmented_dataset'

    # Create directories
    for split in ['train', 'validation', 'test']:
        for cls in ['covid', 'normal']:
            os.makedirs(os.path.join(augmented_dir, split, cls), exist_ok=True)

    # Prepare initial dataset
    base_dir = prepare_dataset()

    # Augment training data
    print("Augmenting COVID training data...")
    augment_data(
        train_datagen,
        os.path.join(base_dir, 'train', 'covid'),
        os.path.join(augmented_dir, 'train', 'covid'),
        n_samples_per_image=10  # Generate 10 augmented samples per COVID image
    )

    print("Augmenting Normal training data...")
    augment_data(
        train_datagen,
        os.path.join(base_dir, 'train', 'normal'),
        os.path.join(augmented_dir, 'train', 'normal'),
        n_samples_per_image=5  # Generate 5 augmented samples per Normal image
    )

    # Copy validation and test data without augmentation
    for split in ['validation', 'test']:
        for cls in ['covid', 'normal']:
            src_dir = os.path.join(base_dir, split, cls)
            dst_dir = os.path.join(augmented_dir, split, cls)
            for file in os.listdir(src_dir):
                shutil.copy2(
                    os.path.join(src_dir, file),
                    os.path.join(dst_dir, file)
                )

    return augmented_dir


# Validation and test data only get rescaled
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)


def build_model(input_size=(224, 224, 3)):
    """
    Build an improved CNN model for binary classification
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_size),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Second Convolutional Block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Third Convolutional Block
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    return model


def plot_training_history(history_dict):
    metrics = [key for key in history_dict.keys()
               if not key.startswith('val_') and key != 'loss']

    if 'lr' in metrics:
        metrics.remove('lr')

    n_metrics = len(metrics) + 1
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.ravel()

    # Plot loss
    axes[0].plot(history_dict['loss'])
    axes[0].plot(history_dict['val_loss'])
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(['Train', 'Validation'])

    # Plot other metrics
    for i, metric in enumerate(metrics, 1):
        axes[i].plot(history_dict[metric])
        val_metric = f'val_{metric}'
        if val_metric in history_dict:
            axes[i].plot(history_dict[val_metric])
            axes[i].legend(['Train', 'Validation'])
        axes[i].set_title(f'Model {metric.replace("_", " ").title()}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.replace('_', ' ').title())

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

    # Calculate metrics
    metrics = model.evaluate(test_generator, verbose=1)
    metrics_names = model.metrics_names

    print("\nTest Results:")
    for name, value in zip(metrics_names, metrics):
        print(f"{name}: {value:.4f}")

    # Calculate F1 score
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

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'COVID']))

    return y_true, y_pred


def train_model_with_augmentation(n_folds=5):
    augmented_dir = prepare_augmented_dataset()

    # Create data generators with the augmented dataset
    train_generator = train_datagen.flow_from_directory(
        os.path.join(augmented_dir, 'train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        os.path.join(augmented_dir, 'validation'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(augmented_dir, 'test'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # Create and train the model
    model = build_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            F1Score(name='f1_score')
        ]
    )

    # Add mixup data augmentation
    def mixup(x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = tf.shape(x)[0]
        index = tf.random.shuffle(tf.range(batch_size))

        mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
        mixed_y = lam * y + (1 - lam) * tf.gather(y, index)

        return mixed_x, mixed_y

    # Custom training step with mixup
    @tf.function
    def train_step(x, y):
        x, y = mixup(x, y)
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = tf.keras.losses.binary_crossentropy(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        CustomLearningRateScheduler(patience=8, factor=0.6),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        class_weight=CLASS_WEIGHTS,
        callbacks=callbacks,
        verbose=1
    )

    return model, history, test_generator


# Main execution
try:
    print("Starting model training with augmented data...")
    model, history, test_generator = train_model_with_augmentation()

    # Plot training history
    plot_training_history(history.history)

    # Evaluate model
    print("\nEvaluating final model:")
    evaluate_model(model, test_generator)

except Exception as e:
    print(f"An error occurred: {str(e)}")
    traceback.print_exc()

finally:
    # Clean up temporary directories
    for dir_name in ['temp_dataset', 'augmented_dataset']:
        try:
            shutil.rmtree(dir_name)
            print(f"{dir_name} cleaned up successfully")
        except:
            print(f"Error while deleting {dir_name}")