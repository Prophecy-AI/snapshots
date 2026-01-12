import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Define paths
TRAIN_PATH = '/home/data/train/'
TEST_PATH = '/home/data/test/'
SAMPLE_SUBMISSION_PATH = '/home/data/sample_submission.csv'

# Get all training images
train_files = os.listdir(TRAIN_PATH)
print(f"Total training images: {len(train_files)}")

# Create labels from filenames (1 = dog, 0 = cat)
train_labels = []
train_paths = []

for file in train_files:
    if file.startswith('dog'):
        train_labels.append(1)  # dog
    elif file.startswith('cat'):
        train_labels.append(0)  # cat
    else:
        continue
    train_paths.append(os.path.join(TRAIN_PATH, file))

train_labels = np.array(train_labels)
print(f"Dog images: {sum(train_labels)}")
print(f"Cat images: {len(train_labels) - sum(train_labels)}")
print(f"Class balance: {sum(train_labels) / len(train_labels):.3f}")

# Create stratified split
X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    train_paths, train_labels, 
    test_size=0.2, 
    stratify=train_labels,
    random_state=42
)

print(f"Training samples: {len(X_train_paths)}")
print(f"Validation samples: {len(X_val_paths)}")
print(f"Training class balance: {y_train.mean():.3f}")
print(f"Validation class balance: {y_val.mean():.3f}")

# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# No augmentation for validation
val_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
def create_generator(file_paths, labels, datagen, batch_size=32, shuffle=True):
    while True:
        indices = np.arange(len(file_paths))
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start+batch_size]
            batch_paths = [file_paths[i] for i in batch_indices]
            batch_labels = labels[batch_indices]
            
            batch_images = []
            for path in batch_paths:
                img = tf.keras.preprocessing.image.load_img(
                    path, target_size=(IMG_HEIGHT, IMG_WIDTH)
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = datagen.random_transform(img_array)
                batch_images.append(img_array)
            
            yield np.array(batch_images), batch_labels

train_generator = create_generator(X_train_paths, y_train, train_datagen, BATCH_SIZE, shuffle=True)
val_generator = create_generator(X_val_paths, y_val, val_datagen, BATCH_SIZE, shuffle=False)

# Load VGG16 without top layers
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Freeze base layers
base_model.trainable = False

# Build custom head
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()

# Calculate steps per epoch
train_steps = len(X_train_paths) // BATCH_SIZE
val_steps = len(X_val_paths) // BATCH_SIZE

print(f"Training steps per epoch: {train_steps}")
print(f"Validation steps per epoch: {val_steps}")

# Callbacks
checkpoint_cb = ModelCheckpoint(
    '/home/code/models/vgg16_baseline.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

early_stop_cb = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Train the model
print("Training model...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=20,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
    verbose=1
)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss
ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Model Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Accuracy
ax2.plot(history.history['accuracy'], label='Training Accuracy')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax2.set_title('Model Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig('/home/code/experiments/vgg16_training_history.png')
plt.show()

# Calculate validation log loss
print("\nCalculating validation log loss...")
val_predictions = []
val_true = []

for i in range(val_steps):
    batch_images, batch_labels = next(val_generator)
    preds = model.predict(batch_images, verbose=0)
    val_predictions.extend(preds.flatten())
    val_true.extend(batch_labels)

val_log_loss = log_loss(val_true, val_predictions)
print(f"Validation Log Loss: {val_log_loss:.6f}")

# Also calculate accuracy
val_predictions_binary = [1 if p > 0.5 else 0 for p in val_predictions]
accuracy = np.mean([p == t for p, t in zip(val_predictions_binary, val_true)])
print(f"Validation Accuracy: {accuracy:.4f}")

# Load test images
test_files = sorted([f for f in os.listdir(TEST_PATH) if f.endswith('.jpg')])
print(f"Total test images: {len(test_files)}")

# Create test generator (no augmentation, only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

def test_generator():
    for file in test_files:
        path = os.path.join(TEST_PATH, file)
        img = tf.keras.preprocessing.image.load_img(
            path, target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = test_datagen.standardize(img_array)
        yield np.expand_dims(img_array, axis=0)

# Predict on test set
print("Generating test predictions...")
test_predictions = []

for i, test_img in enumerate(test_generator()):
    if i % 500 == 0:
        print(f"Processed {i}/{len(test_files)} images")
    pred = model.predict(test_img, verbose=0)
    test_predictions.append(pred[0][0])

print(f"Generated {len(test_predictions)} predictions")
print(f"Prediction range: {min(test_predictions):.4f} to {max(test_predictions):.4f}")

# Clip predictions to avoid log(0) errors
test_predictions = np.clip(test_predictions, 1e-7, 1-1e-7)
print(f"After clipping - Min: {min(test_predictions):.6f}, Max: {max(test_predictions):.6f}")

# Create submission file
submission = pd.DataFrame({
    'id': range(1, len(test_predictions) + 1),
    'label': test_predictions
})

# Save submission
submission_path = '/home/submission/submission_001_vgg16_baseline.csv'
os.makedirs('/home/submission', exist_ok=True)
submission.to_csv(submission_path, index=False)

print(f"Submission saved to: {submission_path}")
print(f"\nSubmission head:")
print(submission.head())
print(f"\nSubmission tail:")
print(submission.tail())

# Verify format matches sample
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
print(f"\nSample submission format:")
print(sample_submission.head())
print(f"\nOur submission matches format: {list(submission.columns) == list(sample_submission.columns)}")

# Save experiment results
results = {
    'val_log_loss': val_log_loss,
    'val_accuracy': accuracy,
    'epochs_trained': len(history.history['loss']),
    'final_train_loss': history.history['loss'][-1],
    'final_val_loss': history.history['val_loss'][-1]
}

import json
with open('/home/code/experiments/001_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nExperiment results saved to: /home/code/experiments/001_results.json")