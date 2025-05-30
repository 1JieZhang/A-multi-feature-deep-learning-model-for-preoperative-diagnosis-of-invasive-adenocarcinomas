from keras.models import Model
from keras.layers import Input, Conv2D, Dense, BatchNormalization, Flatten, \
     Activation, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import *
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
import random
from data_acq import X, text
from scipy import ndimage
import numpy as np

# GPU configuration
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

def get_model(width=32, height=32, depth=3):
    """Build 2D CNN model with proper channel configuration"""
    inputs = keras.Input((width, height, depth))  # Changed depth to 3 to match input

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs, x, name="2d_vgg_cnn")
    return model

# Verify input shape
print("Input shape:", X.shape)  # Should be (num_samples, 32, 32, 3)

# Create model with correct input channels (3)
base_model = get_model(32, 32, 3)  # Changed depth to 3 to match input
base_model.summary()

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, text, test_size=0.2)

# Data augmentation
@tf.function
def rotate(volume):
    """Rotate volume by random angle"""
    def scipy_rotate(volume):
        angles = [-20, -10, -5, 5, 10, 20]
        angle = random.choice(angles)
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume = np.clip(volume, 0, 1)  # Replace manual clipping
        return volume
    
    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    augmented_volume.set_shape(volume.shape)  # Maintain shape
    return augmented_volume

def train_preprocessing(volume, label):
    volume = rotate(volume)
    return volume, label

def validation_preprocessing(volume, label):
    return volume, label

# Create datasets
train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(len(x_train))
                .map(train_preprocessing)
                .batch(2)  # Small batch size for small dataset
                .prefetch(tf.data.AUTOTUNE))

test_dataset = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
               .batch(2)
               .prefetch(tf.data.AUTOTUNE))

# Learning rate schedule
initial_learning_rate = 1e-4  # Increased from 1e-6
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=30, decay_rate=0.96, staircase=True
)

# Compile model
base_model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "2d_image.h5", save_best_only=True
)
early_stopping = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)

# Training
print(f"TensorFlow version: {tf.__version__}")
print(f'GPU Available: {"Yes" if tf.config.list_physical_devices("GPU") else "No"}')

history = base_model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=200,
    verbose=1,  
    callbacks=[checkpoint_cb, early_stopping],
)

# Plot training history
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()