from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

# Initialize CNN
classifier = Sequential([
    Input(shape=(28, 28, 3)),

    # Convolution + Pooling
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Flatten
    Flatten(),

    # Fully connected layers
    Dense(56, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile
classifier.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'shapes/train',
    target_size=(28, 28),
    batch_size=8,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'shapes/test',
    target_size=(28, 28),
    batch_size=8,
    class_mode='categorical'
)

# Callbacks
csv_logger = CSVLogger('log.csv', append=True, separator=';')
early_stopping = EarlyStopping(patience=5)

# Training (UPDATED)
model_info = classifier.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=25,
    validation_data=test_set,
    validation_steps=len(test_set),
    callbacks=[csv_logger, early_stopping]
)

# Save model
classifier.save("drawing_classification.h5")

# Plot history (make sure file exists)
from visulization import plot_model_history
plot_model_history(model_info)