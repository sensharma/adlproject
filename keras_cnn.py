import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, InputLayer
# from keras import layers
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_models')

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential(
    [
        # InputLayer(input_shape=input_shape),
        # layers.Conv2D(32, kernel_size=(4, 4), activation="relu"),
        Conv2D(32, kernel_size=(4, 4),
               padding='valid',
               activation="relu",
               input_shape=input_shape),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Conv2D(64, kernel_size=(4, 4), activation="relu"),
        # Conv2D(64, kernel_size=(4, 4), activation="relu"),
        Conv2D(64, kernel_size=(4, 4),
               padding='valid',
               activation="relu"),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Flatten(),
        Flatten(),
        # layers.Dropout(0.25),
        Dropout(0.25),
        # layers.Dense(1600, activation="relu"),
        Dense(1600, activation="relu"),
        # layers.Dropout(0.5),
        Dropout(0.5),
        # layers.Dense(num_classes, activation="softmax"),
        Dense(num_classes, activation="softmax"),
    ]
)


batch_size = 128
epochs = 20

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=3,
                                        #  restore_best_weights=True
                                         )

model.fit(x_train, y_train, 
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)

model.summary()

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

filepath = f'{MODEL_PATH}/keras_deeplift_model'
model.save_weights(
        filepath,
        overwrite=True,
        # include_optimizer=True,
        # save_format='h5',
)
