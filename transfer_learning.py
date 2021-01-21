import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
import keras

import tensorflow as tf
import matplotlib.pyplot as plt

DATASET_DIR = "monkey_dataset/"
TRAIN_DIR = DATASET_DIR + "training/training/"
VALIDATION_DIR = DATASET_DIR + "validation/validation/"

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BATCH_SIZE = 16

if __name__ == "__main__":
    # Overwrite insufficient GPU memory criteria
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    # Get the labels
    label_txt = pd.read_csv(DATASET_DIR + "monkey_labels.txt", skipinitialspace=True)
    label_txt = label_txt.rename(columns=lambda x: x.strip())
    labels = pd.DataFrame()
    labels["id"] = label_txt["Label"].str.strip()
    labels["name"] = label_txt["Common Name"].str.strip()

    # Train data generator
    train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest'
                                        )
    # Test data generator
    validation_data_gen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_data_gen.flow_from_directory(TRAIN_DIR,
                                                         target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         class_mode="categorical")

    validation_generator = validation_data_gen.flow_from_directory(VALIDATION_DIR,
                                                                   target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                                   batch_size=BATCH_SIZE,
                                                                   shuffle=True,
                                                                   class_mode="categorical")

    training_samples = train_generator.samples
    validation_samples = validation_generator.samples
    total_steps = training_samples // BATCH_SIZE

    # Get the pre-trained VGG-16 model
    model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
                        pooling="max")
    # Make last 5 layers of the VGG model trainable
    for layer in model.layers[:-5]:
        layer.trainable = False

    transfer_model = keras.models.Sequential()
    for layer in model.layers:
        transfer_model.add(layer)
    # Add 3 layers to the original VGG-16 network for transfer learning
    transfer_model.add(keras.layers.Dense(512, activation="relu"))
    transfer_model.add(keras.layers.Dropout(0.5))
    transfer_model.add(keras.layers.Dense(10, activation="softmax"))

    # Define metrics and optimizers for the model
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
    transfer_model.compile(loss="categorical_crossentropy",
                           optimizer=adam,
                           metrics=["acc"])

    # Train the model
    model_history = transfer_model.fit_generator(train_generator, steps_per_epoch=training_samples // BATCH_SIZE,
                                                 epochs=25,
                                                 validation_data=validation_generator,
                                                 validation_steps=validation_samples // BATCH_SIZE)

    # Plot loss and accuracy of the model
    acc = model_history.history['acc']
    val_acc = model_history.history['val_acc']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, val_acc, 'blue', label='Validation acc')
    plt.legend()

    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, 'red', label='Training loss')
    plt.plot(epochs, val_loss, 'blue', label='Validation loss')

    plt.legend()
    plt.show()
