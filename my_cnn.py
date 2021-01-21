import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import tensorflow as tf

DATASET_DIR = "monkey_dataset/"
TRAIN_DIR = DATASET_DIR + "training/training/"
VALIDATION_DIR = DATASET_DIR + "validation/validation/"

IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150
CHANNELS = 3
SEED = 1337
BATCH_SIZE = 16
NUM_CLASSES = 10
EPOCHS = 50

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
                                        fill_mode='nearest')

    train_generator = train_data_gen.flow_from_directory(TRAIN_DIR,
                                                         target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                         batch_size=BATCH_SIZE,
                                                         seed=SEED,
                                                         shuffle=True,
                                                         class_mode='categorical')

    # Test data generator
    test_data_gen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_data_gen.flow_from_directory(VALIDATION_DIR,
                                                             target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                             batch_size=BATCH_SIZE,
                                                             seed=SEED,
                                                             shuffle=False,
                                                             class_mode='categorical')

    train_num = train_generator.samples
    validation_num = validation_generator.samples

    # Define the architecture of the network
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    # Define metrics and optimizers for the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    # Train the model
    model_history = model.fit_generator(train_generator,
                                        steps_per_epoch=train_num // BATCH_SIZE,
                                        epochs=EPOCHS,
                                        validation_data=train_generator,
                                        validation_steps=validation_num // BATCH_SIZE)

    # Plot loss and accuracy of the model
    acc = model_history.history['acc']
    val_acc = model_history.history['val_acc']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.title('Training and validation accuracy')
    plt.plot(EPOCHS, acc, 'red', label='Training acc')
    plt.plot(EPOCHS, val_acc, 'blue', label='Validation acc')
    plt.legend()

    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(EPOCHS, loss, 'red', label='Training loss')
    plt.plot(EPOCHS, val_loss, 'blue', label='Validation loss')

    plt.legend()
    plt.show()
