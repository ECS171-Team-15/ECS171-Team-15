import tensorflow
from tensorflow.keras import layers, models
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import numpy as np
import common

CSV_PATH = "../processed_data/original.csv"
TEST_DATA_SIZE = 0.2
RANDOM_STATE = 6
OUTPUT_DIM = 1
MEAN_WIDTH = 425
MEAN_HEIGHT = 302
EPOCHS = 100

def build_conv_model(input_dim, kernel_regularizer, dropout, dropout_rate=0.5):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_dim))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=kernel_regularizer))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=kernel_regularizer))
    model.add(layers.Flatten())

    if dropout:
        model.add(layers.Dropout(dropout_rate))

    # Classifier layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(OUTPUT_DIM, activation='sigmoid'))

    #model.summary()

    # Default learning rate=0.001
    # Source: https://keras.io/api/optimizers/adam/
    model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
    return model

def plot_accuracy_and_loss(dropout, kernel_regularizer):
    epochs = range(1, EPOCHS + 1)
    title_kr_value = kernel_regularizer
    if (kernel_regularizer == None):
        title_kr_value = 'no'
    if (dropout):
        title_acc = f'Model performance (accuracy) with dropout and {title_kr_value} regularization'
        title_loss = f'Model performance (loss) with dropout and {title_kr_value} regularization'

    else:
        title_acc = f'Model performance (accuracy) with no dropout and {title_kr_value} regularization'
        title_loss = f'Model performance (loss) with no dropout and {title_kr_value} regularization'
    img_name_loss = f'{dropout}{kernel_regularizer}_loss.png'
    img_name_acc = f'{dropout}{kernel_regularizer}_acc.png'

    # get the callbacks of the loss and accuracy metrics for each of the data set
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # 1) plot the loss for the given hp -> 'green' represents training and 'blue' represents validation
    plt.plot(epochs, train_loss, 'g', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title_loss)
    plt.legend()
    plt.savefig(img_name_loss)
    plt.show(block=False)

    # 2) Plot the accuracy for the given HP -> 'green' represents training and 'blue' represents validation
    plt.plot(train_acc, 'g', label='Training Accuracy')
    plt.plot(val_acc, 'b', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title_acc)
    plt.legend()
    plt.savefig(img_name_acc)
    plt.show(block=False)

if __name__ == "__main__":
    # Load data and separate class and features
    print("Loading from CSV file: " + CSV_PATH)
    train_x, test_x, train_y, test_y = common.load_and_split_data(TEST_DATA_SIZE, RANDOM_STATE, CSV_PATH)
    print("Done!")
    print("Shape of tensor train_x:", train_x.shape)
    print("Shape of tensor train_y:", train_y.shape)
    print("Shape of tensor test_x:", test_x.shape)
    print("Shape of tensor test_y:", test_y.shape)
    # Scale data to [0, 1] range
    train_x, test_x = common.rescale_data(train_x, test_x)

    # Reshape data to 2D
    train_x = tensorflow.reshape(train_x, (-1, MEAN_HEIGHT, MEAN_WIDTH, OUTPUT_DIM))
    test_x = tensorflow.reshape(test_x, (-1, MEAN_HEIGHT, MEAN_WIDTH, OUTPUT_DIM))
    print("Shape of tensor train_x:", train_x.shape)
    print("Shape of tensor train_y:", train_y.shape)

    # Model hyperparameters
    param_grid = {
        'dropout': [
                    True, False
                   ],
        'kernel_regularizer': [
                               'l1_l2', None
                              ],
    }

    for dropout in param_grid['dropout']:
        for kernel_regularizer in param_grid['kernel_regularizer']:
            print("Running model with dropout:", dropout, "and kernel_regularizer:", kernel_regularizer)
            # Manually set the mean dimensions of the original dataset
            # 3rd item in tuple is the number of channels
            # Only 1 channel for grayscaleKerasClassifier
            model = build_conv_model((MEAN_HEIGHT, MEAN_WIDTH, OUTPUT_DIM), kernel_regularizer, dropout)
            history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=EPOCHS)
            model.save(f'{dropout}{kernel_regularizer}.h5')
            results = model.evaluate(test_x, test_y)
            print("Test loss and accuracy", results)

            # Plot accuracy and loss for each combination of hyperparameters
            plot_accuracy_and_loss(dropout, kernel_regularizer)
            print("Graphs have been plotted.")