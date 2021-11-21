import tensorflow
from tensorflow.keras import layers, models
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import numpy as np
import common

CSV_PATH = "../processed_data/original.csv"
TEST_DATA_SIZE = 0.2
RANDOM_STATE = 77

def build_conv_model(input_dim, kernel_regularizer, dropout):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_dim))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=kernel_regularizer))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=kernel_regularizer))
    model.add(layers.Flatten())

    if dropout:
        model.add(layers.Dropout(0.5))

    # Classifier layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Default learning rate=0.001
    # Source: https://keras.io/api/optimizers/adam/
    model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load data and separate class and features
    train_x, test_x, train_y, test_y = common.load_and_split_data(TEST_DATA_SIZE, RANDOM_STATE, CSV_PATH)

    # Scale data to [0, 1] range
    train_x, test_x = common.rescale_data(train_x, test_x)

    # Reshape data to 2D
    train_x = tensorflow.reshape(train_x, (-1, 302, 425, 1))
    test_x = tensorflow.reshape(test_x, (-1, 302, 425, 1))

    # Model hyperparameters
    param_grid = {
        'dropout': [
                    True
                   ],
        'kernel_regularizer': [
                               'l1_l2'
                              ],
    }

    for dropout in param_grid['dropout']:
        for kernel_regularizer in param_grid['kernel_regularizer']:
            print("Running model with dropout:", dropout, "and kernel_regularizer:", kernel_regularizer)
            # Manually set the mean dimensions of the original dataset
            # 3rd item in tuple is the number of channels
            # Only 1 channel for grayscaleKerasClassifier
            model = build_conv_model((302, 425, 1), kernel_regularizer, dropout)
            history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100)
            model.save(f'{dropout}{kernel_regularizer}.h5')
            results = model.evaluate(test_x, test_y)
            print("Test loss and accuracy", results)

            #-------PLOT THE ACCURACY AND LOSS FOR EACH OF THE VAL AND TRAINING DATA----------#
            epochs = range(1, 101)
            title_kr_value = kernel_regularizer
            if(kernel_regularizer == None):
                title_kr_value = 'no'
            if(dropout):
                title_acc = f'Model performance (accuracy) with dropout and {title_kr_value} regularization'
                title_loss = f'Model performance (loss) with dropout and {title_kr_value} regularization'

            else:
                title_acc = f'Model performance (accuracy) with no dropout and {title_kr_value} regularization'
                title_loss = f'Model performance (loss) with no dropout and {title_kr_value} regularization'
            img_name_loss = f'{dropout}{kernel_regularizer}_loss.png'
            img_name_acc = f'{dropout}{kernel_regularizer}_acc.png'

            #get the callbacks of the loss and accuracy metrics for each of the data set
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            train_acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']

            #1) plot the loss for the given hp -> 'green' represents training and 'blue' represents validation
            plt.plot(epochs, train_loss, 'g', label = 'Training Loss')
            plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(title_loss)
            plt.legend()
            plt.savefig(img_name_loss)
            plt.show(block=False)

            #2) Plot the accuracy for the given HP -> 'green' represents training and 'blue' represents validation
            plt.plot(train_acc, 'g', label = 'Training Accuracy')
            plt.plot(val_acc, 'b', label = 'Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title(title_acc)
            plt.legend()
            plt.savefig(img_name_acc)
            plt.show(block=False)