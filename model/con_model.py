import tensorflow
from tensorflow.keras import layers, models
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

import common

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
    train_x, test_x, train_y, test_y = common.load_and_split_data()

    # Scale data to [0, 1] range
    train_x, test_x = common.rescale_data(train_x, test_x)

    # Reshape data to 2D
    train_x = tensorflow.reshape(train_x, (-1, 302, 425, 1))
    test_x = tensorflow.reshape(test_x, (-1, 302, 425, 1))

    # Model hyperparameters
    param_grid = {
        'dropout': [
                    False,
                    True
                   ],
        'kernel_regularizer': [
                               None,
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
            model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100)
            results = model.evaluate(test_x, test_y)
            print("Test loss and accuracy", results)

