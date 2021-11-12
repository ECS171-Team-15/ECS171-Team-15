import tensorflow
from tensorflow.keras import layers, models
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from . import common

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
        # Manually set the mean dimensions of the original dataset
        # 3rd item in tuple is the number of channels
        # Only 1 channel for grayscaleKerasClassifier
        'input_dim': [(302, 425, 1)],
        'dropout': [False, True],
        'kernel_regularizer': [None, 'l1_l2'],
    },

    fit_params = {
        'epochs': [100]
    }

    # Train model
    model = KerasClassifier(build_fn=build_conv_model)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
    grid_results = grid.fit(train_x, train_y, fit_params)

    common.print_grid_results(grid_results)
