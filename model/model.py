import time
import numpy as np
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import MinMaxScaler

# import tensorflow.config

def build_dnn_model(input_dim, learning_rate, hidden_nodes):
    model = Sequential()

    # Hidden layers
    for i in range(0, len(hidden_nodes)):
        if i == 0:
            model.add(Dense(hidden_nodes[i], activation='relu', input_dim=input_dim))
            continue
        model.add(Dense(hidden_nodes[i], activation='relu'))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    opt = SGD(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # Limit memory usage of GPU
    '''
    gpus = tensorflow.config.list_physical_devices('GPU')
    if gpus:
        try:
            tensorflow.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    '''

    # Load data and separate class and features
    train_x, test_x, train_y, test_y = common.load_and_split_data()

    # Rescale data
    train_x, test_x = common.rescale_data(train_x, test_x)

    num_pixels_img = train_x.shape[1]

    # Models to train and their parameter grids
    param_grid = {
        'model': KerasClassifier(build_dnn_model),
        'hp': {
            # Tuple with one dimension size
            'input_dim': (num_pixels_img,),
            'learning_rate': [0.1, 0.3, 0.5],
            # Ordered hidden node combinations
            'hidden_nodes': [
                [2000, 300, 40, 10],
                [2500, 400, 50, 13]
            ],
        },
        'fit_params': {
            'epochs': 500
        }
    }

    # Train models
    for model_name, model_properties in models.items():
        grid = GridSearchCV(estimator=model_properties['model'], param_grid=model_properties['hp'], n_jobs=1, cv=3)

        # feature_data = np.asarray(feature_data).astype('float64')
        # class_data = np.asarray(class_data).astype('float64')

        if model_name == 'cnn':
            # Resize training image data into original 2D shape
        else:
            # Data can be 1D for DNNs
            train_x = tensorflow.reshape(train_x, (-1,))
            test_x = tensorflow.reshape(test_x, (-1,))

        grid_result = grid.fit(training_x, training_y, fit_params=model_properties['fit_params'])
