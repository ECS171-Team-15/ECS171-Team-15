import time
import numpy as np
import sys

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow.config

def create_model(learning_rate, hidden_nodes, input_dim):
    # Set up layers for the model
    model = Sequential()

    # Input layer
    model.add(InputLayer(input_dim))

    # Hidden layers
    for i in range(0, len(hidden_nodes)):
        model.add(Dense(hidden_nodes[i], activation='relu'))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    opt = SGD(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt , metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # Save the start time to calculate the total running time
    start_time = time.time()

    # Limit memory usage of GPU
    gpus = tensorflow.config.list_physical_devices('GPU')
    if gpus:
        try:
            tensorflow.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    # Input validation
    if len(sys.argv) < 2 and (sys.argv[1] != "original" or sys.argv[1] != "modified"):
        print("Usage: python3 model.py [original|modified] [count1] [count2] ...")
        exit()

    print('Opening dataset...', end='')

    # Load data and separate class and features
    df = pd.read_csv(f'../processed_data/{sys.argv[1]}.csv')
    feature_data = df.drop("class", axis=1)
    class_data = df.iloc[:, df.shape[1]-1]

    print('Done.')
    print('Creating model...', end='')

    new_model = KerasClassifier(build_fn=create_model, verbose=0)

    # Read hidden nodes from program arguments
    hidden_nodes = []
    for i in range(1, len(sys.argv)):
        hidden_nodes.append(sys.argv[i])

	# Grid search parameters
    learning_rate = [0.1, 0.3, 0.5]
    epochs = [10, 100, 500]
    input_dim = [feature_data.shape[1]]
    param_grid = dict(input_dim=input_dim, hidden_nodes=hidden_nodes, learning_rate=learning_rate, epochs=epochs)

    grid = GridSearchCV(estimator=new_model, param_grid=param_grid, n_jobs=1, cv=3)

    # Train model
    feature_data = np.asarray(feature_data).astype('float64')
    class_data = np.asarray(class_data).astype('float64')
    grid_result = grid.fit(feature_data, class_data)

    # Results
    print("MAX ACCURACY: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))
    print("TRACEBACK:\n")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("Accuracy: %f (STD: %f) with: %r" % (round(mean, 2), round(stdev, 4), param))

    # Time elapsed
    print()
    print(f"Program time: {time.time() - start_time} seconds")
