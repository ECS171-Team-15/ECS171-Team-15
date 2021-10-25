import time
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def create_model(learning_rate, hidden_nodes, input_dim):
    # Set up layers for the model
    model = Sequential()

    # Input layer
    model.add(Dense(hidden_nodes[0], input_dim=input_dim, activation='sigmoid'))
    
    # Hidden layers
    for i in range(len(hidden_nodes)):
        if i == 0:
            # Already specified first hidden layer's # of nodes
            continue
        model.add(Dense(hidden_nodes[i], activation='relu'))
        
    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    opt = SGD(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt , metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # Save the start time to calculate the total running time
    start_time = time.time()

    # load data and separate them into class and features
    # TODO: create module of file paths
    df = pd.read_csv("../processed_data/full_data.csv")
    feature_data = df.drop("class", axis=1)
    class_data = df.iloc[:, df.shape[1]-1]

    # create classifier with our model
    new_model = KerasClassifier(build_fn=create_model, verbose=0)

    # define the grid search parameters
    hidden_nodes = [(1000, 50, 5), (1500, 100, 10), (2000, 200, 20, 5)]
    learning_rate = [0.1, 0.3, 0.5]
    epochs = [10, 50, 100]
    input_dim = [feature_data.shape[1]]
    
    param_grid = dict(input_dim=input_dim, hidden_nodes=hidden_nodes, learning_rate=learning_rate, epochs=epochs)
    grid = GridSearchCV(estimator=new_model, param_grid=param_grid, n_jobs=-1, cv=3)
    
    # Train model
    grid_result = grid.fit(np.asarray(feature_data).astype('float64'), np.asarray(class_data).astype('float64'))

    # summarize results
    print("MAX ACCURACY: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))

    print("TRACEBACK:\n")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("Accuracy: %f (STD: %f) with: %r" % (round(mean, 2), round(stdev, 4), param))

    # Print total running time of this program
    print(f"\nIt took {time.time() - start_time} seconds to run this program\n")
