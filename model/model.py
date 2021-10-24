import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


def create_model(learning_rate, num_layers, num_nodes, input_dim):
    # Set up layers for the model
    model = Sequential()

    # Input layer
    model.add(Dense(16, input_dim=input_dim, activation='sigmoid'))
    # Hidden layer
    for _ in range(num_layers):
        model.add(Dense(num_nodes, activation='relu'))
        model.add(Dense(num_nodes, activation='relu'))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    opt = SGD(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt , metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # load data and separate them into class and features
    df = pd.read_csv("../data/full_data.csv")
    feature_data = df.drop("class", axis=1)
    class_data = df.iloc[:, df.shape[1]-1]

    # create classifier with our model
    new_model = KerasClassifier(build_fn=create_model, verbose=0)

    # define the grid search parameters
    num_layers = [3, 6, 12]
    num_nodes = [3, 6, 12]
    learning_rate = [0.1, 0.3, 0.5]
    epochs = [1, 5, 10]
    input_dim = [feature_data.shape[1]]
    param_grid = dict(input_dim=input_dim, num_layers=num_layers, num_nodes=num_nodes, learning_rate=learning_rate, epochs=epochs)
    grid = GridSearchCV(estimator=new_model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(feature_data, class_data)

    # summarize results
    print("MAX ACCURACY: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))

    print("TRACEBACK:\n")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("Accuracy: %f (STD: %f) with: %r" % (round(mean, 2), round(stdev, 4), param))