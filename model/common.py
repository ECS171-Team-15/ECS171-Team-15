import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_split_data(tSiz, ranSt):
    # Load data
    df = pd.read_csv('../processed_data/original.csv')
    feature_data = df.drop(columns='class')
    label_data = df.iloc[:, df.shape[1]-1]
    # Split data
    #ZZ train_x, test_x, train_y, test_y = train_test_split(feature_data, label_data, test_size=0.2, random_state=6)
    train_x, test_x, train_y, test_y = train_test_split(feature_data, label_data, test_size=tSiz, random_state=ranSt)
    return (train_x, test_x, train_y, test_y)

def rescale_data(train_x, test_x):
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)
    return train_x, test_x

def print_grid_results(grid_results):
    print("Results:")
    print("Max Accuracy: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))

    # Print the results of each grid item
    print("Traceback:\n")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("Accuracy: %f (STD: %f) with: %r" % (round(mean, 2), round(stdev, 4), param))
