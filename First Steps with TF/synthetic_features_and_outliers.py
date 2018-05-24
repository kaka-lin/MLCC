import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

from utils import *

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

def train_model(data, learning_rate, steps, batch_size, features="total_rooms", targets="median_house_value"):
    periods = 10
    steps_per_period = steps / periods
   
    # Set up to plot the state of our model's line each period.
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(targets)
    plt.xlabel(features)
    sample = data.sample(n=300)
    plt.scatter(sample[features], sample[targets])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]
    
    ###############################################################################
    # retrieve data
    X, Y = load_data(data, features=features, targets=targets)

    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column(key=features)]
    
    # Create input functions.
    training_input_fn = lambda: my_input_fn(X, Y, batch_size=batch_size)
    prediction_input_fn = lambda: my_input_fn(X, Y, num_epochs=1, shuffle=False)
    
    ## Build model
    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    ###############################################################################
    ## Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute predictions.
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])
    
        # Compute loss.
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, Y))
        
        # Occasionally print the current loss.
        print("period {:2d} : {:.2f}".format(period, root_mean_squared_error))
        
        
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)

        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.
        y_extents = np.array([0, sample[targets].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/{}/weights'.format(features))[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        # np.minimum(X, Y), np.maxmun(X, Y): 會逐位比較(broadcasting)
        x_extents = np.maximum(np.minimum(x_extents, sample[features].max()), sample[features].min())
    
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period]) 
        
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    # Output a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(Y)
    #display.display(calibration_data.describe())
    print(calibration_data.describe())

    print("Final RMSE (on training data): {:.2f}".format(root_mean_squared_error))

    plt.figure()
    plt.title("Calibration data")
    _ = plt.scatter(calibration_data["predictions"], calibration_data["targets"])
  

if __name__ == '__main__':
    # Read CSV file
    california_housing_df = pd.read_csv("../data/california_housing_train.csv", sep=",")
    
    # randomize the data and scale median_house_value to be in units of thousands
    california_housing_df = df_processing(california_housing_df)

    # Synthetic Feature
    california_housing_df["rooms_per_person"] = (
        california_housing_df["total_rooms"] / california_housing_df["population"])

    plt.figure()
    plt.title("Before clip outliers")
    _ = california_housing_df["rooms_per_person"].hist()

    # Clip Outliers
    california_housing_df["rooms_per_person"] = (
    california_housing_df["rooms_per_person"]).apply(lambda x: min(x, 5))

    plt.figure()
    plt.title("After clip outliers")
    _ = california_housing_df["rooms_per_person"].hist()

    features="rooms_per_person"
    targets="median_house_value"
    
    train_model(
        data=california_housing_df,
        learning_rate=0.05,
        steps=500,
        batch_size=5,
        features=features,
        targets=targets
    )

    plt.show()

    