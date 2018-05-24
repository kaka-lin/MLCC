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

def train_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
    """ Trains a linear regression model of multiple features.
  
    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.
  
    @learning_rate:         A `float`, the learning rate.
    @steps:                 A non-zero `int`, the total number of training steps. A training step
                            consists of a forward and backward pass using a single batch.
    @batch_size:            A non-zero `int`, the batch size.
    @training_examples:     A `DataFrame` containing one or more columns from
                            `california_housing_df` to use as input features for training.
    @training_targets:      A `DataFrame` containing exactly one column from
                            `california_housing_df` to use as target for training.
    @validation_examples:   A `DataFrame` containing one or more columns from
                            `california_housing_df` to use as input features for validation.
    @validation_targets:    A `DataFrame` containing exactly one column from
                            `california_housing_df` to use as target for validation.
      
    @returns:               A `LinearRegressor` object trained on the training data.

    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    # Create input functions
    training_input_fn = lambda: my_input_fn(training_examples, training_targets, batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets, num_epochs=1, shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets, num_epochs=1, shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []

    for period in range(0, periods):
        # Training the model, starting from the prior state
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )

        # Take a break and compute predictions.
        #training_predictions = linear_regressor.evaluate(predict_training_input_fn)
        training_predictions = linear_regressor.predict(predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        
        #validation_predictions = linear_regressor.evaluate(predict_validation_input_fn)
        validation_predictions = linear_regressor.predict(predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))

        # Occasionally print the current loss.
        print("  period {:2d} : {:.2f}".format(period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    
    print("Model training finished.")
    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    return linear_regressor

if __name__ == '__main__':
    california_housing_df = pd.read_csv("../data/california_housing_train.csv", sep=",")
    # numpy.random.permutation(length)用來產生一個隨機序列
    california_housing_df = california_housing_df.reindex(np.random.permutation(california_housing_df.index))

    # Get train data and validation data
    training_examples = preprocess_features(california_housing_df.head(12000))
    #print(training_examples.describe())
    training_targets = preprocess_targets(california_housing_df.head(12000))
    #print(training_targets.describe())

    validation_examples = preprocess_features(california_housing_df.tail(5000))
    #print(validation_examples.describe())
    validation_targets = preprocess_targets(california_housing_df.tail(5000))
    #print(validation_targets.describe())

    '''
    # Plot Latitude/Longitude vs. Median House Value
    plt.figure(figsize=(13, 8))

    ax = plt.subplot(1, 2, 1)
    ax.set_title("Validatoin Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(validation_examples["longitude"], 
                validation_examples["latitude"],
                cmap="coolwarm", 
                c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())
    
    ax = plt.subplot(1,2,2)
    ax.set_title("Training Data")

    ax.set_autoscaley_on(False)
    ax.set_ylim([32, 43])
    ax.set_autoscalex_on(False)
    ax.set_xlim([-126, -112])
    plt.scatter(training_examples["longitude"],
                training_examples["latitude"],
                cmap="coolwarm",
                c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
    
    plt.show()
    '''

    # Train and Evaluate a Model
    linear_regressor = train_model(
        # TWEAK THESE VALUES TO SEE HOW MUCH YOU CAN IMPROVE THE RMSE
        learning_rate=0.001,
        steps=10,
        batch_size=1,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets
    )  

    #plt.show()


    # Test data
    california_housing__test_df = pd.read_csv("../data/california_housing_test.csv", sep=",")
    test_examples = preprocess_features(california_housing_df)
    test_targets = preprocess_targets(california_housing_df)
    test_input_fn = lambda: my_input_fn(test_examples, test_targets, num_epochs=1, shuffle=False)

    # Evaluate accuracy.
    test_score = linear_regressor.evaluate(input_fn=test_input_fn)
    print("EVALUATE: Final RMSE (on test data): ", math.sqrt(test_score['loss']))
    
    # Predict Output and calculate rmse
    test_predictions = linear_regressor.predict(input_fn=test_input_fn)
    test_predictions = np.array([item['predictions'][0] for item in test_predictions])

    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(test_predictions, test_targets))
    print("PREDICT: Final RMSE (on test data): ", root_mean_squared_error)
