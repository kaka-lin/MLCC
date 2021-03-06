import math

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from matplotlib import pyplot as plt
from tensorflow.python.data import Dataset


def preprocess_features(california_housing_df):
    """ Prepares input features from California housing data set.
      
    @california_housing_df:  A Pandas DataFrame expected to contain data
                             from the California housing data set.
    
    @Returns:                A DataFrame that contains the features to be used for the model, 
                             including synthetic features.

    """

    selected_features = california_housing_df[[
        "latitude",
        "longitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"
    ]]

    processed_features = selected_features.copy()

    # Create a synthetic feature.
    processed_features["rooms_per_person"] = (
        california_housing_df["total_rooms"] /
        california_housing_df["population"]
    )

    return processed_features

def preprocess_targets(california_housing_df, threshold=265000):
    """ Prepares target features (i.e., labels) from California housing data set.

    @california_housing_df:  A Pandas DataFrame expected to contain data
                             from the California housing data set.
    
    @Returns:                A DataFrame that contains the target feature.

    """

    output_targets = pd.DataFrame()

    # Create a boolean categorical feature representing whether the
    # median_house_value is above a set threshold.
    output_targets["median_house_value_is_high"] = (
        california_housing_df["median_house_value"] > threshold
    ).astype(float)

    return output_targets

def construct_feature_columns(input_features):
    """ Construct the TensorFlow Feature Columns.

    @input_features: The names of the numerical input features to use.

    @Returns:        A set of feature columns

    """ 
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """ Trains a linear regression model of multiple features.
  
    @features:      pandas DataFrame of features
    @targets:       pandas DataFrame of targets
    @batch_size:    Size of batches to be passed to the model
    @shuffle:       True or False. Whether to shuffle the data.
    @num_epochs:    Number of epochs for which data should be repeated. None = repeat indefinitely
    
    @Returns:       Tuple

    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_linear_regressor_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
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

# Logistic Regression
def train_linear_classifier_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
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
      
    @returns:               A `LinearClassifier` object trained on the training data.

    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear classifier object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
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
    print("LogLoss (on training data):")
    training_log_losses = []
    validation_log_losses = []

    for period in range(0, periods):
        # Training the model, starting from the prior state
        linear_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period, 
        )

        # Take a break and compute predictions.
        training_probabilities = linear_classifier.predict(predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
        
        validation_probabilities = linear_classifier.predict(predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        # Compute training and validation loss.
        # metrics.log_loss = cross entropy loss
        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)

        # Occasionally print the current loss.
        print("  period {:2d} : {:.2f}".format(period, training_log_loss))
        # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.figure()
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()

    return linear_classifier
