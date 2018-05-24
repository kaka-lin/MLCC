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


def get_quantile_based_buckets(feature_values, num_buckets):
    quantiles = feature_values.quantile(
        # 分位數: df['a'].quantile(0,1) -> 10th percentile
        # 將data切成X個buckets
        [(i+1.) / (num_buckets + 1.) for i in range(num_buckets)]
    )
    
    return [quantiles[q] for q in quantiles.keys()]


def construct_feature_columns(training_examples):
    """ Construct the TensorFlow Feature Columns.

    @Returns:   A set of feature columns

    """ 

    bucketized_households = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column("households"),
        boundaries=get_quantile_based_buckets(training_examples["households"], 10))

    bucketized_longitude = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column("longitude"),
        boundaries=get_quantile_based_buckets(training_examples["longitude"], 50))
    
    bucketized_latitude = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column("latitude"),
        boundaries=get_quantile_based_buckets(training_examples["latitude"], 50))
    
    bucketized_housing_median_age = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column("housing_median_age"),
        boundaries=get_quantile_based_buckets(training_examples["housing_median_age"], 10))
    
    bucketized_total_rooms = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column("total_rooms"),
        boundaries=get_quantile_based_buckets(training_examples["total_rooms"], 10))
    
    bucketized_total_bedrooms = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column("total_bedrooms"),
        boundaries=get_quantile_based_buckets(training_examples["total_bedrooms"], 10))
    
    bucketized_population = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column("population"),
        boundaries=get_quantile_based_buckets(training_examples["population"], 10))
    
    bucketized_median_income = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column("median_income"),
        boundaries=get_quantile_based_buckets(training_examples["median_income"], 10))
    
    bucketized_rooms_per_person = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column("rooms_per_person"),
        boundaries=get_quantile_based_buckets(
        training_examples["rooms_per_person"], 10))

    # Make a feature column for the long_x_lat feature cross
    long_x_lat = tf.feature_column.crossed_column(
        keys=set([bucketized_longitude, bucketized_latitude]), 
        hash_bucket_size=1000
    )

    feature_columns = set([
        long_x_lat,
        bucketized_longitude,
        bucketized_latitude,
        bucketized_housing_median_age,
        bucketized_total_rooms,
        bucketized_total_bedrooms,
        bucketized_population,
        bucketized_households,
        bucketized_median_income,
        bucketized_rooms_per_person])
  
    return feature_columns

################################################################################################################################################
# Logistic Regression
def train_linear_classifier_model(
    learning_rate, 
    regularization_strength,
    steps, 
    batch_size, 
    training_examples, 
    training_targets, 
    validation_examples, 
    validation_targets):
    """ Trains a linear regression model
  
    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.
  
    @learning_rate:            A `float`, the learning rate.
    @regularization_strength:  A `float` that indicates the strength of the L1
                               regularization. A value of `0.0` means no regularization.
    @steps:                    A non-zero `int`, the total number of training steps. A training step
                               consists of a forward and backward pass using a single batch.
    @batch_size:               A non-zero `int`, the batch size.
    @training_examples:        A `DataFrame` containing one or more columns from
                               `california_housing_df` to use as input features for training.
    @training_targets:         A `DataFrame` containing exactly one column from
                               `california_housing_df` to use as target for training.
    @validation_examples:      A `DataFrame` containing one or more columns from
                               `california_housing_df` to use as input features for validation.
    @validation_targets:       A `DataFrame` containing exactly one column from
                               `california_housing_df` to use as target for validation.
      
    @returns:                  A `LinearClassifier` object trained on the training data.

    """

    periods = 7
    steps_per_period = steps / periods

    # Create a linear classifier object.
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=regularization_strength)
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
    print("LogLoss (on validation data):")
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
