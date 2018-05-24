# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

def get_quantile_based_boundaries(feature_values, num_buckets):
    """ Bucketized (Binned) Features """
    boundaries = np.arange(1.0, num_buckets) / num_buckets
    quantiles = feature_values.quantile(boundaries)
    
    return [quantiles[q] for q in quantiles.keys()]

# Bucketized (Binned) Features
def construct_feature_columns_2(input_features):
    """ Construct the TensorFlow Feature Columns.

    @Returns:        A set of feature columns

    """

    # First, convert the raw input to a numeric column.
    households = tf.feature_column.numeric_column("households")
    longitude = tf.feature_column.numeric_column("longitude")
    latitude = tf.feature_column.numeric_column("latitude")
    housing_median_age = tf.feature_column.numeric_column("housing_median_age")
    median_income = tf.feature_column.numeric_column("median_income")
    rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")
    
    # Divide households into 7 buckets.
    bucketized_households = tf.feature_column.bucketized_column(
        source_column=households, 
        boundaries=get_quantile_based_boundaries(
            input_features["households"], 7)
    )
    
    # Divide longitude into 10 buckets.
    bucketized_longitude = tf.feature_column.bucketized_column(
        source_column=longitude, 
        boundaries=get_quantile_based_boundaries(
            input_features["longitude"], 10)
    )

    bucketized_latitude = tf.feature_column.bucketized_column(
        source_column=latitude,
        boundaries=get_quantile_based_boundaries(
            input_features["latitude"], 10)
    )
    
    bucketized_housing_median_age = tf.feature_column.bucketized_column(
        source_column=housing_median_age, 
        boundaries=get_quantile_based_boundaries(
            input_features["housing_median_age"], 7)
    )

    bucketized_median_income = tf.feature_column.bucketized_column(
        source_column=median_income, 
        boundaries=get_quantile_based_boundaries(
            input_features["median_income"], 7)
    )

    bucketized_rooms_per_person = tf.feature_column.bucketized_column(
        source_column=rooms_per_person, 
        boundaries=get_quantile_based_boundaries(
            input_features["rooms_per_person"], 7)
    )

    feature_columns = set([
        bucketized_longitude,
        bucketized_latitude,
        bucketized_housing_median_age,
        bucketized_households,
        bucketized_median_income,
        bucketized_rooms_per_person])
  
    return feature_columns

# Feature Crosses
def construct_feature_columns_3(input_features):
    """ Construct the TensorFlow Feature Columns.

    @Returns:        A set of feature columns

    """

    # First, convert the raw input to a numeric column.
    households = tf.feature_column.numeric_column("households")
    longitude = tf.feature_column.numeric_column("longitude")
    latitude = tf.feature_column.numeric_column("latitude")
    housing_median_age = tf.feature_column.numeric_column("housing_median_age")
    median_income = tf.feature_column.numeric_column("median_income")
    rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")

    # Divide households into 7 buckets.
    bucketized_households = tf.feature_column.bucketized_column(
        source_column=households, 
        boundaries=get_quantile_based_boundaries(
            input_features["households"], 7)
    )
    
    # Divide longitude into 10 buckets.
    bucketized_longitude = tf.feature_column.bucketized_column(
        source_column=longitude, 
        boundaries=get_quantile_based_boundaries(
            input_features["longitude"], 10)
    )

    bucketized_latitude = tf.feature_column.bucketized_column(
        source_column=latitude,
        boundaries=get_quantile_based_boundaries(
            input_features["latitude"], 10)
    )
    
    bucketized_housing_median_age = tf.feature_column.bucketized_column(
        source_column=housing_median_age, 
        boundaries=get_quantile_based_boundaries(
            input_features["housing_median_age"], 7)
    )

    bucketized_median_income = tf.feature_column.bucketized_column(
        source_column=median_income, 
        boundaries=get_quantile_based_boundaries(
            input_features["median_income"], 7)
    )

    bucketized_rooms_per_person = tf.feature_column.bucketized_column(
        source_column=rooms_per_person, 
        boundaries=get_quantile_based_boundaries(
            input_features["rooms_per_person"], 7)
    )

    # Make a feature column for the long_x_lat feature cross
    long_x_lat = tf.feature_column.crossed_column(
        keys=set([bucketized_longitude, bucketized_latitude]), 
        hash_bucket_size=1000 # The number of categories
    )
    
    feature_columns = set([
        bucketized_longitude,
        bucketized_latitude,
        bucketized_housing_median_age,
        bucketized_households,
        bucketized_median_income,
        bucketized_rooms_per_person,
        long_x_lat])
  
    return feature_columns

if __name__ == '__main__':
    california_housing_df = pd.read_csv("../data/california_housing_train.csv", sep=",")
    # numpy.random.permutation(length)用來產生一個隨機序列
    california_housing_df = california_housing_df.reindex(
        np.random.permutation(california_housing_df.index)
    )

    # Choose the first 12000 (out of 17000) examples for training.
    training_examples = preprocess_features(california_housing_df.head(12000))
    training_targets = preprocess_targets(california_housing_df.head(12000))

    # Choose the last 5000 (out of 17000) examples for validation.
    validation_examples = preprocess_features(california_housing_df.tail(5000))
    validation_targets = preprocess_targets(california_housing_df.tail(5000))

    '''
    # Double-check that we've done the right thing.
    print("Training examples summary:")
    display.display(training_examples.describe())
    print("Validation examples summary:")
    display.display(validation_examples.describe())

    print("Training targets summary:")
    display.display(training_targets.describe())
    print("Validation targets summary:")
    display.display(validation_targets.describe())
    '''

    """ FTRL Optimization Algorithm

    High dimensional linear models benefit from using 
    a variant of gradient-based optimization called FTRL.
    This algorithm has the benefit of scaling the learning rate 
    differently for different coefficients, 
    which can be useful if some features 
    rarely take non-zero values 
    (it also is well suited to support L1 regularization). 
    We can apply FTRL using the FtrlOptimizer.

    """

    '''
    # Bucketized (Binned) Features
    _ = train_model(
        learning_rate=1.0,
        steps=500,
        batch_size=100,
        feature_columns=construct_feature_columns_2(training_examples),
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets
    )
    '''

    # Train the Model Using Feature Crosse
    _ = train_model(
        learning_rate=1.0,
        steps=500,
        batch_size=100,
        feature_columns=construct_feature_columns_3(training_examples),
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets
    )
