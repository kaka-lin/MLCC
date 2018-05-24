import numpy as np
import pandas as pd
import tensorflow as tf
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

def preprocess_targets(california_housing_df):
    """ Prepares target features (i.e., labels) from California housing data set.

    @california_housing_df:  A Pandas DataFrame expected to contain data
                             from the California housing data set.
    
    @Returns:                A DataFrame that contains the target feature.

    """

    output_targets = pd.DataFrame()

    # Scale the target to be in units of thousands of dollars
    output_targets["median_house_value"] = (
        california_housing_df["median_house_value"] / 1000.0
    )

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


def construct_feature_columns(input_features):
    """ Construct the TensorFlow Feature Columns.

    @input_features: The names of the numerical input features to use.

    @Returns:        A set of feature columns

    """ 
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])