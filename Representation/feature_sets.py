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

def select_and_transform_features(source_df, LATITUDE_RANGES):
    selected_examples = pd.DataFrame()
    selected_examples["median_income"] = source_df["median_income"]

    for r in LATITUDE_RANGES:
        selected_examples["latitude_{}_to_{}".format(r)] = source_df["latitude"].apply(
            lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0
        )

    return selected_examples

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

    # Develop a Good Feature Set: 
    """ Correlation matrix

    shows pairwise correlations both for each feature
    
    1. feature compared to the target -> (feature, target)
    2. feature compared to other feature -> (feature, other feature)

    Here, correlation is defined as the ```Pearson correlation coefficient```. 
    You don't have to understand the mathematical details for this exercise.
    
    Correlation values have the following meanings:
    * -1.0: perfect negative correlation
    * 0.0:  no correlation
    * 1.0:  perfect positive correlation

    Ideally, we'd like to have features that are strongly correlated with the target.
    We'd also like to have features that aren't so strongly correlated with each other, 
    so that they add independent information.

    """
    correlation_df = training_examples.copy()
    correlation_df["target"] = training_targets["median_house_value"]

    display.display(correlation_df.corr())

    ## Add features of choice as a list of quoted strings
    minimal_features = [
        "median_income",
        "latitude"
    ]

    assert minimal_features, "You must select at least one feature!"

    
    minimal_training_examples = training_examples[minimal_features]
    minimal_validation_examples = validation_examples[minimal_features]

    '''
    train_model(
        learning_rate=0.01,
        steps=500,
        batch_size=5,
        training_examples=minimal_training_examples,
        training_targets=training_targets,
        validation_examples=minimal_validation_examples,
        validation_targets=validation_targets
    )
    '''

    # Make Better Use of Latitude
    #plt.scatter(training_examples["latitude"], training_targets["median_house_value"])

    LATITUDE_RANGES = zip(range(32, 44), range(33, 45))

    selected_training_examples = select_and_transform_features(training_examples, LATITUDE_RANGES)
    selected_validation_examples = select_and_transform_features(validation_examples, LATITUDE_RANGES) 

    train_model(
        learning_rate=0.01,
        steps=500,
        batch_size=5,
        training_examples=selected_training_examples,
        training_targets=training_targets,
        validation_examples=selected_validation_examples,
        validation_targets=validation_targets
    )

    plt.show()