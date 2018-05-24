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

def model_size(estimator):
    """ Calculate the Model Size

    To calculate the model size, 
    we simply count the number of parameters that are non-zero. 
    We provide a helper function below to do that.
    The function uses intimate knowledge of the Estimators API - 
    don't worry about understanding how it works.

    """
    variables = estimator.get_variable_names()
    size = 0
    for variable in variables:
        if not any(x in variable
                   for x in ['global_step',
                             'centered_bias_weight',
                             'bias_weight',
                             'Ftrl']):
            size += np.count_nonzero(estimator.get_variable_value(variable))
    return size


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

    # Double-check that we've done the right thing.
    print("Training examples summary:")
    display.display(training_examples.describe())
    print("Validation examples summary:")
    display.display(validation_examples.describe())

    print("Training targets summary:")
    display.display(training_targets.describe())
    print("Validation targets summary:")
    display.display(validation_targets.describe())

    """ Reduce the Model Size

    Your team needs to build a highly accurate Logistic Regression model on the SmartRing, 
    a ring that is so smart it can sense the demographics of a city block 
    ('median_income', 'avg_rooms', 'households', ..., etc.) 
    and tell you whether the given city block is high cost city block or not.

    Since the SmartRing is small, 
    the engineering team has determined that it can only handle a model 
    that has no more than 600 parameters. 
    On the other hand, the product management team has determined 
    that the model is not launchable unless the LogLoss is less than 0.35 on the holdout test set.

    Can you use your secret weapon—L1 regularization—to tune the model 
    to satisfy both the size and accuracy constraints?

    """

    # Find a good regularization coefficient
    linear_classifier = train_linear_classifier_model(
        learning_rate=0.1,
        # TWEAK THE REGULARIZATION VALUE BELOW
        # Althou solution is 0.1, but test result: 0.1 -> model size ~= 760 (loss: 0.24)
        # test 0.5 -> model size ~= 630 (loss: 0.25)
        # test 0.7 -> model size ~= 590 (loss: 0.25)
        regularization_strength=0.7,
        steps=300,
        batch_size=100,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

    # Calculate the Model Size
    print("Model size:", model_size(linear_classifier))

    plt.show()

    

    