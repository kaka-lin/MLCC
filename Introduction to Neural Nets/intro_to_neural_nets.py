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

    dnn_regressor = train_nn_regression_model(
        learning_rate=0.001,
        steps=2000,
        batch_size=100,
        hidden_units=[10, 10],
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)
    
    # Evaluate on Test Data
    california_housing_test_data = pd.read_csv("../data/california_housing_test.csv", sep=",")

    test_examples = preprocess_features(california_housing_test_data)
    test_targets = preprocess_targets(california_housing_test_data)

    predict_test_input_fn = lambda: my_input_fn(test_examples, test_targets, num_epochs=1, shuffle=False)

    test_pred = dnn_regressor.predict(predict_test_input_fn)
    test_pred = np.array([item['predictions'][0] for item in test_pred])

    test_rmse = math.sqrt(metrics.mean_squared_error(test_pred, test_targets))
    print("Final RMSE (on test data): {:.2f}".format(test_rmse)) 
    # in mac, rmse: 109.46
    # in colab, rmse: 101.13

    