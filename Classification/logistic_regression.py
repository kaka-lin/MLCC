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

    '''
    # How Would Linear Regression Fare?
    # let us first train a naive model that uses linear regression.
    linear_regressor = train_linear_regressor_model(
        learning_rate=0.000001,
        steps=200,
        batch_size=20,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)
    '''
    
    # Train a Logistic Regression Model 
    # and Calculate LogLoss on Validation Set
    linear_classifier = train_linear_classifier_model(
        learning_rate=0.000003,
        steps=20000,
        batch_size=500,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

    # Create input functions
    training_input_fn = lambda: my_input_fn(training_examples, training_targets, batch_size=20)
    predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets, num_epochs=1, shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets, num_epochs=1, shuffle=False)

    #####################################################################################
    # Calculate Accuracy and plot a ROC Curve for the Validation Set
    evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
    print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
    print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
    #####################################################################################

    #####################################################################################
    # You may use class probabilities, 
    # such as those calculated by LinearClassifier.predict, 
    # and Sklearn's roc_curve to obtain the true positive and false positive rates 
    # needed to plot a ROC curve.
    validation_probabilities = linear_classifier.predict(predict_validation_input_fn)
    validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])
    false_positive_rate, true_positive_rate, thresholds_ = metrics.roc_curve(
        validation_targets, validation_probabilities)

    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, label="our model")
    plt.plot([0, 1], [0, 1], label="random classifier")
    _ = plt.legend(loc=2)
        #####################################################################################

    plt.show()
