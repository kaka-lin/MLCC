import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset

def df_processing(df, feature="median_house_value"):
    # numpy.random.permutation(length)用來產生一個隨機序列
    df = df.reindex(np.random.permutation(df.index))
    df[feature] /= 1000.0

    return df

def load_data(df, features="total_rooms", targets="median_house_value"):
    # Define the input feature: total_rooms.
    features = features
    X = df[[features]].astype('float32') # DataFrame

    # Define the label.
    label = targets
    Y = df[label].astype('float32')

    return [X, Y]


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                             
    
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels