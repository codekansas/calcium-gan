"""Some utils for loading data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import urllib

import numpy as np
import pandas as pd

import keras
import keras.backend as K

import matplotlib.pyplot as plt

# Defines some constants.
DATA_URL = ('https://s3.amazonaws.com/neuro.datasets/'
            'challenges/spikefinder/spikefinder.train.zip')
DATA_FNAME = 'spikefinder.train'

BASE = os.path.dirname(os.path.realpath(__file__))
DATA_FPATH = os.path.join(BASE, DATA_FNAME)

if not os.path.exists(DATA_FPATH):  # Downloads the data from online.
    zipped_name = DATA_FNAME + '.zip'
    zipped_path = os.path.join(BASE, zipped_name)
    urllib.urlretrieve(DATA_URL, zipped_name)
    assert os.path.exists(zipped_path), 'Downloading "%s" failed.' % DATA_URL

    import zipfile
    with zipfile.ZipFile(zipped_path, 'r') as z:
        z.extractall(BASE)

assert os.path.isdir(DATA_FPATH)


def iterate_files(only_theis=True):
    """Iterates through the files and yields pandas dataframes.

    Args:
        only_theis: bool, whether or not to include all the data, or only
            use the data from Theis et. al. 2016.

    Yields:
        calcium_pd: the calcium dataframe for the current dataset.
        spikes_pd: the spikes dataframe for the current dataset.
    """

    for i in range(1, 11):
        calcium_fpath = os.path.join(DATA_FPATH, '%d.train.calcium.csv' % i)
        spikes_fpath = os.path.join(DATA_FPATH, '%d.train.spikes.csv' % i)
        assert os.path.exists(calcium_fpath) and os.path.exists(spikes_fpath)

        yield pd.read_csv(calcium_fpath), pd.read_csv(spikes_fpath)

def plot_sample(calcium=None,
                spikes=None,
                t_start=0,
                t_end=100,
                sampling_rate=100):
    """Plots samples of calcium and spikes.

    Args:
        calcium: 1D Numpy float array containing the calcium
            fluorescences (each entry is the average fluorescence
            in that time bin).
        spikes: 1D Numpy float array containing the
            spike data (each entry is the number of spikes
            in that time bin).
        t_start: float (default: 0), the start time, in seconds.
        t_end: float (default: 100), the end time, in seconds.
        sampling_rate: float (default: 100), the sampling rate.
    """
    
    if calcium is None and spikes is None:
        raise ValueError('Must pass at least one of `calcium` or `spikes`.')

    panel = [t_start, t_end]
    x_len = len(calcium) if calcium is not None else len(spikes)
    if t_start >= 0:
        x = np.arange(x_len) / sampling_rate
    else:
        x = (np.arange(x_len) + t_start) / sampling_rate
    if calcium is not None:
        plt.plot(x, calcium, color=(.1, .6, .4))
    if spikes is not None:
        plt.plot(x, spikes / 2.0 - 1, color='k')
    plt.ylim([-3., 3.])
    plt.xlim(panel)
    plt.xlabel('time' + (' (s)' if sampling_rate == 100 else ''))
    plt.grid()


def partition_data(len_pre, len_post=None, spike_n=2,
                   skip=10, iterate=False):
    """Partitions the data into spike-eliciting and non-spike-eliciting.
    
    This will return two lists, the calcium data and the spikes list. The
    calcium data is a list of 1D Numpy arrays, corresponding to the calcium
    fluorescences around the spike in question. The spikes list contains
    boolean values saying whether or not there was a spike in the `spike_n`
    bins directly before or after the current timestep.

    Args:
        len_pre: int, the number of bins before the current spike
            to include.
        len_post: int (default: None, the number of bins after the current
            spike to include. If it isn't set, it defaults to len_pre.
        spike_n: int (default: 2), the number of timesteps forward and
            backward to look for whether or nor a spike was produced.
        skip: int (default: 10), the number of bins to skip on each step.
        iterate: if set, instead of returning lists, iterates through
            the data.
    
    Yields:
        calcium: a 1D Numpy arrays, the calcium fluorescence windows.
        spikes: a 1D Numpy array, the spiking window.
        did_spike: a boolean values, whether or not there was a spike
            associated with that particular calcium window.
    """
    
    if len_post is None:
        len_post = len_pre

    for calcium_df, spikes_df in iterate_files():
        for column in calcium_df:
            calcium_col = calcium_df[column].dropna()
            spikes_col = spikes_df[column].dropna()
            
            # Walk through each column, getting the data window.
            for i in range(len_pre, len(calcium_col) - len_post, skip):
                did_spike = any(spikes_col[i-spike_n:i+spike_n])
                spikes = spikes_col[i-len_pre:i+len_post]
                calcium = calcium_col[i-len_pre:i+len_post]
                
                yield calcium, spikes, did_spike


def load_dataset(cache='spikefinder.data.npz'):
    """Loads the dataset, caching it in `cache`.
    
    This dataset uses parameters that have been set for this particular task.
    These parameters can be altered depending on the application. Each
    timeslice is also normalized to have mean 0.

    Args:
        cache: str (default: "spikefinder.data.npz"), where to cache the
            dataset once it is created.
    """
    
    if cache and os.path.exists(cache):
        data = np.load(cache)
        return data['calcium'], data['did_spike']
    
    calcium_list, did_spike_list = [], []
    iterable = enumerate(partition_data(10, spike_n=2, skip=10))
    for i, (calcium, _, did_spike) in iterable:
        calc_norm = (calcium - np.mean(calcium)) / (np.std(calcium) + 1e-7)
        calcium_list.append(calc_norm)
        did_spike_list.append(did_spike)
        if i % 1000 == 0:
            sys.stderr.write('\rprocessed %d samples' % i)
            sys.stderr.flush()
    sys.stderr.write('\rprocessed %d samples' % len(calcium_list))
    
    calcium_data = np.asarray(calcium_list)
    did_spike_data = np.asarray(did_spike_list)
    
    with open(cache, 'wb') as f:
        np.savez(f, calcium=calcium_data, did_spike=did_spike_data)
    
    return calcium_data, did_spike_data


class DeltaFeature(keras.layers.Layer):
    """Layer for calculating time-wise deltas."""

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('DeltaFeature input should have three '
                             'dimensions. Got %d.' % len(input_shape))
        super(DeltaFeature, self).build(input_shape)

    def call(self, x, mask=None):
        x_a, x_b = K.zeros_like(x[:, 1:]), x[:, :1]
        x_shifted = K.concatenate([x_a, x_b], axis=1)
        return x - x_shifted

    def compute_output_shape(self, input_shape):
        return input_shape


class QuadFeature(keras.layers.Layer):
    """Layer for calculating quadratic feature (square inputs)."""

    def call(self, x, mask=None):
        return K.square(x)

    def compute_output_shape(self, input_shape):
        return input_shape

