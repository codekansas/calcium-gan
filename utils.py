"""Some utils for loading data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import urllib

import numpy as np
import pandas as pd

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
