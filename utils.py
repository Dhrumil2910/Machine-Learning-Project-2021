# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 17:59:34 2021

@author: reuel
"""

import gzip
import json
import pickle

def load_dataset(path, count=None):
    
    with gzip.open(path) as train_file:
        for idx, entry in enumerate(train_file):
            article = json.loads(entry)
            
            if count is not None and idx == count :
                break
            
            yield article["text"], article["summary"]
            
def pickle_data(data, filepath):
    """Pickle Data to disk
    
    Parameters
    ----------
    data :
        Data that is to be pickled.
    filepath : str
        The location where to save the pickled file.

    """
    pickle.dump(data, open(filepath, "wb"))


def load_pickled_data(filepath):
    """Load the pickled data

    Parameters
    ----------
    filepath : str
        The location where to save the pickled file is saved on disk.

    Returns
    -------
    object
        An object that was pickled.

    """
    return pickle.load(open(filepath, "rb"))
