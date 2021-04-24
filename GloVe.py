# -*- coding: utf-8 -*-
"""
Create and load Glove Embeddings

This module creates a mapping of words to their embeddings from the GloVe file.
The mappings created by this function is used by the Vocabulary class
to create a embedding matrix. This embedding matrix can be used to 
initialize the Embedding Layer of the Neural Network.

The "create_glove_mappings" function creates a dictionary is used to map 
a word in the GloVe file to its word embedding which is stored
as a numpy array of size (1, embedd_dim)
"""


import time
import pickle

import numpy as np

from tqdm import tqdm


def create_glove_mappings(GloVe_PATH, embedd_dim=50):
    """Create a mapping of words to their embeddings from the GloVe file.
    
    The mappings created by this function is used by the Vocabulary class
    to create a embedding matrix. This embedding matrix can be used to 
    initialize the Embedding Layer of the Neural Network.

    Parameters
    ----------
    GloVe_PATH : string
        The Path to the GloVe embeddings folder.

    embedd_dim : int, optional
        The dimension size of the embeddings to use. The default is 50.

    Returns
    -------
    stev : dict
        A dictionary mapping the words to its embedding.

    """
    
    # 1. Initialize Variables
    
    # The total number of tokens in the GloVe files
    # This variable is used by tqdm to diplsay progress bar
    GloVe_SIZE = 400000
    
    
    # The stev or String to Embedding Vector dictionary is used to map 
    # a word in the GloVe file to its word embedding which is stored
    # as a numpy array of size (1, embedd_dim)
    stev = {}
    

    glove_path = f'{GloVe_PATH}/glove.6B/glove.6B.{embedd_dim}d.txt'
    
    with open(glove_path, 'r', encoding="utf8") as f:
        for idx, line in enumerate(tqdm(f, total=GloVe_SIZE)):

            # Split the line and assign first value to word and
            # the rest to the embedding variable
            # e.g. First three embedding dim values for the word "the"
            #   the 0.418 0.24968 -0.41242
            #
            #   word = "the"
            #   embedding = [0.418, 0.24968, -0.41242]
            word, *embedding = line.split()


            # Convert the embeddint into numpy array
            #
            # Then ndmin is set to 2 so that the embedding_vector has a shape
            # of (1, embedd_dim) and not (embedding_vector,)
            embedding_vector = np.array(embedding, ndmin=2).astype(np.float)
            
            
            stev[word] = embedding_vector

    # Save the String to Embedding Vector dictionary to disk
    stev_path = f'{GloVe_PATH}/vector_cache/glove{embedd_dim}d.pkl'
    pickle.dump(stev, open(stev_path, "wb"))
    
    return stev


def load_glove(GloVe_PATH, embedd_dim=50):
    """Load the String to Embedding Vector dictionary from disk
    

    Parameters
    ----------
    GloVe_PATH : string
        The Path to the GloVe embeddings folder.

    embedd_dim : int, optional
        The dimension size of the embeddings to use. The default is 50.

    Returns
    -------
    dict
        A dictionary mapping the words to its embedding.

    """
    stev_path = f'{GloVe_PATH}/vector_cache/glove{embedd_dim}d.pkl'
    return pickle.load(open(stev_path, "rb"))


def generate_word_embeddings_matrix(vocab, glove, embedd_dim=50):
    """Generate Weight Matrix.
    
    This function generates a weight matrix of size (vocab_size, embedd_dim) 
    for initializing the Embedding layer in the Neural Network.
    
    The weight matrix is initialized with the GloVe Embeddings at the
    corrosponding index of the word in the Vocabulary has a GloVe Embedding.
   
    If the word does not have a GloVe Embedding, the  corrosponding index is
    initialized with a random vector with Normal Distribution which has
    a mean of 0 and a standard deviation of 1/sqrt(vocab_size)

    Parameters
    ----------
    vocab : object
        An object of the Vocab class.
        
    glove : dict
         A dictionary mapping the words to its GloVe embedding.
         
    embedd_dim : int, optional
        The dimension size of the embeddings to use. The default is 50.

    Returns
    -------
    weights_matrix : numpy.ndarray
        Weights Matrix of size (vocab_size, embedd_dim) which can be used 
        to initialize the Embedding layer in a Neural Network.
        
    words_found : int
        The number of words in the Vocabulary which have GloVe embeddings.

    """
    
    
    # Initialize the Embeddings Matrix
    weights_matrix = np.zeros((vocab.vocab_size, embedd_dim))
    words_found = 0

    # Iterate over all the words present in the Vocabulary
    for word, index in vocab.stoi.items():
        
        try:
            # Assign the i-th position of the weights matrix with the GloVe 
            # embedding

            weights_matrix[index] = glove[word]
            words_found += 1

        except KeyError:
            
            # Word does not have a GloVe Embedding.
            #
            # Initialize with a random vector with Normal Distribution
            # with a mean of 0 and a standard deviation of 1/sqrt(vocab_size)

            scale = 1 / np.sqrt(vocab.vocab_size)
            size = (embedd_dim,)
    
            weights_matrix[index] = np.random.normal(scale=scale, size=size)

    return weights_matrix, words_found

if __name__ == "__main__":
    
    GloVe_PATH = "../Datasets/GloVe"

    GloVe_DIM = 100

    # glove = create_glove_mappings(GloVe_PATH, GloVe_DIM)
    glove = load_glove(GloVe_DIM)

    start_time = time.time()
    tokens = glove["hello"]
    end_time = time.time()
    print("\n\nTime Elapsed: {:15f}".format(end_time - start_time))
