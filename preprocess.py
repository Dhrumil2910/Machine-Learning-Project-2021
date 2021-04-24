# -*- coding: utf-8 -*-
"""
Pre-Process Text

Steps:
    1. Strip newline and linefeed characters
    2. Expand Contractions
    3. Remove Accented Characters
    4. Tokenize
    5. POS Tagging
    6. Remove tokens that are Punctuations
    7. Lemmatize tokens & convert token to lower case
"""

import re
import json
import time
import string

import nltk
import spacy
import unidecode
import contractions

from nltk import tokenize
from nltk.corpus.reader import wordnet


def strip_newlines(text):
    """Removes newlines and linefeeds in a document
    
    This Function removes the "\n\n" characters that occur
    in the Text Article and the "\r\n" characters in the summary.
    
    The function is specific to the Cornell Newsroom Dataset.

    
    Parameters
    ----------
    text : string
        A Text Article or a Summary.

    Returns
    -------
    string
        The text without "\n\n" and "\r\n" characters.

    """
    return re.sub("[\n|\r]\n", r" ", text)


def expand_contractions(text):
    """Expands contractions in the text
    
    This function uses the contractions(https://github.com/kootenpv/contractions) 
    package to remove any contractions from the text.
    
    e.g.
        original text:  I'm standing in the shadow of an equestrian statue.
        output:  I am standing in the shadow of an equestrian statue.
        
        original text:  We're not meant to be here.
        output: We are not meant to be here.


    Parameters
    ----------
    text : string
        A Text Article or a Summary.

    Returns
    -------
    string
        The text without the contractions.

    """
    return contractions.fix(text)


def remove_accented_chars(text):
    """Remove accented characters from text, e.g. café
    

    Parameters
    ----------
    text : string
        A Text Article or a Summary.

    Returns
    -------
    string
        The text without the accented characters.

    """
    return unidecode.unidecode(text)


def preprocess_text(text):
    """Main function to pre-process text.
    
    Pre-processing Steps:
        1. Strip newline and linefeed characters
        2. Expand Contractions
        3. Remove Accented Characters
        4. Tokenize
        5. POS Tagging
        6. Remove tokens that are Punctuations
        7. Lemmatize tokens & convert token to lower case
    

    Parameters
    ----------
    text : string
        A Text Article or a Summary.

    Returns
    -------
    tokens : list of strings
        A list of strings of the tokens generated when the above
        pre-processing steps are appplied to the input text.

    """
    
    # Step 1. Strip newline and linefeed characters
    text = strip_newlines(text)
    
    # Step 2. Expand Contractions
    # text = expand_contractions(text)
    
    # Step 3. Remove Accented Characters
    text = remove_accented_chars(text)
    
    # Add a token signifying the start of the sequence
    tokens = ["<SOS>"]
    
    # Step 4. Tokenize
    words = tokenize.word_tokenize(text)
    
    # Initialize the WordNet Lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    # Step 5. POS Tagging
    pos_tagged_tokens = nltk.pos_tag(words)
    
    for word, tag in pos_tagged_tokens:
        
        # Step 6. Filter out punctuations
        if not word in string.punctuation:
            tag = tag[0].lower()
            tag = tag if tag in wordnet.POS_LIST else wordnet.NOUN
            
            # Step 7. Lemmatize & convert to lower casae
            lemma = lemmatizer.lemmatize(word, pos=tag)
            lemma = lemma.lower()
            tokens.append(lemma)
        
    # Add a token signifying the end of the sequence
    tokens.append("<EOS>")
    
    return tokens
        

def preprocess_text_spacy(text, nlp):
    """Pre-process text using SpaCy
    
    Pre-processing Steps:
        1. Strip newline and linefeed characters
        2. Expand Contractions
        3. Remove Accented Characters
        4. Tokenize
        5. POS Tagging
        6. Remove tokens that are Punctuations
        7. Lemmatize tokens & convert token to lower case
    

    Parameters
    ----------
    text : string
        A Text Article or a Summary.

    Returns
    -------
    tokens : list of strings
        A list of strings of the tokens generated when the above
        pre-processing steps are appplied to the input text.

    """
    
     # Step 1. Strip newline and linefeed characters
    text = strip_newlines(text)
    
    # Step 2. Expand Contractions
    # text = expand_contractions(text)
    
    # Step 3. Remove Accented Characters
    text = remove_accented_chars(text)
    
    tokens = ["<SOS>"]
    
    # Step 4. & 5. Tokenize and POS Tag
    doc = nlp(text)
    
    for token in doc:
        
        # Step 6. Filter out punctuations
        if not token.text in string.punctuation:
            
            # Step 7. Lemmatize & convert to lower casae
            tokens.append(token.lemma_.lower())
     
    # Add a token signifying the end of the sequence
    tokens.append("<EOS>")
        
    return tokens
    

if __name__ == "__main__":
    
    # path = "./WD/test.json"

    # with open(path, "r") as f:
    #     article = json.load(f)
        
    # text = article["text"]
    text = "tell me something who'll win? I'd think that it'll be mets won't it? John's betting on it. Let's go to John's house."
    
    # nlp = spacy.load("en_core_web_sm")
    
    start = time.time()
    tokens = preprocess_text(text)
    # tokens = preprocess_text_spacy(text, nlp)
    end = time.time()
    
    print(f"Time Taken to pre-process one article: {end - start}")
