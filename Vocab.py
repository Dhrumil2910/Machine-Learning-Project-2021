# -*- coding: utf-8 -*-
"""

"""

import collections

import utils
import preprocess

import tqdm


class Vocabulary:
    
    # Special Tokens:
    #
    #   <PAD>: The PAD token is used to pad sequences to match the
    #          longest sequence in the batch
    #   <UNK>: The UNK token is used to represent tokens that are not
    #          present in the Vocabulary.
    #   <SOS>: Used to represent Start of Sequence
    #   <EOS>: Used to represent End of Sequence
    PAD = "<PAD>"
    UNK = "<UNK>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    
    specials = [PAD, UNK, SOS, EOS]

    def __init__(self, max_count):
        """Initialize the Vocabulary
        
        Parameters
        ----------
        max_count : int
            The maxixmum size of the Vocabulary.

        """

        # The is_trimmed Flag is used to indicate whether the Vocabulary
        # is trimmed or not.
        # The number of words in a trimmed Vocabulary is equal to the max_count.
        self.trimmed = False

        # The maximum number of words in the vocabulary.
        # It is used by the trim method to trim the vocabulary to have
        # max_count words.    
        self.max_count = max_count

        # The counter keeps track of frequency of the words that appear in the
        # dataset.
        # The trim method uses this counter to trim the words that have a 
        # frequency less than min_count.
        self.counter = collections.Counter()
        

        self._initialize_dicts()


    def __getitem__(self, word):
        """Returns the index of the word.
        
        Usage:
            vocab = Vocabulary()
            word_index = vocab["Foobar"]
        
        Returns 1 (i.e. the index of the <UNK> token) if the word is not 
        present in the Vocabulary.

        Parameters
        ----------
        word : string
            A word that is present in the Vocabulary.

        Returns
        -------
        int
            The index of the the word. Returns 1 if word is not in the Vocabulary.

        """
        return self.stoi.get(word, self.stoi.get(Vocabulary.UNK))


    @property
    def vocab_size(self):
        """Returns the size of the Vocabulary

        Returns
        -------
        int
            The size of the Vocabulary.

        """
        return len(self.stoi)


    def _initialize_dicts(self):
        """Initializes the dictionaries.
        
        This method initializes the stoi (string to index) and the
        itos (index to string) dictionaries.
        
        The stoi dictionary maps the words to its index and the itos is the
        reverse mapping of index to the word.
        
        """
        
        # Used by defaultdict to return the index of the <UNK> token
        # if the word is not in the Vocabulary
        default_index = lambda : 1

        # The stoi dictionary maps the words to its index. The stoi is an
        # instance of DefaultDict. So, if the word is not present 
        # in the Vocabulary it returns the index of the <UNK> token 
        # insted of raising a KeyError.
        # self.stoi = collections.defaultdict(default_index)
        self.stoi = {}
        
        # The reverse mapping of index to the word.
        self.itos = {}
        
        # The word_count is used to keep track of the current index 
        # which is used when building the Vocabulary
        self.word_count = 0


    def _add_specials(self):
        """Adds special tokens to the Vocabulary

        """
        for token in Vocabulary.specials:

            # Add Special tokens
            self.add_word(token)


    def count_word(self, word):
        """Used to keep track of the frequency of the word in the Dataset.
        
        The count_word method updates the counter when building the 
        Vocabulary.

        If the word is not present in the counter, it adds the word to the 
        counter and sets its frequency to 1.

        If the word is already present in the counter, it just updates its
        frequency by 1.
        
        See Vocabulary.build method for more details.

        Parameters
        ----------
        word : string
            A word that is present in the Dataset.

        """
        self.counter.update([word])


    def add_word(self, word):
        """Adds a word to the Vocabulary

        Parameters
        ----------
        word : string
            A word to be added to the Vocabulary that is present in the Dataset.

        """
        
        # Skip if the word is already in the Dataset
        if word not in self.stoi:
            
            # Add the word with its index to the string to index dictionary 
            # and keep its reverse mapping in the index to string dictionary.
            self.stoi[word] = self.word_count
            self.itos[self.word_count] = word
            self.word_count += 1


    def trim(self, min_count=0):
        """Trims the Vocabulary so that it has max_count words.
        
        Parameters
        ----------
        min_count : int, optional
            The minimum frequency of the word in the Dataset. The default is 1.

        """
        
        # Return if the Vocabulary is already trimmed
        if self.trimmed:
            return
        self.trimmed = True

        # Keeps a list of words to be added to the Vocabulary
        keep_words = []
        
        # Get the most top max_count words from the counter.
        for word, count in tqdm.tqdm(self.counter.most_common(self.max_count - len(Vocabulary.specials)), total=20000):
            
            # Skip if the frequency of the word is less than min_count
            if count > min_count:
                keep_words.append(word)

        # Build the Vocabulary.
        self._initialize_dicts()
        self._add_specials()

        for word in keep_words:
            self.add_word(word)
            
    
    def article_to_ids(self, article_tokens):
        
        article_ids = []
        article_ids_ext = []
        oovs = []
        
        unk_id = self.stoi[Vocabulary.UNK]
        
        for token in article_tokens:

            token_id = self.stoi.get(token, unk_id)
            article_ids.append(token_id)
            
            if token_id == unk_id:
                

                if token not in oovs:
                    oovs.append(token)

                token_id = self.vocab_size + oovs.index(token)

            article_ids_ext.append(token_id)
        
        return article_ids, article_ids_ext, oovs
                
    
    def summary_to_ids(self, summary_tokens, oovs):
        
        summary_ids = []
        
        unk_id = self.stoi[Vocabulary.UNK]
        
        for token in summary_tokens:
            token_id = self.stoi.get(token, unk_id)
            
            if token_id == unk_id:
                
                if token in oovs:
                    token_id = self.vocab_size + oovs.index(token)
                else:
                    token_id = unk_id
                    
            summary_ids.append(token_id)
            
        return summary_ids
    
    
    def output_to_words(self, idices, oovs):
        
        output_summary = []
        
        for _id in idices:
            
            try:
                word = self.itos[_id]
            except KeyError:
                
                try:
                    oov_index = _id - self.vocab_size
                    word = oovs[oov_index]
                except IndexError:
                    word = Vocabulary.UNK
                    
            output_summary.append(word)
        
        return output_summary
    
    @property
    def pad_index(self):
        return self.stoi.get(Vocabulary.PAD)
                
    @property
    def unk_index(self):
        return self.stoi.get(Vocabulary.UNK)

    
    @classmethod
    def build(cls, datagen, vocab_size, save_path=None, min_count=3):
        """Builds the Vocabulary and returns the Vocabulary object.
        
        The classmethod takes in a generator which yields a single item at
        a time. It then preprocesses the item to generate text tokens and then
        adds these tokens to the Vocabulary.

        Parameters
        ----------
        cls : class
            Then name of the class. Here Vocabulary. This is Passed by default.

        datagen : generator
            A Generator which yields the Articles in the dataset one at a time.

        vocab_size : int
            The Maximum size of the Vocabulary.

        save_path : string, optional
            The path where the Vocabulary should be saved. The default is None.
        
        min_count : int, optional
            The minimum frequency of the word in the Dataset. The default is 1.

        Returns
        -------
        vocab : object
            An Instance of Vocabulary class.

        """
        
        # Create a Vocabulary instance
        vocab = cls(vocab_size)
        
        for article, summary in tqdm.tqdm(datagen, total=50000):
            
            
            # Preprocess the article and summary to get the tokens
            article_tokens = preprocess.preprocess_text(article)
            summary_tokens = preprocess.preprocess_text(summary)
            
            
            # Add the tokens to the Vocabulary
            for token in article_tokens:
                vocab.count_word(token)
                
            for token in summary_tokens:
                vocab.count_word(token)
    
    
    
        # Trim the Vocabulary to have words equal to vocab_size.
        vocab.trim(min_count)
    
        # Save the Vocabulary object to the disk
        if save_path is not None:
            utils.pickle_data(vocab, save_path)
    
        return vocab
    
    @staticmethod
    def load_vocabulary(filepath):
        vocab = utils.load_pickled_data(filepath)
        return vocab


if __name__ == "__main__":
    
    path = "./WD/test.gz"
    dataset_gen = utils.load_dataset(path)
    
    vocab = Vocabulary.build(dataset_gen, 20000, "./vocab.pkl")
