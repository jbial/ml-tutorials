"""
Load and prepare text documents for machine learning
Written by Patrick Coady (pcoady@alum.mit.edu)

1. load_book(): Return list of words and word counter for document.
    Also basic document statistics.
2. build_dict(): Build word -> integer dictionary
3. doc2num(): Transform document word list to integer numpy array
4. build_word_array(): Convenience function that runs 3 functions
    above to build an integer numpy word array from a file.
5. save_word_array(): Save a word array and dictionary to file for
    future fast loading.
6. save_word_array(): Load previously saved word array and dictionary.
"""
import collections
import numpy as np
import pickle
import nltk
import re


def load_books(filename):
    """
    Read files and count number of occurrences of each unique word in the
    file. Also return the document as a list of words in the same order
    as the original document.
    Notes:
        The following punctuation are treated as separate words: ;:-()&.,!?'"
        All letters changed to lower-case
        Contractions (e.g. don't, we'll) returned as-is (i.e. ' treated as
            letter). This could cause problems for text that uses single
            quotes (') for other purposes
    :param filename: filename of book to read in
    :return: tuple:
        0) collections.Counter() with unique word counts
        1) list with document words in order
        2) tuples: (number of lines read, number of words read)
    """
    word_counter = collections.Counter()
    word_list = []
    num_lines, num_words = (0, 0)
    in_book = False
    with open(filename, 'r') as f:
        for line in f.readlines():
            if '*** START' in line:
                in_book = True
                continue
            if 'CHAPTER' in line:
                continue
            if '*** END' in line:
                break
            if 'Project Gutenberg' in line:
                continue
            if not in_book:
                continue
            # TODO: check reg-exp below
#             words = nltk.word_tokenize(line.lower().strip())
            words = re.findall("[\\w']+|[;:\-\(\)&.,!?\"]", line.lower().strip('\n'))
#             for i, word in enumerate(words):
#                 words[i] = re.sub("\d", "0", word)
            word_counter.update(words)
            word_list.extend(words)
            num_lines += 1
            num_words += len(words)

    return word_counter, word_list, num_lines, num_words


def build_dict(word_counter, vocab_size):
    """
    Builds dictionary and reverse dictionary of most common words in word_counter.
        Number of words to include in the dictionary is set by dict_size.
    :param word_counter: collections.Counter() with keys = word and values = number of
    occurrences. Case sensitive.
    :param vocab_size: Upper limit on vocabulary size. If number of unique words
        greater than vocab_size, will take most commonly occurring words
    :return: tuple:
        0) dictionary of words to integers (most common word is 0, next most
            common is 1, ...)
        1) reverse dictionary of integers to words (same integer to word mapping as
            "forward dictionary"
    """
    top_words = word_counter.most_common(vocab_size)
    top_words.sort(key=lambda t: -t[1])
    dictionary = dict()
    for idx, word in enumerate(map(lambda t: t[0], top_words)):
        dictionary[word] = idx

    return dictionary


def doc2num(word_list, dictionary):
    """
    Maps list of words to np.array of integers using key/value pairs in
    dictionary. Words not found in dictionary will be mapped to len(dictionary)
    (i.e. 1 larger than biggest value in dictionary).
    :param word_list: List of words
    :param dictionary: Dictionary mapping words to integers
    :return: return numpy array of type np.int32 corresponding to integer mapping
        of words, with words not appearing in dictionary being mapped to
        largest integer in dictionary (i.e. len(dictionary)-1)
    """
    word_array = []
    unknown_val = len(dictionary)
    for word in word_list:
        word_array.append(dictionary.get(word, unknown_val))

    return np.array(word_array, dtype=np.int32)


def build_word_array(filename, vocab_size):
    """
    Convenience function that runs: 1) load_books(), 2) build_dict(),
        and doc2num() in sequence and returns integer word array of documents,
        a dictionary and basic document statistics.
    :param filename: filename of book
    :param vocab_size: Upper limit on vocabulary size. If number of unique words
        greater than vocab_size, will take most commonly occurring words
    :param gutenberg: Set flag to True for .txt files from Project Gutenberg.
        Loader will then skip Gutenberg preamble and license text at end of
        file.
    :return: 3-tuple:
        0) numpy array of type np.int32 corresponding to integer mapping
        of words in documents. Words not in dictionary are mapped to
        largest integer in dictionary (i.e. len(dictionary)-1)
        1) dictionary: word -> int dictionary
        2) 2-tuple: (number of lines read, number of words read)
    Note: no integration coverage
    """
    word_counter, word_list, num_lines, num_words = load_books(filename)
    dictionary = build_dict(word_counter, vocab_size)
    word_array = doc2num(word_list, dictionary)

    return word_array, dictionary, num_lines, num_words


def build_training_set(word_array, window_size):
        """
        Build training set for learning word vectors based on 2 neighboring
        words on each side of a target word. For example, for 'the cat sat on mat'.
        'sat' is the target. 'the', 'cat', 'on' and 'mat' are the training features.
        The data is actually passed to this method as an array of integers, with
        integers representing words.
        :param word_array: Array of integers representing words in a document. The
            array order should match the order in the document (i.e. word_array[0] is
            the first word in the document, word_array[1] is the 2nd word, ...)
        :return: 2-tuple (features, target):
            1. np.array(dtype=np.int32, shape=(N, 4): features (i.e. neighbor words)
            2. np.array(dtype=np.int32, shape=(N, 1): targets (i.e. middle word)
        """
        num_words = len(word_array)
        x = np.zeros((num_words-4, 4), dtype=np.int32)
        y = np.zeros((num_words-4, 1), dtype=np.int32)
        shift = np.array([-2, -1, 1, 2], dtype=np.int32)
        for idx in range(2, num_words-2):
            y[idx-2, 0] = word_array[idx]
            x[idx-2, :] = word_array[idx+shift]

        return x, y


def save_word_array(filename, word_array, dictionary):
    """
    Save word array and dictionary for faster load.
    :param filename: Filename, with path. Saved as python pickle file.
    :param word_array: Numpy integer word array of document
    :param dictionary: Word -> int document
    :return: None
    Note: no unit test coverage
    """
    word_array_dict = dict()
    word_array_dict['word_array'] = word_array
    word_array_dict['dictionary'] = dictionary
    with open(filename + '.p', 'wb') as f:
        pickle.dump(word_array_dict, f)


def load_word_array(filename):
    """
    Load integer word array and dictionary saved by save_word_array()
    :param filename: Same filename used with save_word_array()
    :return: 2-tuple
        0) Numpy word array of integers (document representation)
        1) Word -> int dictionary
    Note: no unit test coverage
    """
    with open(filename + '.p', 'rb') as f:
        word_array_dict = pickle.load(f)

    return word_array_dict['word_array'], word_array_dict['dictionary']