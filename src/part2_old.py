# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from collections import defaultdict

symbols = ['O', 'B-positive', 'I-positive', 'B-neutral', 'I-neutral', 'B-negative', 'I-negative']

def get_symbol_word_counts(training_file):

    # initialize dictionary of counts
    symbol_word_counts = {}
    for symbol in symbols:
        symbol_word_counts[symbol] = defaultdict(int)

    with open(training_file, encoding="utf8") as f:
        for line in f:
            if line.isspace():
                continue
            word = line.split(' ')[0].strip()
            symbol = line.split(' ')[-1].strip()
            symbol_word_counts[symbol][word] += 1

    # find total symbol counts
    symbol_counts = {}
    for symbol in symbol_word_counts:
        symbol_counts[symbol] = sum(symbol_word_counts[symbol].values())

    return symbol_word_counts, symbol_counts

def estimate_emission_params(symbol_word_counts, symbol_counts):
    """
    Returns a nested dictionary of emission probabilities
    where the value of d[symbol][word] is the emission probability
    for that word and symbol.
    If the word is unseen in the training data used to produce the
    symbol_word_counts and symbol_counts, trying to access it from
    the dictionary will raise a KeyError
    """

    emission_probabilities = {}
    for symbol in symbol_word_counts:
        emission_probabilities[symbol] = {}
        for word in symbol_word_counts[symbol]:
            emission_probabilities[symbol][word] = float(symbol_word_counts[symbol][word])/(symbol_counts[symbol] + 1)

    return emission_probabilities

def get_emission_probabilities(training_file):
    """
    Returns a nested dictionary of emission probabilities
    from a given training file
    """

    symbol_word_counts, symbol_counts = get_symbol_word_counts(training_file)
    return estimate_emission_params(symbol_word_counts, symbol_counts)

def emission_probability(symbol, word, emission_probabilities, symbol_counts):
    """
    Takes a symbol, a word, a nested dictionary of emission probabilities,
    and a dictionary of symbol counts
    and returns the emission probability for that symbol and word
    If the word has not been encountered in the training data
    we assign it a fixed probability based on the symbol count
    """

    unseen_word = True

    for sym in emission_probabilities:
        if word in emission_probabilities[sym]:
            unseen_word = False

    if unseen_word:
        return 1/(1 + symbol_counts[symbol])
    else:
        if word in emission_probabilities[symbol]:
            return emission_probabilities[symbol][word]
        else:
            return 0

def find_symbol_estimate(dev_file, emission_probabilities, symbol_counts):
    predicted_word_symbol_sequence = []
    with open(dev_file, encoding="utf8") as f:
        for line in f:
            if not line.isspace():
                word = line.strip()
                scores_and_symbols = [(emission_probability(symbol, word, emission_probabilities, symbol_counts), symbol) for symbol in symbols]
                argmax = max(scores_and_symbols, key=lambda score_and_symbol: score_and_symbol[0])[1]
                predicted_word_symbol_sequence.append((word, argmax))
            else:
                predicted_word_symbol_sequence.append(('',''))

    return predicted_word_symbol_sequence

def write_part_2_dev_out(filename, predicted_word_symbol_sequence):
    result_file = open(filename, "w", encoding="utf8")

    for word_and_symbol in predicted_word_symbol_sequence:
        result_file.write(' '.join(word_and_symbol) + "\n")