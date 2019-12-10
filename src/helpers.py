"""
Helper functions
"""
import collections
import itertools
import random
from pathlib import Path
import numpy as np


def load_dataset(path, split=True, shuffle=False):
    """
    Load a dataset from a specified path.

    Args:
        path: The path to read the data from
        split (bool): Whether to split labels from each line of data
    """
    with open(path) as f:
        sequences = [sent.split("\n") for sent in f.read().split("\n\n")][:-1]
    if shuffle:
        random.shuffle(sequences)
    if split:
        sequences = [[pair.split() for pair in seq] for seq in sequences]
        sequences = [[[pair[i] for pair in s] for s in sequences] for i in [0, 1]]
    return sequences


def log(x, inf_replace=-100):
    out = np.log(x)
    out[~np.isfinite(out)] = inf_replace
    return out


def flatten(sequences):
    """
    Flatten a nested sequence
    """
    return itertools.chain.from_iterable(sequences)


def unique(sequences, sort=True):
    items = set(flatten(sequences))
    if sort:
        items = sorted(items)
    return items


def count(sequences, as_probability=False):
    """
    Get a dictionary of word-count pairs in a dataset.

    Args:
        sequences: The sequence (or collection of sequences) of words to count
        as_probability (bool): Whether to return the counts as probabilties (over the entire dataset)
    """
    counts = dict(collections.Counter(flatten(sequences)))
    if as_probability:
        counts = {k: v / sum(counts.values()) for k, v in counts.items()}
    return counts


def smooth(inputs, thresh):
    """
    Replace tokens appearing less than `thresh` times with a "#UNK#" token.

    Args:
        inputs: The collection of sequences to smooth
        thresh (bool): The minimum number of occurrences required for a word to not be replaced
    """
    inputs = list(inputs)
    to_replace = {
        k for k, v in count(inputs, as_probability=False).items() if v < thresh
    }
    return [["#UNK#" if x in to_replace else x for x in sub] for sub in inputs]


def smooth_dev(sequences, train_sequences):
    """
    For each token in the given inputs, replace it with "#UNK#" if it doesn't appear in the training corpus.
    """
    train_sequences = unique(train_sequences, sort=False)
    return [
        [x if x in train_sequences else "#UNK#" for x in sequence]
        for sequence in sequences
    ]


def get_token_map(sequences):
    """
    Get token_to_id and id_to_token maps from a collection of sequences
    """
    tokens = unique(sequences)
    return {token: i for i, token in enumerate(tokens)}


def encode_numeric(sequences, token_map=None):
    """
    Encode a collection of token sequences as numerical values
    """
    token_map = token_map or get_token_map(
        sequences
    )  # Compute token map if not provided
    return (
        [[token_map[token] for token in sequence] for sequence in sequences],
        token_map,
    )


def decode_numeric(sequences, token_map):
    """
    Decode a collection of token ID sequences to tokens
    """
    token_map = {token: i for i, token in token_map.items()}  # Reverse token map
    return [[token_map[val] for val in sequence] for sequence in sequences]
