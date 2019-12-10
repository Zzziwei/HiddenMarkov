#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from tqdm import tqdm

from helpers import *


def get_emission_parameters(observations, states):
    """
    Estimate emission paramters from a collection of observation-state pairs
    """
    n_observations = max(flatten(observations)) + 1  # Observation space size
    n_states = max(flatten(states)) + 1  # State space size
    emission_matrix = np.zeros((n_states, n_observations))
    #     emission_matrix = [[0 for _ in range(n_observations)] for _ in range(n_states)]

    for state, obs in zip(states, observations):
        for s, o in zip(state, obs):
            emission_matrix[s, o] += 1

    emission_matrix /= emission_matrix.sum(axis=0)
    #     for i in range(n_states):
    #         row_sum = sum(emission_matrix[i])
    #         for j in range(n_observations):
    #             emission_matrix[i][j] /= row_sum

    return emission_matrix


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
        return 1 / (1 + symbol_counts[symbol])
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
                scores_and_symbols = [
                    (
                        emission_probability(
                            symbol, word, emission_probabilities, symbol_counts
                        ),
                        symbol,
                    )
                    for symbol in symbols
                ]
                argmax = max(
                    scores_and_symbols, key=lambda score_and_symbol: score_and_symbol[0]
                )[1]
                predicted_word_symbol_sequence.append((word, argmax))
            else:
                predicted_word_symbol_sequence.append(("", ""))

    return predicted_word_symbol_sequence


def predict_and_save(train_file, test_file, output_file):
    features, labels = load_dataset(train_file)
    smoothed_features = smooth(features, 3)  # Input feature smoothing

    # Numerically encode dataset
    feature_ids, feature_map = encode_numeric(smoothed_features)
    label_ids, label_map = encode_numeric(labels)

    # Get emission matrix
    emission_matrix = get_emission_parameters(feature_ids, label_ids)

    # Load dev dataset, smooth and numerically encode it
    dev_features = load_dataset(test_file, split=False)
    smoothed_dev_features = smooth_dev(dev_features, smoothed_features)
    dev_feature_ids, _ = encode_numeric(
        smoothed_dev_features, token_map=feature_map
    )  # Make sure to reuse the same token map as the training set

    # Get most likely labels
    print("Predicting labels...")
    predicted_dev_labels = [
        np.argmax(emission_matrix[:, feature_id], axis=0)
        for feature_id in tqdm(dev_feature_ids)
    ]
    predicted_dev_labels = decode_numeric(predicted_dev_labels, label_map)

    # Write predictions to file
    print("Writing to file...")
    with open(output_file, "w") as outfile:
        for dev_feature_sequence, predicted_dev_label_sequence in zip(
            dev_features, predicted_dev_labels
        ):
            for dev_feature, predicted_dev_label in zip(
                dev_feature_sequence, predicted_dev_label_sequence
            ):
                print(dev_feature, predicted_dev_label, file=outfile)
            print(file=outfile)


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": lambda x: f"{x:0.3f}"})
    DATA_ROOT = Path("data")
    DATASETS = ["SG", "CN", "EN", "AL"]
    for dataset in DATASETS:
        print(f"Dataset: {dataset}")
        predict_and_save(
            f"data/{dataset}/train",
            f"data/{dataset}/dev.in",
            f"data/{dataset}/dev.p2.out",
        )
