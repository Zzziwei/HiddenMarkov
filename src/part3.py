#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from tqdm import tqdm

from helpers import *
from part2 import get_emission_parameters


def get_transition_parameters(state_sequences):
    """
    Estimate transition paramters from a collection of state sequences
    """
    n_states = max(flatten(state_sequences)) + 1  # State space size (Excluding START and STOP)
    transition_matrix = np.zeros((n_states + 1, n_states + 1))

    for state_sequence in state_sequences:
        transition_matrix[0, state_sequence[0]] += 1
        transition_matrix[state_sequence[-1] + 1, n_states] += 1
        for i in range(len(state_sequence)):
            transition_matrix[state_sequence[i - 1] + 1, state_sequence[i]] += 1

    transition_matrix = (transition_matrix.T / transition_matrix.sum(axis=1)).T
    return transition_matrix


def viterbi(observations, log_transition_matrix, log_emission_matrix):
    start_p = log_transition_matrix[0]
    log_transition_matrix = log_transition_matrix[1:]

    n_states = len(log_transition_matrix)
    n_observations = len(observations)
    states = list(range(n_states))

    V_prob = np.full((len(observations), len(states)), -np.inf)
    V_prev = np.full((len(observations), len(states)), None)
    V = [[[-float("inf"), None] for _ in states] for _ in observations]

    # First layer
    for state in states:
        V[0][state] = [start_p[state] + log_emission_matrix[state, observations[0]], None]

    for t in range(1, len(observations)):  # Exclude first observation
        for state in states:
            max_tr_prob = V[t - 1][states[0]][0] + log_transition_matrix[states[0], state]
            prev_state_selected = states[0]
            for prev_state in states[1:]:
                tr_prob = V[t - 1][prev_state][0] + log_transition_matrix[prev_state, state]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_state_selected = prev_state
            max_prob = max_tr_prob + log_emission_matrix[state, observations[t]]
            V[t][state] = max([V[t - 1][y0][0] + log_transition_matrix[y0, state] + log_emission_matrix[state, observations[t]], y0] for y0 in states)

    opt = []
    max_prob = max(value[0] for value in V[-1])
    prev = None
    for state, value in enumerate(V[-1]):
        if value[0] == max_prob:
            opt.append(state)
            prev = state
            break
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][prev][1])
        prev = V[t + 1][prev][1]
    return opt


def predict_and_save(train_file, test_file, output_file):
    features, labels = load_dataset(train_file)
    smoothed_features = smooth(features, 3)  # Input feature smoothing

    # Numerically encode dataset
    feature_ids, feature_map = encode_numeric(smoothed_features)
    label_ids, label_map = encode_numeric(labels)

    # Get HMM model parameters
    emission_matrix = get_emission_parameters(feature_ids, label_ids)
    transition_matrix = get_transition_parameters(label_ids)

    # Load dev dataset, smooth and numerically encode it
    dev_features = load_dataset(test_file, split=False)
    smoothed_dev_features = smooth_dev(dev_features, smoothed_features)
    dev_feature_ids, _ = encode_numeric(smoothed_dev_features, token_map=feature_map)  # Make sure to reuse the same token map as the training set

    # Run Viterbi algorithm to get most likely labels
    print("Running Viterbi algorithm...")
    predicted_dev_labels = [viterbi(feature_id, log(transition_matrix), log(emission_matrix)) for feature_id in tqdm(dev_feature_ids)]

    # Write predictions to file
    print("Writing to file...")
    predicted_dev_labels = decode_numeric(predicted_dev_labels, label_map)
    with open(output_file, "w") as outfile:
        for dev_feature_sequence, predicted_dev_label_sequence in zip(dev_features, predicted_dev_labels):
            for dev_feature, predicted_dev_label in zip(dev_feature_sequence, predicted_dev_label_sequence):
                print(dev_feature, predicted_dev_label, file=outfile)
            print(file=outfile)


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': lambda x: f"{x:0.3f}"})
    DATA_ROOT = Path("data")
    DATASETS = ["SG", "CN", "EN", "AL"]
    for dataset in DATASETS:
        print(f"Dataset: {dataset}")
        predict_and_save(f"data/{dataset}/train", f"data/{dataset}/dev.in", f"data/{dataset}/dev.p3.out")
