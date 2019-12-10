#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from tqdm import tqdm
import heapq

from helpers import *
from part2 import get_emission_parameters


def get_transition_parameters(state_sequences):
    """
    Estimate transition paramters from a collection of state sequences
    """
    n_states = (
        max(flatten(state_sequences)) + 1
    )  # State space size (Excluding START and STOP)
    transition_matrix = np.zeros((n_states + 1, n_states + 1))

    for state_sequence in state_sequences:
        transition_matrix[0, state_sequence[0]] += 1
        transition_matrix[state_sequence[-1] + 1, n_states] += 1
        for i in range(len(state_sequence)):
            transition_matrix[state_sequence[i - 1] + 1, state_sequence[i]] += 1

    transition_matrix = (transition_matrix.T / transition_matrix.sum(axis=1)).T
    return transition_matrix


def viterbi(observations, log_transitions, log_emissions, k=1):
    log_transitions = log_transitions.tolist()
    start_p = log_transitions.pop(0)

    n_states = len(log_transitions)
    n_obs = len(observations)
    states = tuple(range(n_states))

    # Default values for viterbi-table
    # The first item in the innermost list is the actual log-score.
    # The second is the "winning" state path that created this score.
    # You need the top k instead of the top 1 now
    V = [[[] for state in states] for _ in range(n_obs)]

    # Base case
    for state in states:
        # Not that many paths we can work with here
        V[0][state].append(
            (
                # This is the score
                start_p[state] + log_emissions[state, observations[0]],
                # This is the list of states that led here.
                # Technically this should be [START]
                tuple(),
            )
        )

    # For each observation...
    for t in range(1, n_obs):
        # For each state...
        for state in states:
            # A heap that holds the scores
            h = []

            # Grab ALL the previous score-path pairs
            for prev_state, prev_state_scores in enumerate(V[t - 1]):
                for i, entry in enumerate(prev_state_scores):
                    prev_score, prev_state_path = entry

                    # Calculate a new score from previous state to the current one
                    new_score = (
                        prev_score
                        + log_transitions[prev_state][state]
                        + log_emissions[state, observations[t]]
                    )
                    # Also save the chain of states that generated this score
                    new_state_path = tuple(prev_state_path) + (prev_state,)
                    h.append((new_score, new_state_path))
                    # heapq.heappush(h, (new_score, new_state_path))

            # At this point, for regular viterbi just take the max item from the heap.
            # But this isn't regular viterbi
            V[t][state] = sorted(h, reverse=True)[:k]

    # We did a lot of the heavy lifting before
    opt = []
    out = []
    for last_state, entries in enumerate(V[-1]):
        for e in entries:
            heapq.heappush(opt, [e, last_state])
    for (score, state_path), last_state in heapq.nlargest(k, opt):
        out.append(tuple(state_path) + (last_state,))

    return out



def predict_and_save(train_file, test_file, output_file, k=1):
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
    dev_feature_ids, _ = encode_numeric(
        smoothed_dev_features, token_map=feature_map
    )  # Make sure to reuse token map from the training set

    # Run Viterbi algorithm to get most likely labels
    print("Running Viterbi algorithm...")
    predicted_dev_labels = []
    for feature_id in tqdm(dev_feature_ids):
        pred = viterbi(feature_id, log(transition_matrix), log(emission_matrix), k=k)[-1]
        predicted_dev_labels.append(pred)

    # Write predictions to file
    print("Writing to file...")
    predicted_dev_labels = decode_numeric(predicted_dev_labels, label_map)
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
            f"data/{dataset}/dev.p3.out",
            k=1,
        )
